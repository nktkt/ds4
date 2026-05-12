//! Metal graph executor. Ports the large `ds4_metal_graph_*` family from
//! `ds4_metal.m` — the entry point upstream is the per-token forward pass
//! that opens a `MTLCommandBuffer`, walks every layer, encodes compute
//! dispatches against the cached pipelines, commits, and waits.
//!
//! This file currently lays down the *shape* of the executor (runtime +
//! pipeline cache + a forward entry point that opens / commits / waits a
//! command buffer for the supplied tokens). Per-kernel encoding (RMSNorm,
//! RoPE, MoE matmul, flash attention, …) is layered in by later commits as
//! the matching kernels are ported.

#![cfg(target_os = "macos")]

use anyhow::{anyhow, Result};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCommandEncoder,
};

use crate::kernels::Pipelines;
use crate::runtime::MetalRuntime;

/// Owns a `MetalRuntime` (device + queue + library) and a pipeline cache.
/// Re-used across forward passes so kernel compilation amortises.
pub struct Graph {
    pub runtime: MetalRuntime,
    pub pipelines: Pipelines,
}

impl Graph {
    pub fn new(runtime: MetalRuntime) -> Self {
        Self { runtime, pipelines: Pipelines::new() }
    }

    /// Borrow the underlying runtime.
    pub fn runtime(&self) -> &MetalRuntime { &self.runtime }

    /// Borrow the pipeline cache.
    pub fn pipelines(&self) -> &Pipelines { &self.pipelines }

    /// Encode and run one forward pass over `tokens`.
    ///
    /// This is the Rust analogue of the prefill/decode call site in
    /// `ds4_metal.m`: open a fresh command buffer, open a compute encoder,
    /// (eventually) dispatch every kernel, end the encoder, commit, and
    /// wait. With no kernels wired up yet this commits an empty command
    /// buffer, which is still a useful smoke test — it exercises queue
    /// creation, encoder lifecycle, and error reporting end-to-end.
    pub fn forward(&mut self, tokens: &[i32]) -> Result<()> {
        // A no-op forward over an empty prompt is a no-op.
        if tokens.is_empty() {
            return Ok(());
        }

        let cb = new_command_buffer(self.runtime.queue())
            .ok_or_else(|| anyhow!("ds4-metal: [queue commandBuffer] returned nil"))?;
        let enc = new_compute_encoder(&cb)
            .ok_or_else(|| anyhow!("ds4-metal: [cb computeCommandEncoder] returned nil"))?;

        // TODO: per-layer kernel encoding lands here as the kernels port.
        // We only borrow `self.pipelines` / `self.runtime` so future code
        // can fetch and dispatch without further plumbing changes.
        let _ = (&self.pipelines, &self.runtime, tokens);

        end_compute_encoder(&enc);
        commit_command_buffer(&cb);
        wait_command_buffer(&cb)?;

        Ok(())
    }
}

// ---------- safe wrappers over the unsafe Metal command-buffer dance ----------

/// `[queue commandBuffer]`.
fn new_command_buffer(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
    queue.commandBuffer()
}

/// `[cb computeCommandEncoder]`.
fn new_compute_encoder(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    cb.computeCommandEncoder()
}

/// `[encoder endEncoding]`.
fn end_compute_encoder(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
    enc.endEncoding();
}

/// `[cb commit]`.
fn commit_command_buffer(cb: &ProtocolObject<dyn MTLCommandBuffer>) {
    cb.commit();
}

/// `[cb waitUntilCompleted]` followed by an error-status check.
fn wait_command_buffer(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    // SAFETY: `waitUntilCompleted` is declared `unsafe` only because the
    // bindings cannot guarantee we are not blocking the main thread; here
    // we own the command buffer and Apple's API is itself thread-safe.
    unsafe { cb.waitUntilCompleted() };

    let status = cb.status();
    if status == MTLCommandBufferStatus::Error {
        // SAFETY: `error()` is unsafe only because it returns a retained
        // NSError without `nonnull`; we treat `None` as "no error attached"
        // and rely on `Retained`'s Drop to release any NSError we get.
        let err_msg = match unsafe { cb.error() } {
            Some(e) => format!("{}", e),
            None => "(no NSError attached)".to_owned(),
        };
        return Err(anyhow!("ds4-metal: command buffer failed: {}", err_msg));
    }
    Ok(())
}
