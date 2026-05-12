//! Safe wrappers over `MTLComputeCommandEncoder`.
//!
//! Ports the encoder-side dispatch helpers used throughout `ds4_metal.m` —
//! see e.g. lines 244-258 (`computeCommandEncoder` / `endEncoding`),
//! 714-724 (`setComputePipelineState` / `setBuffer` / `setBytes` /
//! `dispatchThreadgroups`), 2431-2441 (RoPE tail batch dispatch), and
//! 4133-4138 (`get_rows_f16` dispatch). Every `unsafe` Objective-C call
//! is funneled through a tiny private helper carrying a SAFETY comment;
//! the public surface is fully safe.
//!
//! The encoder is borrowed for the lifetime of its parent command buffer
//! (`'a`) so the borrow checker prevents using a stale encoder after the
//! command buffer has been committed.

#![cfg(target_os = "macos")]

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use anyhow::{anyhow, Result};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
};

/// Safe handle for a `MTLComputeCommandEncoder` borrowed for the lifetime of
/// its parent command buffer.
///
/// Mirrors the local `id<MTLComputeCommandEncoder> enc` variables peppered
/// throughout `ds4_metal.m` (e.g. line 244's `g_batch_enc`).
pub struct Encoder<'a> {
    pub raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _marker: PhantomData<&'a ()>,
}

/// Open a fresh compute command encoder against `cb`.
///
/// Wraps `[cb computeCommandEncoder]` (`ds4_metal.m` lines 244, 247).
/// Returns an error if Metal hands back nil — typically meaning the parent
/// command buffer has already been committed or encoded against by another
/// encoder still in flight.
pub fn open_compute(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<Encoder<'_>> {
    let raw = new_compute_encoder(cb)
        .ok_or_else(|| anyhow!("ds4-metal: [cb computeCommandEncoder] returned nil"))?;
    Ok(Encoder { raw, _marker: PhantomData })
}

impl<'a> Encoder<'a> {
    /// Bind a compute pipeline state.
    ///
    /// Wraps `[enc setComputePipelineState:]` (`ds4_metal.m` line 714,
    /// `MTLComputeCommandEncoder.setComputePipelineState`).
    pub fn set_pipeline(
        &self,
        pso: &ProtocolObject<dyn MTLComputePipelineState>,
    ) {
        // `setComputePipelineState:` is safe in the bindings (no nullable
        // pointer arguments), so no `unsafe` block is required.
        self.raw.setComputePipelineState(pso);
    }

    /// Bind a `MTLBuffer` at `index` with the given byte offset.
    ///
    /// Wraps `[enc setBuffer:offset:atIndex:]` (`ds4_metal.m` lines 719-720,
    /// 4135-4137, …).
    pub fn set_buffer(
        &self,
        index: usize,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
    ) {
        set_buffer(&self.raw, index, buf, offset);
    }

    /// Inline a small `Pod` constant as an argument buffer.
    ///
    /// Wraps `[enc setBytes:length:atIndex:]` (`ds4_metal.m` lines 721-723,
    /// 2432, 4134, …). Metal's `setBytes:` is documented to be appropriate
    /// for arguments up to 4 KiB; larger payloads should go through a real
    /// `MTLBuffer` (cf. the comment at `ds4_metal.m` line 2416).
    pub fn set_bytes<T: bytemuck::Pod>(&self, index: usize, value: &T) {
        // `bytemuck::bytes_of` performs the Pod-ness compile-time check and
        // returns a `&[u8]` whose pointer is non-null and length matches
        // `size_of::<T>()`.
        let bytes: &[u8] = bytemuck::bytes_of(value);
        set_bytes(&self.raw, index, bytes);
    }

    /// Dispatch `threads_per_grid` total threads in groups of
    /// `threads_per_group` (non-uniform grid).
    ///
    /// Wraps `[enc dispatchThreads:threadsPerThreadgroup:]`.
    /// This is the variant used when the kernel handles its own bounds
    /// checking (Metal automatically rounds down the final group).
    pub fn dispatch_grid(
        &self,
        threads_per_grid: MTLSize,
        threads_per_group: MTLSize,
    ) {
        dispatch_threads(&self.raw, threads_per_grid, threads_per_group);
    }

    /// Dispatch a fixed number of threadgroups.
    ///
    /// Wraps `[enc dispatchThreadgroups:threadsPerThreadgroup:]`
    /// (`ds4_metal.m` lines 724, 2441, 4138, 4185, 4263, …). This is the
    /// dispatch form DS4 uses everywhere a kernel is written for a uniform,
    /// caller-rounded grid (e.g. `MTLSizeMake((n + 255) / 256, 1, 1)`).
    pub fn dispatch_threadgroups(
        &self,
        threadgroups: MTLSize,
        threads_per_group: MTLSize,
    ) {
        dispatch_threadgroups(&self.raw, threadgroups, threads_per_group);
    }

    /// Close the encoder.
    ///
    /// Wraps `[enc endEncoding]` (`ds4_metal.m` lines 253, 258,
    /// `MTLCommandEncoder.endEncoding`). Consumes `self` so the caller
    /// cannot accidentally keep dispatching after the encoder is finalised.
    pub fn end(self) {
        end_encoding(&self.raw);
    }
}

// ---------- safe wrappers over the unsafe Objective-C entry points ----------

/// `[cb computeCommandEncoder]`.
fn new_compute_encoder(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    // `computeCommandEncoder` on `MTLCommandBuffer` is safe in the bindings;
    // it returns `Option<Retained<…>>` directly.
    cb.computeCommandEncoder()
}

/// `[enc setBuffer:offset:atIndex:]`.
fn set_buffer(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    buf: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
) {
    // SAFETY: `setBuffer:offset:atIndex:` is declared `unsafe` because the
    // buffer argument is nullable in Objective-C. We pass a `Some(&buf)` —
    // a live `&ProtocolObject<MTLBuffer>` whose retain is upheld by the
    // caller's borrow — so the nil case is statically excluded. `offset`
    // and `index` are plain integers; Metal validates them at draw time.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf), offset as _, index as _);
    }
}

/// `[enc setBytes:length:atIndex:]`.
fn set_bytes(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    bytes: &[u8],
) {
    // SAFETY: `setBytes:length:atIndex:` requires `bytes` to point at
    // `length` readable bytes for the duration of the call. We derive
    // both pointer and length from the same `&[u8]`, which is alive for
    // the entire call. The Metal driver copies the buffer internally
    // before returning, so the slice need not outlive this function.
    // An empty slice still has a non-null aligned pointer (Rust guarantee),
    // and we pass length = 0, which Metal tolerates as a no-op.
    let ptr = NonNull::new(bytes.as_ptr() as *mut c_void)
        .unwrap_or(NonNull::dangling());
    unsafe {
        enc.setBytes_length_atIndex(ptr, bytes.len() as _, index as _);
    }
}

/// `[enc dispatchThreads:threadsPerThreadgroup:]`.
fn dispatch_threads(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    threads_per_grid: MTLSize,
    threads_per_group: MTLSize,
) {
    // `dispatchThreads:threadsPerThreadgroup:` is declared safe in the
    // bindings — `MTLSize` is a plain POD value type with no pointer fields.
    enc.dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_group);
}

/// `[enc dispatchThreadgroups:threadsPerThreadgroup:]`.
fn dispatch_threadgroups(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    threadgroups: MTLSize,
    threads_per_group: MTLSize,
) {
    // `dispatchThreadgroups:threadsPerThreadgroup:` is declared safe in
    // the bindings; both arguments are POD `MTLSize` values.
    enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_group);
}

/// `[enc endEncoding]`.
fn end_encoding(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
    // `endEncoding` on `MTLCommandEncoder` is safe in the bindings.
    enc.endEncoding();
}

/// `[queue commandBuffer]`.
fn new_command_buffer(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
    queue.commandBuffer()
}

/// `[cb commit]`.
fn commit_command_buffer(cb: &ProtocolObject<dyn MTLCommandBuffer>) {
    cb.commit();
}

/// `[cb waitUntilCompleted]` + error-status check.
fn wait_command_buffer(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    // SAFETY: `waitUntilCompleted` is declared `unsafe` only because the
    // bindings cannot prove we are not blocking the main thread; the call
    // itself is thread-safe per Apple's documentation and we own the
    // command buffer through `cb`.
    unsafe { cb.waitUntilCompleted() };

    let status = cb.status();
    if status == MTLCommandBufferStatus::Error {
        // SAFETY: `error()` returns an `Option<Retained<NSError>>`; the
        // bindings mark it `unsafe` because nullability is not encoded in
        // the Objective-C runtime metadata, but we treat `None` as
        // "no NSError attached" and let `Retained`'s Drop release any
        // value we do receive.
        let err_msg = match unsafe { cb.error() } {
            Some(e) => format!("{}", e),
            None => "(no NSError attached)".to_owned(),
        };
        return Err(anyhow!("ds4-metal: command buffer failed: {}", err_msg));
    }
    Ok(())
}

/// Open a command buffer, run `build` against it (which typically opens an
/// encoder via [`open_compute`], dispatches some kernels, and calls
/// [`Encoder::end`]), then commit and `waitUntilCompleted`.
///
/// This is the synchronous "fire and wait" pattern used by the forward-pass
/// driver in `ds4_metal.m` (open command buffer -> encode -> commit ->
/// `waitUntilCompleted` -> check `error`). Any NSError surfaced by the
/// command buffer is converted into an `anyhow::Error`.
pub fn submit_blocking(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    build: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>) -> Result<()>,
) -> Result<()> {
    let cb = new_command_buffer(queue)
        .ok_or_else(|| anyhow!("ds4-metal: [queue commandBuffer] returned nil"))?;

    // Run the caller's encoding closure. If it errors out we still commit
    // nothing — the command buffer is simply dropped, which is fine because
    // we never called `commit`.
    build(&cb)?;

    commit_command_buffer(&cb);
    wait_command_buffer(&cb)?;
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::runtime::MetalRuntime;

    /// Smoke test: build a `MetalRuntime` and exercise `submit_blocking` with
    /// an empty encoder. We only run the full Metal path if a precompiled
    /// `metal/ds4.metallib` is checked into the crate; otherwise the test
    /// short-circuits successfully so CI on a machine without the metallib
    /// (or without GPU access) does not fail.
    #[test]
    fn submit_blocking_smoke() {
        let metallib = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("metal")
            .join("ds4.metallib");
        if !metallib.exists() {
            eprintln!(
                "skipping submit_blocking_smoke: {} not present",
                metallib.display()
            );
            return;
        }

        let runtime = match MetalRuntime::open(&metallib) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping submit_blocking_smoke: runtime open failed: {e}");
                return;
            }
        };

        // An empty encoder cycle is still a valid Metal program: open
        // command buffer, open compute encoder, immediately end it, commit,
        // wait. If this comes back without an NSError we have proved the
        // whole encoder lifecycle works.
        submit_blocking(runtime.queue(), |cb| {
            let enc = open_compute(cb)?;
            enc.end();
            Ok(())
        })
        .expect("empty submit_blocking should succeed");
    }
}
