//! CUDA forward-pass executor.
//!
//! Ports the host-side orchestration that lives around
//! `ds4_gpu_*_tensor` calls in `ds4_cuda.cu`. The actual `__global__` kernels
//! stay in the `.cu` source compiled by `build.rs`; this module wires
//! [`CudaRuntime`] + [`Kernels`] together and exposes a single
//! [`Graph::forward`] entry point that mirrors the C++ control flow.
//!
//! Only compiled on Linux.

#![cfg(target_os = "linux")]

use anyhow::Result;

use crate::kernels::Kernels;
use crate::runtime::CudaRuntime;

/// Forward-pass executor: owns the CUDA runtime + the loaded kernel module,
/// and (eventually) all of the per-layer GPU scratch buffers that the
/// upstream C++ keeps in globals (`g_cuda_tmp`, `g_model_stage[]`, etc.).
pub struct Graph {
    pub runtime: CudaRuntime,
    pub kernels: Kernels,
}

impl Graph {
    /// Build a [`Graph`] from a runtime by loading the compiled PTX module.
    pub fn new(runtime: CudaRuntime) -> Result<Self> {
        let kernels = Kernels::load(runtime.device.clone())?;
        Ok(Self { runtime, kernels })
    }

    /// Run a forward pass for the provided input tokens. Mirrors the
    /// `cuda_graph_*` / per-layer launches in `ds4_cuda.cu`.
    ///
    /// Currently a synchronization-only stub: per-op kernel launches will
    /// land alongside the matching host wrappers in subsequent ports.
    pub fn forward(&mut self, _tokens: &[i32]) -> Result<()> {
        // TODO: port `ds4_gpu_embed_tokens_hc_tensor` → attention → MoE →
        // output head sequence here, using `self.kernels.func(...)` to get
        // each `CudaFunction` and launching with `.launch(cfg, params)`.
        self.runtime.synchronize()?;
        Ok(())
    }
}
