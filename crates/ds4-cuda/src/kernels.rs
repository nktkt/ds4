//! Pre-compiled `.cu` kernel module cache.
//!
//! `build.rs` compiles `ds4_cuda.cu` into a PTX file (path exported via the
//! `DS4_CUDA_PTX` env var at compile time). At runtime, [`Kernels::load`]
//! reads that PTX and loads it into the active CUDA context as a named
//! module; individual `__global__` entry points are then resolved via
//! [`Kernels::func`].
//!
//! Only compiled on Linux.

#![cfg(target_os = "linux")]

use anyhow::{anyhow, Context, Result};
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};

/// Name of the cudarc module the PTX gets loaded under. Function lookup
/// always goes through this name, so keep it stable.
pub const MODULE_NAME: &str = "ds4_cuda";

/// Compile-time PTX path. `None` when the build skipped nvcc (no CUDA_HOME or
/// non-Linux host). At runtime [`Kernels::load`] will fail cleanly in that
/// case so a missing GPU build doesn't crash the rest of the crate.
pub const PTX_PATH: Option<&str> = option_env!("DS4_CUDA_PTX");

/// All `__global__` kernel names we want pre-registered with the module.
///
/// The list is conservative — adding a name here that isn't actually present
/// in the PTX would cause `load_ptx` to error out at startup. Start with the
/// hot-path forward kernels; expand as the matching host wrappers get ported
/// from `ds4_cuda.cu`. The full upstream surface is ~80 `__global__`
/// functions; this subset matches the top-of-stack ops referenced by
/// `graph::Graph::forward`.
pub const KERNEL_NAMES: &[&str] = &[
    // Embedding.
    "embed_token_hc_kernel",
    "embed_tokens_hc_kernel",
    // GEMM variants.
    "matmul_f16_kernel",
    "matmul_f32_kernel",
    "matmul_q8_0_preq_warp8_kernel",
    "matmul_q8_0_pair_preq_warp8_kernel",
    "matmul_q8_0_preq_batch_warp8_kernel",
    // Quantization helpers.
    "quantize_q8_0_f32_kernel",
    "dequant_q8_0_to_f16_kernel",
    "dequant_q8_0_to_f32_kernel",
    "f32_to_f16_kernel",
    // Normalization / RoPE.
    "rms_norm_plain_kernel",
    "rms_norm_weight_kernel",
    "head_rms_norm_kernel",
    "rope_tail_kernel",
    // Attention (decode / prefill core).
    "attention_decode_mixed_kernel",
    "attention_prefill_raw_kernel",
    "attention_prefill_mixed_kernel",
    // MoE / SwiGLU / utility.
    "swiglu_kernel",
    "add_kernel",
    "fill_f32_kernel",
    "zero_kernel",
];

/// Loaded CUDA kernel module. Lookup is fallible because kernel names that
/// were not declared in [`KERNEL_NAMES`] are simply absent.
pub struct Kernels {
    device: Arc<CudaDevice>,
}

impl Kernels {
    /// Load the PTX module compiled by `build.rs` and register all
    /// [`KERNEL_NAMES`] entry points with cudarc's module cache.
    pub fn load(device: Arc<CudaDevice>) -> Result<Self> {
        let ptx_path = PTX_PATH.ok_or_else(|| {
            anyhow!(
                "ds4-cuda: DS4_CUDA_PTX is unset — build.rs did not compile the .cu source. \
                 Ensure CUDA_HOME points at a CUDA toolkit on a Linux host and rebuild."
            )
        })?;

        let ptx = cudarc::nvrtc::Ptx::from_file(ptx_path);

        device
            .load_ptx(ptx, MODULE_NAME, KERNEL_NAMES)
            .with_context(|| {
                format!("ds4-cuda: load_ptx failed for {ptx_path} (module {MODULE_NAME})")
            })?;

        Ok(Self { device })
    }

    /// Resolve a kernel function by name. The name must appear in
    /// [`KERNEL_NAMES`] (or have been loaded by other means via the same
    /// module name).
    pub fn func(&self, name: &str) -> Result<CudaFunction> {
        self.device
            .get_func(MODULE_NAME, name)
            .ok_or_else(|| anyhow!("ds4-cuda: kernel `{name}` not found in module `{MODULE_NAME}`"))
    }

    /// Borrow the device this module is bound to.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}
