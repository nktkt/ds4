//! Typed launch helpers for the hottest forward-pass kernels.
//!
//! Each method here corresponds to one `__global__` kernel in the upstream
//! `ds4_cuda.cu` and to one of the names registered in
//! [`crate::kernels::KERNEL_NAMES`]. They centralise the grid/block selection
//! and argument-tuple shape so the higher-level [`crate::graph::Graph`]
//! executor can drive the GPU without re-deriving launch math at every call
//! site.
//!
//! Linux-only — these helpers are gated on `target_os = "linux"` because
//! `cudarc` (and hence the entire CUDA backend) is only built there.

#![cfg(target_os = "linux")]

use anyhow::Result;

use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};

use crate::kernels::Kernels;
use crate::runtime::CudaRuntime;

/// Bundles a [`CudaRuntime`] handle with a loaded [`Kernels`] module and
/// exposes one method per hot-path kernel. Cheap to construct — the
/// expensive state (device handle, PTX module) lives in the borrows.
pub struct Launcher<'a> {
    /// CUDA device + optional cuBLAS handle. Currently only used for the
    /// device reference; held so future helpers can reach cuBLAS without
    /// changing the public surface.
    pub runtime: &'a CudaRuntime,
    /// Loaded PTX module exposing every name in
    /// [`crate::kernels::KERNEL_NAMES`].
    pub kernels: &'a Kernels,
}

impl<'a> Launcher<'a> {
    /// Construct a launcher view over an existing runtime + kernel module.
    pub fn new(runtime: &'a CudaRuntime, kernels: &'a Kernels) -> Self {
        Self { runtime, kernels }
    }

    /// Launch `quantize_q8_0_f32_kernel`.
    ///
    /// Upstream: `ds4_cuda.cu::quantize_q8_0_f32_kernel`. Quantises a single
    /// row (`n_tok == 1`) of `n` `f32` values into Q8_0 blocks of 32, writing
    /// `(n + 31) / 32` `int8` blocks into `xq` plus the matching per-block
    /// `f32` scale into `scale`.
    ///
    /// Grid is `(blocks, n_tok=1)` with a 32-thread warp per block, matching
    /// the upstream launch at `ds4_gpu_matmul_q8_0_tensor`.
    pub fn quantize_q8_0_f32(
        &self,
        x: &CudaSlice<f32>,
        xq: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        let func = self.kernels.func("quantize_q8_0_f32_kernel")?;
        let blocks: u32 = (n + 31) / 32;
        // Single token: grid.x = blocks, grid.y = 1, block = 32 threads.
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        // Upstream signature:
        //   (int8_t *xq, float *xscale, const float *x,
        //    uint64_t in_dim, uint64_t blocks)
        let in_dim: u64 = n as u64;
        let blocks_u64: u64 = blocks as u64;
        unsafe { func.launch(cfg, (xq, scale, x, in_dim, blocks_u64)) }
            .map_err(anyhow::Error::from)?;
        Ok(())
    }

    /// Launch `matmul_q8_0_preq_warp8_kernel`.
    ///
    /// Upstream: `ds4_cuda.cu::matmul_q8_0_preq_warp8_kernel`. Single-token
    /// Q8_0 GEMM where the weights `w` are already pre-quantised on device
    /// (the `_preq_` infix in the name) and `xq` / `scale` are the per-token
    /// quantised activations produced by [`Launcher::quantize_q8_0_f32`].
    ///
    /// Grid is `((n_rows + 7) / 8)` blocks of 256 threads (8 warps), each
    /// warp computing one output row, matching the upstream launch in
    /// `ds4_gpu_matmul_q8_0_tensor` when `n_tok == 1`.
    pub fn matmul_q8_0_preq_warp8(
        &self,
        w: &CudaSlice<u8>,
        xq: &CudaSlice<i8>,
        scale: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_rows: u32,
        in_dim: u32,
    ) -> Result<()> {
        let func = self.kernels.func("matmul_q8_0_preq_warp8_kernel")?;
        let blocks: u32 = (in_dim + 31) / 32;
        let grid_x: u32 = (n_rows + 7) / 8;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Upstream signature:
        //   (float *out, const unsigned char *w, const int8_t *xq,
        //    const float *xscale, uint64_t in_dim, uint64_t out_dim,
        //    uint64_t blocks, int use_dp4a)
        //
        // We default `use_dp4a` to 1 — sm_61+ supports it and the upstream
        // host wrapper only disables it on older arches via env var.
        let in_dim_u64: u64 = in_dim as u64;
        let out_dim_u64: u64 = n_rows as u64;
        let blocks_u64: u64 = blocks as u64;
        let use_dp4a: i32 = 1;
        unsafe {
            func.launch(
                cfg,
                (out, w, xq, scale, in_dim_u64, out_dim_u64, blocks_u64, use_dp4a),
            )
        }
        .map_err(anyhow::Error::from)?;
        Ok(())
    }

    /// Launch `rms_norm_weight_kernel`.
    ///
    /// Upstream: `ds4_cuda.cu::rms_norm_weight_kernel`. Single-row RMSNorm
    /// (`rows = 1`) with a learned weight vector `w`, matching the host
    /// wrapper `ds4_gpu_rms_norm_weight_tensor` which launches `<<<1, 256>>>`.
    pub fn rms_norm_weight(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        n: u32,
        eps: f32,
    ) -> Result<()> {
        let func = self.kernels.func("rms_norm_weight_kernel")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Upstream signature:
        //   (float *out, const float *x, const float *w,
        //    uint32_t n, uint32_t rows, float eps)
        let rows: u32 = 1;
        unsafe { func.launch(cfg, (out, x, w, n, rows, eps)) }.map_err(anyhow::Error::from)?;
        Ok(())
    }

    /// Launch `rope_tail_kernel`.
    ///
    /// Upstream: `ds4_cuda.cu::rope_tail_kernel`. Rotary embedding over the
    /// trailing `n_rot` lanes of each head, used during decode where
    /// `n_tok == 1`. Grid sized to one thread per pair, in 256-thread blocks
    /// — same as the upstream launch in `ds4_gpu_rope_tail_tensor`.
    ///
    /// `pos` is widened to `u64` on the Rust side for ergonomics; the kernel
    /// itself takes a `uint32_t pos0`, so this is truncated at launch.
    pub fn rope_tail(
        &self,
        x: &mut CudaSlice<f32>,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos: u64,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        let func = self.kernels.func("rope_tail_kernel")?;
        let n_tok: u32 = 1;
        let pairs: u32 = n_tok
            .saturating_mul(n_head)
            .saturating_mul(n_rot / 2);
        let cfg = LaunchConfig::for_num_elems(pairs);
        // Upstream signature:
        //   (float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        //    uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig,
        //    int inverse, float freq_base, float freq_scale,
        //    float ext_factor, float attn_factor,
        //    float beta_fast, float beta_slow)
        //
        // We feed default RoPE-scaling parameters (no YaRN extension) so the
        // helper matches the common decode path.
        let pos0: u32 = pos as u32;
        let n_ctx_orig: u32 = 0;
        let inverse: i32 = 0;
        let ext_factor: f32 = 0.0;
        let attn_factor: f32 = 1.0;
        let beta_fast: f32 = 32.0;
        let beta_slow: f32 = 1.0;
        unsafe {
            func.launch(
                cfg,
                (
                    x,
                    n_tok,
                    n_head,
                    head_dim,
                    n_rot,
                    pos0,
                    n_ctx_orig,
                    inverse,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    beta_fast,
                    beta_slow,
                ),
            )
        }
        .map_err(anyhow::Error::from)?;
        Ok(())
    }

    /// Launch `swiglu_kernel`.
    ///
    /// Upstream: `ds4_cuda.cu::swiglu_kernel`. Elementwise `swish(gate) * up`
    /// (optionally clamped and weighted). Grid sized via
    /// `LaunchConfig::for_num_elems(n)`, matching the upstream launch in
    /// `ds4_gpu_swiglu_tensor` which uses `((n + 255) / 256, 256)`.
    pub fn swiglu(
        &self,
        out: &mut CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        let func = self.kernels.func("swiglu_kernel")?;
        let cfg = LaunchConfig::for_num_elems(n);
        // Upstream signature:
        //   (float *out, const float *gate, const float *up,
        //    uint32_t n, float clamp, float weight)
        //
        // Default to "no clamp, unit weight" — the upstream host wrapper
        // exposes both, but the forward pass we target here always passes
        // (0.0, 1.0).
        let clamp: f32 = 0.0;
        let weight: f32 = 1.0;
        unsafe { func.launch(cfg, (out, gate, up, n, clamp, weight)) }
            .map_err(anyhow::Error::from)?;
        Ok(())
    }

    /// Launch `attention_decode_mixed_kernel`.
    ///
    /// Upstream: `ds4_cuda.cu::attention_decode_mixed_kernel`. The signature
    /// is large (17 parameters spanning raw KV cache, compressed KV cache,
    /// optional mask, ratios, windows) and depends on KV cache layout that
    /// hasn't been ported yet. Leaving as a stub so the rest of the
    /// launcher can land; revisit when the KV cache types exist on the
    /// Rust side.
    pub fn attention_decode_mixed(&self) -> Result<()> {
        anyhow::bail!("TODO: port attention_decode_mixed signature")
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    /// Smoke test: just verifies the `Launcher` type composes against the
    /// real `CudaRuntime` + `Kernels` API. Constructing a `CudaRuntime`
    /// requires an NVIDIA driver / device, so this is `#[ignore]` by default
    /// and only runs under `cargo test -- --ignored` on a GPU host. The
    /// purpose is to keep the borrow shape and signatures honest at compile
    /// time without forcing CI to provision a GPU.
    #[test]
    #[ignore]
    fn launcher_constructs() {
        let runtime = CudaRuntime::open(0).expect("CudaRuntime::open(0)");
        let kernels = Kernels::load(runtime.device.clone()).expect("Kernels::load");
        let launcher = Launcher::new(&runtime, &kernels);
        // Touch the borrows so the compiler can't optimise the construction away.
        let _ = launcher.runtime.ordinal();
    }
}
