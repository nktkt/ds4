//! CUDA runtime via `cudarc`. Ports the host-side device setup from
//! `ds4_cuda.cu::ds4_gpu_init` (and surrounding globals such as `g_cublas` /
//! `g_cublas_ready`) onto a per-instance Rust handle.
//!
//! Only compiled on Linux — the macOS / non-Linux build of the workspace
//! relies on `ds4-metal` and never references this module (see
//! `lib.rs`'s `cfg`-gated re-exports).

#![cfg(target_os = "linux")]

use anyhow::{Context, Result};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

/// Owns the CUDA device handle plus an optional cuBLAS handle.
///
/// The upstream C++ keeps a global `cublasHandle_t g_cublas` plus an
/// initialization flag.  We hold the equivalent state inline; callers create
/// one `CudaRuntime` per inference engine instance.
pub struct CudaRuntime {
    /// `cudarc` device handle — `Arc<CudaDevice>` is the cheap-cloneable token
    /// returned by `CudaDevice::new`. All `cudarc` APIs accept it by clone.
    pub device: Arc<CudaDevice>,

    /// Lazily-initialized cuBLAS handle.  cuBLAS is only required for the
    /// large F16 / F32 GEMMs reachable via `ds4_gpu_matmul_f16_tensor` and
    /// `ds4_gpu_matmul_f32_tensor`; smaller kernels run as plain CUDA launches
    /// and don't need it.
    pub cublas: Option<Arc<cudarc::cublas::CudaBlas>>,
}

impl CudaRuntime {
    /// Open the CUDA device with the given ordinal (0 == first GPU). Does not
    /// initialize cuBLAS — call [`CudaRuntime::with_cublas`] for that.
    pub fn open(ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(ordinal)
            .with_context(|| format!("ds4-cuda: CudaDevice::new({ordinal}) failed"))?;
        Ok(Self {
            device,
            cublas: None,
        })
    }

    /// Open the CUDA device and eagerly create a cuBLAS handle bound to it.
    pub fn with_cublas(ordinal: usize) -> Result<Self> {
        let mut rt = Self::open(ordinal)?;
        rt.ensure_cublas()?;
        Ok(rt)
    }

    /// Lazily allocate the cuBLAS handle if not already present, returning it.
    pub fn ensure_cublas(&mut self) -> Result<Arc<cudarc::cublas::CudaBlas>> {
        if let Some(ref h) = self.cublas {
            return Ok(h.clone());
        }
        let blas = cudarc::cublas::CudaBlas::new(self.device.clone())
            .context("ds4-cuda: cublasCreate failed")?;
        let arc = Arc::new(blas);
        self.cublas = Some(arc.clone());
        Ok(arc)
    }

    /// Block until all previously-enqueued work on the default stream finishes.
    /// Mirrors `ds4_gpu_synchronize` in the upstream C++.
    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .context("ds4-cuda: cudaDeviceSynchronize failed")?;
        Ok(())
    }

    /// Device ordinal — convenience for logging / error messages.
    pub fn ordinal(&self) -> usize {
        self.device.ordinal()
    }
}
