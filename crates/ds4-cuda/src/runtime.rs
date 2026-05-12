//! CUDA runtime via `cudarc`. Ports `ds4_cuda.cu::ds4_cuda_init` /
//! `ds4_cuda_stream_*`.

use anyhow::Result;
use std::sync::Arc;

pub struct CudaRuntime {
    pub device: Arc<cudarc::driver::CudaDevice>,
}

impl CudaRuntime {
    pub fn open(ordinal: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(ordinal)?;
        Ok(Self { device })
    }
}
