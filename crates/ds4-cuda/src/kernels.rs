//! Pre-compiled `.cu` kernel handle cache. The actual `.cu` source is compiled
//! by `build.rs` (invoking `nvcc`) into a `.ptx`/`.cubin`; this module loads
//! that artifact and resolves function symbols on demand.

use anyhow::{anyhow, Result};
use std::sync::Arc;

pub struct Kernels {
    pub module: Arc<cudarc::driver::CudaDevice>,
}

impl Kernels {
    pub fn open(device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let _ = device.clone();
        // TODO: load PTX module via cudarc::driver::CudaDevice::load_ptx.
        Err(anyhow!("ds4-cuda: kernels not built yet"))
    }
}
