//! CUDA forward-pass executor. Ports `ds4_cuda.cu::cuda_graph_*`.

use crate::runtime::CudaRuntime;
use anyhow::Result;

pub struct Graph {
    pub runtime: CudaRuntime,
}

impl Graph {
    pub fn new(runtime: CudaRuntime) -> Self { Self { runtime } }
    pub fn forward(&mut self, _tokens: &[i32]) -> Result<()> {
        Ok(())
    }
}
