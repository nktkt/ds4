//! Metal graph executor. Ports the large `ds4_metal_graph_*` family from
//! `ds4_metal.m`.

use crate::kernels::Pipelines;
use crate::runtime::MetalRuntime;
use anyhow::Result;

pub struct Graph {
    pub runtime: MetalRuntime,
    pub pipelines: Pipelines,
}

impl Graph {
    pub fn new(runtime: MetalRuntime) -> Self {
        Self { runtime, pipelines: Pipelines::new() }
    }
}

impl Graph {
    pub fn forward(&mut self, _tokens: &[i32]) -> Result<()> { Ok(()) }
}
