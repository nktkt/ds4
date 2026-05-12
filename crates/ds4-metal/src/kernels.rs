//! Compiled compute-pipeline cache. Mirrors `ds4_metal.m::pipeline_*` — looks
//! up Metal kernel functions by name and memoizes the resulting
//! `MTLComputePipelineState`.

use anyhow::{bail, Result};

pub struct Pipelines {
    // TODO: device + library + AHashMap<String, MTLComputePipelineState>
}

impl Pipelines {
    pub fn new() -> Self { Self {} }
    pub fn get(&self, _name: &str) -> Result<()> {
        bail!("ds4-metal: pipeline cache not yet ported")
    }
}

impl Default for Pipelines {
    fn default() -> Self { Self::new() }
}
