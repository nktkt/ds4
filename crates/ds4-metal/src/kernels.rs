//! Compiled compute-pipeline cache. Mirrors `ds4_metal.m::pipeline_*` and
//! `g_pipeline_cache` (line 104 of `ds4_metal.m`): we look up Metal kernel
//! functions by name and memoize the resulting `MTLComputePipelineState` so
//! that repeated graph executions don't re-pay the JIT cost.
//!
//! The cache keyed by function name only; specialised variants that use
//! `MTLFunctionConstantValues` (e.g. `mul_mm_id_*` with `bc_inp`/`bc_out`)
//! will go through a separate API once those kernels are ported.

#![cfg(target_os = "macos")]

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary,
};

use crate::runtime::MetalRuntime;

/// Function-name -> compiled pipeline cache.
///
/// `MTLComputePipelineState` objects are device-owned and reference-counted,
/// so this struct can be cloned through a `Retained` if ever needed; for now
/// we hand out borrows via `get`.
pub struct Pipelines {
    // We use a `Mutex<HashMap<...>>` rather than e.g. `DashMap` because the
    // critical section is tiny (one Objective-C call on miss) and pipelines
    // are realistically created once at graph build time.
    cache: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

impl Pipelines {
    /// Construct an empty cache. The runtime is passed through `get` so we
    /// don't have to hold a reference to it.
    pub fn new() -> Self {
        Self { cache: Mutex::new(HashMap::new()) }
    }

    /// Look up the pipeline for `function_name`, compiling and caching it
    /// the first time. Ports `ds4_gpu_get_pipeline` from `ds4_metal.m`.
    pub fn get(
        &self,
        runtime: &MetalRuntime,
        function_name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        {
            let cache = self.cache.lock().expect("pipeline cache poisoned");
            if let Some(p) = cache.get(function_name) {
                return Ok(p.clone());
            }
        }

        let pipeline = compile_pipeline(runtime, function_name)?;

        let mut cache = self.cache.lock().expect("pipeline cache poisoned");
        // Another thread may have raced us to insert; prefer the existing.
        let entry = cache.entry(function_name.to_owned()).or_insert(pipeline);
        Ok(entry.clone())
    }

    /// Number of pipelines currently memoized (diagnostics).
    pub fn len(&self) -> usize {
        self.cache.lock().expect("pipeline cache poisoned").len()
    }

    /// Whether the cache holds zero pipelines.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for Pipelines {
    fn default() -> Self { Self::new() }
}

// ---------- safe wrapper over the unsafe Metal compile path ----------

/// `[library newFunctionWithName:]` + `[device newComputePipelineStateWithFunction:error:]`.
fn compile_pipeline(
    runtime: &MetalRuntime,
    function_name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let ns_name = NSString::from_str(function_name);

    // `newFunctionWithName:` returns an Option<id<MTLFunction>> (nil if the
    // name is missing).  It is declared safe in the bindings.
    let function: Retained<ProtocolObject<dyn MTLFunction>> = runtime
        .library()
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| {
            anyhow!(
                "ds4-metal: Metal function '{}' not found in library",
                function_name
            )
        })?;

    // `newComputePipelineStateWithFunction:error:` is safe in the bindings.
    runtime
        .device()
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| {
            anyhow!(
                "ds4-metal: pipeline compile failed for '{}': {}",
                function_name,
                e
            )
        })
        .with_context(|| format!("compiling pipeline for {}", function_name))
}
