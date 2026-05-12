//! Metal runtime: device, command queue, library loading. Ports
//! `ds4_metal.m::ds4_metal_init` / `ds4_metal_queue_*`.
//!
//! Concrete objc2-metal bindings live behind a follow-up commit; this file
//! exposes the public type surface so dependents can compile and import the
//! correct names.

use anyhow::{bail, Result};
use std::path::Path;

pub struct MetalRuntime {
    // TODO: hold Retained<ProtocolObject<dyn MTLDevice>> / MTLCommandQueue /
    // MTLLibrary once the objc2-metal API is wired up.
    pub(crate) metallib_path: std::path::PathBuf,
}

impl MetalRuntime {
    pub fn open(metallib_path: &Path) -> Result<Self> {
        if !metallib_path.exists() {
            bail!("metal: metallib not found at {}", metallib_path.display());
        }
        Ok(Self { metallib_path: metallib_path.to_owned() })
    }
}
