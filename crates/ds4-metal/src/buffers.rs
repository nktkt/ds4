//! Metal buffer allocation helpers. Ports the wrapper macros around
//! `[device newBufferWithLength:...]` and the per-graph slab arena from
//! `ds4_metal.m`.

use anyhow::{bail, Result};

pub fn new_buffer(_bytes: usize) -> Result<()> {
    bail!("ds4-metal: buffer allocator not yet ported")
}
