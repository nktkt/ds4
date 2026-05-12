//! Metal buffer allocation helpers. Ports the thin wrappers around
//! `[device newBufferWithLength:options:]` (e.g. `ds4_gpu_new_transient_buffer`,
//! `ds4_gpu_ensure_scratch_buffer`) plus a minimal bump-arena used by the
//! per-graph scratch carved out of `g_transient_buffers` in `ds4_metal.m`.
//!
//! The full per-graph slab allocator (model-view splitting,
//! `DS4_METAL_MAX_MODEL_VIEWS`, residency sets) lives upstream of this slice
//! and will be ported in follow-up commits — what's here is the safe
//! foundation those higher layers will build on.

#![cfg(target_os = "macos")]

use anyhow::{anyhow, Result};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::runtime::MetalRuntime;

/// Allocate a fresh shared-storage `MTLBuffer` of `bytes` length. Zero-sized
/// requests are bumped to 1 byte (matches `ds4_gpu_new_transient_buffer`).
///
/// Returns a retained handle; drop it to release the GPU allocation.
pub fn new_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    label: Option<&str>,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let length = if bytes == 0 { 1 } else { bytes };

    let buffer = runtime
        .device()
        .newBufferWithLength_options(length, MTLResourceOptions::MTLResourceStorageModeShared)
        .ok_or_else(|| {
            anyhow!(
                "ds4-metal: newBufferWithLength:{} returned nil (label={:?})",
                length,
                label
            )
        })?;

    if let Some(name) = label {
        let ns_name = NSString::from_str(name);
        set_buffer_label(&buffer, &ns_name);
    }

    Ok(buffer)
}

/// Set the `label` debug name on `buffer`. Safe because `setLabel:` on
/// `MTLResource` is itself safe in the bindings.
fn set_buffer_label(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    label: &NSString,
) {
    use objc2_metal::MTLResource;
    buffer.setLabel(Some(label));
}

/// Simple bump arena over one shared-storage Metal buffer.
///
/// Carved straight out of one `MTLBuffer`: every `alloc` call hands back a
/// byte offset into the backing storage and advances the cursor. There is no
/// per-allocation free — call `reset` between iterations to recycle the whole
/// arena, the same way `ds4_metal.m` clears `g_transient_buffers` after each
/// `commit`+`waitUntilCompleted` pair.
pub struct BumpArena {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    capacity: usize,
    cursor: usize,
}

impl BumpArena {
    /// Allocate the underlying buffer up front. Use `with_capacity` to right-
    /// size for a given prefill / decode round.
    pub fn with_capacity(
        runtime: &MetalRuntime,
        capacity: usize,
        label: Option<&str>,
    ) -> Result<Self> {
        let buffer = new_buffer(runtime, capacity, label)?;
        Ok(Self { buffer, capacity, cursor: 0 })
    }

    /// Reserve `bytes` bytes at the given alignment and return the byte
    /// offset into `self.buffer()`. Returns an error if the arena is full.
    pub fn alloc(&mut self, bytes: usize, align: usize) -> Result<usize> {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        let aligned = (self.cursor + align - 1) & !(align - 1);
        let end = aligned
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("ds4-metal: BumpArena overflow"))?;
        if end > self.capacity {
            return Err(anyhow!(
                "ds4-metal: BumpArena out of space ({} + {} > {})",
                aligned,
                bytes,
                self.capacity
            ));
        }
        self.cursor = end;
        Ok(aligned)
    }

    /// Reset the bump pointer to zero. The buffer itself is kept alive.
    pub fn reset(&mut self) { self.cursor = 0; }

    /// Capacity in bytes.
    pub fn capacity(&self) -> usize { self.capacity }

    /// Bytes currently handed out.
    pub fn used(&self) -> usize { self.cursor }

    /// Borrow the backing `MTLBuffer`.
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> { &self.buffer }
}
