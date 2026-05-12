//! Metal runtime: device, command queue, library loading. Ports
//! `ds4_metal.m::ds4_metal_init` / `ds4_metal_queue_*` — the C globals
//! `g_device`, `g_queue`, `g_library` (lines 34-36 of `ds4_metal.m`) are
//! collapsed into a single `MetalRuntime` value the caller owns.
//!
//! All `unsafe` Objective-C message-sends are confined to small, clearly
//! named functions; the rest of the crate only sees safe Rust wrappers.

// The crate's `lib.rs` already gates this whole module on macOS, but we
// repeat the attribute here so that opening the file in isolation makes the
// platform requirement obvious.
#![cfg(target_os = "macos")]

use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};

// `MTLCreateSystemDefaultDevice` lives in the Metal framework but the symbol
// table is satisfied by CoreGraphics on macOS — link it explicitly so we
// don't have to pull in `objc2-app-kit` just for that side effect.
// See objc2-metal's crate-level docs.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// Owning handle on the three Metal objects every kernel dispatch needs:
/// the default `MTLDevice`, a single `MTLCommandQueue`, and a `MTLLibrary`
/// loaded from a precompiled `.metallib` on disk.
///
/// Mirrors `g_device` / `g_queue` / `g_library` in `ds4_metal.m`.
pub struct MetalRuntime {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    metallib_path: PathBuf,
}

impl MetalRuntime {
    /// Build a runtime from the system-default Metal device and a `.metallib`
    /// file on disk. Equivalent to `ds4_metal.m::ds4_metal_init` minus the
    /// model-residency / mmap bookkeeping (those land in `buffers.rs`).
    pub fn open(metallib_path: &Path) -> Result<Self> {
        if !metallib_path.exists() {
            bail!("ds4-metal: metallib not found at {}", metallib_path.display());
        }

        let device = default_device()
            .context("ds4-metal: MTLCreateSystemDefaultDevice() returned nil")?;
        let queue = make_command_queue(&device)
            .context("ds4-metal: [device newCommandQueue] returned nil")?;
        let library = load_library_from_url(&device, metallib_path)
            .with_context(|| format!(
                "ds4-metal: failed to load metallib at {}",
                metallib_path.display()))?;

        Ok(Self {
            device,
            queue,
            library,
            metallib_path: metallib_path.to_owned(),
        })
    }

    /// Borrow the underlying `MTLDevice` (no retain).
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Borrow the shared `MTLCommandQueue`.
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Borrow the `MTLLibrary` containing all DS4 compute kernels.
    pub fn library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }

    /// Path the metallib was loaded from. Useful for diagnostics.
    pub fn metallib_path(&self) -> &Path {
        &self.metallib_path
    }

    /// Human-readable Metal device name (the `[device name]` property).
    pub fn device_name(&self) -> String {
        // `name` is a non-`unsafe` accessor in the bindings.
        self.device.name().to_string()
    }

    /// Largest single `MTLBuffer` the device will allocate, in bytes.
    /// Used by the model-view splitter in `buffers.rs`.
    pub fn max_buffer_length(&self) -> usize {
        self.device.maxBufferLength() as usize
    }
}

// ---------- safe wrappers over the unsafe Metal entry points ----------

/// Resolve the system-default `MTLDevice`. Returns `None` if Metal is
/// unavailable (e.g. running under a headless CI VM).
fn default_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    // SAFETY: `MTLCreateSystemDefaultDevice` returns a +1-retained
    // `id<MTLDevice>` or nil. `Retained::from_raw` consumes that ownership.
    unsafe {
        let raw = MTLCreateSystemDefaultDevice();
        if raw.is_null() {
            None
        } else {
            Retained::from_raw(raw)
        }
    }
}

/// Create a fresh `MTLCommandQueue` on `device`.
fn make_command_queue(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<Retained<ProtocolObject<dyn MTLCommandQueue>>> {
    device.newCommandQueue()
}

/// `[device newLibraryWithURL:fileURLWithPath:error:]`.
fn load_library_from_url(
    device: &ProtocolObject<dyn MTLDevice>,
    path: &Path,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    // Build an NSString and an NSURL from the OS path.
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("metallib path is not valid UTF-8: {:?}", path))?;
    let ns_path = NSString::from_str(path_str);
    // SAFETY: NSURL::fileURLWithPath is declared `unsafe` only because it
    // dereferences the NSString it receives; we pass a valid retained one.
    let url: Retained<NSURL> = unsafe { NSURL::fileURLWithPath(&ns_path) };

    // SAFETY: `newLibraryWithURL:error:` takes an NSURL by reference and
    // returns either a retained library or an NSError; both ownership rules
    // are handled by `objc2`'s `method_id` shim.
    unsafe { device.newLibraryWithURL_error(&url) }
        .map_err(|e| anyhow!("Metal newLibraryWithURL failed: {}", e))
}
