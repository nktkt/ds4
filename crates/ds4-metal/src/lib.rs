//! ds4-metal — Metal backend for DwarfStar 4 (macOS only).
//!
//! Port of `ds4_metal.m`. The original is a single Objective-C translation
//! unit that loads `.metal` shaders, builds command queues, allocates per-graph
//! buffers, and dispatches the DS4 graph. The Rust side mirrors that one to
//! one but uses `objc2-metal` for the Objective-C bridging.

#[cfg(target_os = "macos")]
pub mod runtime;
#[cfg(target_os = "macos")]
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod graph;
#[cfg(target_os = "macos")]
pub mod buffers;

#[cfg(not(target_os = "macos"))]
mod stub {
    //! On non-Apple targets the Metal backend is unavailable. We keep the
    //! crate compilable so the workspace builds, but every entry point
    //! returns an "unsupported" error.
    use anyhow::bail;
    pub fn available() -> bool { false }
    pub fn ensure_available() -> anyhow::Result<()> {
        bail!("ds4-metal: Metal backend is only available on macOS");
    }
}
#[cfg(not(target_os = "macos"))]
pub use stub::*;
