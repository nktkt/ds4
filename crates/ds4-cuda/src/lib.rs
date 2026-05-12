//! ds4-cuda — CUDA backend for DwarfStar 4 (Linux + NVIDIA).
//!
//! Port of `ds4_cuda.cu`. CUDA kernels themselves stay in `.cu` form and are
//! compiled by `build.rs` via `nvcc`, then linked in. The Rust side uses
//! `cudarc` for context / stream / memory management and kernel launches.

#[cfg(target_os = "linux")]
pub mod runtime;
#[cfg(target_os = "linux")]
pub mod kernels;
#[cfg(target_os = "linux")]
pub mod graph;

#[cfg(not(target_os = "linux"))]
mod stub {
    use anyhow::bail;
    pub fn available() -> bool { false }
    pub fn ensure_available() -> anyhow::Result<()> {
        bail!("ds4-cuda: CUDA backend is only built on Linux");
    }
}
#[cfg(not(target_os = "linux"))]
pub use stub::*;
