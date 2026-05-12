//! Context-window memory estimator (`ds4_context_memory_estimate`).
//!
//! Mirrors the per-backend table baked into `ds4.c`. The exact numbers come
//! from the DS4 graph: tokens × per-token-bytes for the raw and compressed
//! KV streams, plus a constant scratch reservation that the Metal/CUDA
//! command buffers allocate. We keep the constants in one place so the CLI
//! and server can both quote the same memory budget back to the user.

use crate::api::{Backend, ContextMemory};

// Per-token bytes for the DS4 KV streams. These are the values from the
// original C source; they intentionally do not scale with batch size.
const RAW_BYTES_PER_TOKEN: u64 = 7168 * 2;       // f16
const COMP_BYTES_PER_TOKEN: u64 = 512 + 64;      // compressed kv + meta
const METAL_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;
const CUDA_SCRATCH_BYTES: u64 = 384 * 1024 * 1024;
const CPU_SCRATCH_BYTES: u64 = 128 * 1024 * 1024;

pub fn estimate(backend: Backend, ctx_size: i32) -> ContextMemory {
    let ctx = ctx_size.max(0) as u64;
    let raw  = ctx * RAW_BYTES_PER_TOKEN;
    let comp = ctx * COMP_BYTES_PER_TOKEN;
    let scratch = match backend {
        Backend::Metal => METAL_SCRATCH_BYTES,
        Backend::Cuda  => CUDA_SCRATCH_BYTES,
        Backend::Cpu   => CPU_SCRATCH_BYTES,
    };
    ContextMemory {
        total_bytes: raw + comp + scratch,
        raw_bytes: raw,
        compressed_bytes: comp,
        scratch_bytes: scratch,
        prefill_cap: ctx_size as u32,
        raw_cap: ctx_size as u32,
        comp_cap: ctx_size as u32,
    }
}
