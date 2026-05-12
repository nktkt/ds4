//! Backend dispatch trait. The Metal, CUDA and CPU implementations all expose
//! the same trait so [`crate::session::Session`] is unaware of which one it's
//! talking to.
//!
//! This is the Rust analogue of the `ds4_gpu.h` function-pointer table in the
//! C source — the C version uses function pointers because it's a single
//! object file; we use a trait because the impls live in separate crates.

use crate::api::{Backend, Tokens, TokenScore};
use crate::model::Config;
use anyhow::Result;

#[derive(Debug)]
pub struct PrefillStats {
    pub tokens: u32,
    pub elapsed_ms: f64,
}

pub trait BackendImpl: Send {
    fn name(&self) -> Backend;
    /// Reset session state.
    fn reset(&mut self);
    /// Run a prefill over `tokens[start..]` (i.e. resume from `start`).
    fn prefill(&mut self, tokens: &Tokens, start: u32) -> Result<PrefillStats>;
    /// Decode a single token and append to KV.
    fn decode(&mut self, token: i32) -> Result<()>;
    /// Get current logits.
    fn logits(&self) -> &[f32];
    /// Rewind to `pos` tokens (truncate KV). May fail with
    /// [`anyhow::Error`]; the session treats that as "rebuild required".
    fn rewind(&mut self, pos: u32) -> Result<()>;
    /// Sum of accepted tokens (for stats).
    fn n_tokens(&self) -> u32;
    /// Encode a session payload (raw + compressed KV). Returns the bytes to
    /// persist; format is opaque to the caller.
    fn save_payload(&self) -> Result<Vec<u8>>;
    /// Restore a session payload previously produced by `save_payload`.
    fn load_payload(&mut self, bytes: &[u8]) -> Result<()>;
    /// Approximate logprob ranking (used by `top_logprobs`).
    fn top_logprobs(&self, k: usize) -> Vec<TokenScore>;
}

/// Build a backend instance for the given `Backend` choice. Backends that
/// aren't compiled in return `Error::Unsupported`.
pub fn open(backend: Backend, cfg: &Config, ctx_size: u32) -> Result<Box<dyn BackendImpl>> {
    let _ = (cfg, ctx_size);
    match backend {
        Backend::Metal => {
            #[cfg(feature = "metal")]
            { return crate::backend::metal::open(cfg, ctx_size); }
            #[cfg(not(feature = "metal"))]
            anyhow::bail!("backend metal: not built into this binary")
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            { return crate::backend::cuda::open(cfg, ctx_size); }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("backend cuda: not built into this binary")
        }
        Backend::Cpu => {
            cpu::open(cfg, ctx_size)
        }
    }
}

pub mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;
