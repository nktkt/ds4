//! CPU reference backend.
//!
//! In the original C codebase the CPU path was kept around as a slow,
//! readable reference implementation against which the Metal/CUDA outputs
//! get diff'd. We do the same: this backend never wins a race against a
//! real GPU but it's the canonical algorithmic reference, intentionally
//! written without microoptimization.
//!
//! TODO: port the actual matmuls / RMSNorm / softmax / RoPE / MLA from
//! `ds4.c` CPU section. This stub builds and reports its presence so the
//! workspace compiles and the `--backend cpu` dispatch path exercises the
//! trait without panicking.

use crate::api::{Backend, Tokens, TokenScore};
use crate::backend::{BackendImpl, PrefillStats};
use crate::model::Config;
use anyhow::{anyhow, Result};

pub fn open(_cfg: &Config, ctx_size: u32) -> Result<Box<dyn BackendImpl>> {
    Ok(Box::new(CpuBackend {
        logits: Vec::new(),
        n_tokens: 0,
        ctx_size,
    }))
}

pub struct CpuBackend {
    logits: Vec<f32>,
    n_tokens: u32,
    ctx_size: u32,
}

impl BackendImpl for CpuBackend {
    fn name(&self) -> Backend { Backend::Cpu }
    fn reset(&mut self) { self.n_tokens = 0; self.logits.clear(); }
    fn prefill(&mut self, tokens: &Tokens, start: u32) -> Result<PrefillStats> {
        let n = tokens.len() as u32 - start;
        if self.n_tokens + n > self.ctx_size {
            return Err(anyhow!("cpu backend: context overflow"));
        }
        self.n_tokens += n;
        // TODO: actual prefill
        Ok(PrefillStats { tokens: n, elapsed_ms: 0.0 })
    }
    fn decode(&mut self, _token: i32) -> Result<()> {
        if self.n_tokens >= self.ctx_size {
            return Err(anyhow!("cpu backend: context full"));
        }
        self.n_tokens += 1;
        Ok(())
    }
    fn logits(&self) -> &[f32] { &self.logits }
    fn rewind(&mut self, pos: u32) -> Result<()> {
        if pos > self.n_tokens { return Err(anyhow!("rewind past end")); }
        self.n_tokens = pos;
        Ok(())
    }
    fn n_tokens(&self) -> u32 { self.n_tokens }
    fn save_payload(&self) -> Result<Vec<u8>> { Ok(Vec::new()) }
    fn load_payload(&mut self, _bytes: &[u8]) -> Result<()> { Ok(()) }
    fn top_logprobs(&self, _k: usize) -> Vec<TokenScore> { Vec::new() }
}
