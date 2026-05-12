//! Session — Rust mirror of `ds4_session_*` from `ds4.c`.
//!
//! A session owns the live KV cache for one inference timeline. The caller
//! provides a full prompt as a `Tokens` slice and asks the session to "sync":
//! the session figures out how much of its current KV cache still matches the
//! prompt (the common prefix), keeps that, throws away the rest, and prefills
//! only the suffix. This is the central optimization that makes interactive
//! agent loops cheap.

use crate::api::{Tokens, TokenScore, ProgressSink, SessionRewriteResult};
use crate::backend::BackendImpl;
use crate::sampler;
use crate::tokens::common_prefix_len;
use anyhow::Result;
use rand::SeedableRng;

pub struct Session {
    backend: Box<dyn BackendImpl>,
    tokens: Tokens,
    ctx_size: u32,
    progress: Option<Box<dyn ProgressSink>>,
}

impl Session {
    pub fn new(backend: Box<dyn BackendImpl>, ctx_size: u32) -> Self {
        Self { backend, tokens: Tokens::new(), ctx_size, progress: None }
    }

    pub fn set_progress<P: ProgressSink + 'static>(&mut self, p: P) {
        self.progress = Some(Box::new(p));
    }

    pub fn pos(&self) -> u32 { self.backend.n_tokens() }
    pub fn ctx(&self) -> u32 { self.ctx_size }
    pub fn tokens(&self) -> &Tokens { &self.tokens }
    pub fn invalidate(&mut self) {
        self.backend.reset();
        self.tokens.clear();
    }

    /// Find the longest token prefix of `prompt` that we have already
    /// committed to KV. Mirrors `ds4_session_common_prefix`.
    pub fn common_prefix(&self, prompt: &Tokens) -> usize {
        common_prefix_len(self.tokens.as_slice(), prompt.as_slice())
    }

    /// Returns true if the engine cannot rewrite the live KV state at this
    /// gap. Mirrors `ds4_session_rewrite_requires_rebuild`.
    pub fn rewrite_requires_rebuild(live_len: i32, canonical_len: i32, common: i32) -> bool {
        // Original heuristic: a rewrite is safe only if we still hold at least
        // `common` tokens and the live tail is a strict superset of the
        // canonical tail. Anything that would require splicing tokens into
        // the *middle* of the KV cache forces a full rebuild.
        live_len < common || canonical_len < common
    }

    pub fn rewrite_from_common(
        &mut self,
        prompt: &Tokens,
        common: u32,
    ) -> Result<SessionRewriteResult> {
        if Self::rewrite_requires_rebuild(
            self.backend.n_tokens() as i32,
            prompt.len() as i32,
            common as i32,
        ) {
            return Ok(SessionRewriteResult::RebuildNeeded);
        }
        self.backend.rewind(common)?;
        self.tokens = Tokens::from_vec(prompt.as_slice()[..common as usize].to_vec());
        // Append the suffix.
        if (prompt.len() as u32) > common {
            self.backend.prefill(prompt, common)?;
            self.tokens.extend(&prompt.as_slice()[common as usize..]);
        }
        Ok(SessionRewriteResult::Ok)
    }

    /// Synchronize live state to `prompt`. Mirrors `ds4_session_sync`.
    pub fn sync(&mut self, prompt: &Tokens) -> Result<()> {
        let common = self.common_prefix(prompt) as u32;
        if let Some(p) = self.progress.as_deref_mut() {
            p.report("sync", common as i32, prompt.len() as i32);
        }
        match self.rewrite_from_common(prompt, common)? {
            SessionRewriteResult::Ok => Ok(()),
            SessionRewriteResult::RebuildNeeded => {
                self.invalidate();
                self.backend.prefill(prompt, 0)?;
                self.tokens.copy_from(prompt);
                Ok(())
            }
        }
    }

    pub fn argmax(&self) -> i32 {
        sampler::argmax(self.backend.logits())
    }
    pub fn argmax_excluding(&self, excluded: i32) -> i32 {
        sampler::argmax_excluding(self.backend.logits(), excluded)
    }
    pub fn sample(&self, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: &mut u64) -> i32 {
        let mut logits = self.backend.logits().to_vec();
        let mut prng = rand_xoshiro::SplitMix64::seed_from_u64(*rng);
        let tok = sampler::sample(&mut logits, temperature, top_k, top_p, min_p, &mut prng);
        // Advance the seed by drawing a fresh u64 (so the caller's `rng` cursor
        // moves deterministically).
        use rand::RngCore;
        *rng = prng.next_u64();
        tok
    }
    pub fn top_logprobs(&self, k: usize) -> Vec<TokenScore> {
        self.backend.top_logprobs(k)
    }
    pub fn eval(&mut self, token: i32) -> Result<()> {
        self.backend.decode(token)?;
        self.tokens.push(token);
        Ok(())
    }
    pub fn eval_speculative_argmax(
        &mut self,
        first_token: i32,
        max_tokens: i32,
        eos_token: i32,
        accepted: &mut Vec<i32>,
    ) -> Result<i32> {
        // Reference implementation: no draft model — fall back to plain greedy
        // decode, returning at most `max_tokens` tokens or until we hit
        // `eos_token`. The optimized speculative path is wired in once the
        // MTP backend lands.
        let _ = first_token;
        accepted.clear();
        for _ in 0..max_tokens {
            let t = self.argmax();
            accepted.push(t);
            self.eval(t)?;
            if t == eos_token { break; }
        }
        Ok(accepted.len() as i32)
    }
    pub fn rewind(&mut self, pos: u32) {
        let _ = self.backend.rewind(pos);
        self.tokens.as_mut_slice(); // ensure mutability
        let v = self.tokens.clone().into_vec();
        let trimmed: Vec<i32> = v.into_iter().take(pos as usize).collect();
        self.tokens = Tokens::from_vec(trimmed);
    }

    pub fn payload_bytes(&self) -> Result<Vec<u8>> { self.backend.save_payload() }
    pub fn load_payload(&mut self, bytes: &[u8]) -> Result<()> { self.backend.load_payload(bytes) }
}
