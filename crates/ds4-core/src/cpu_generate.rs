//! End-to-end CPU generation driver.
//!
//! Single entry point that takes the weights it needs to run the output head,
//! a prompt, an `n_predict` count, and a per-layer compute closure, then
//! drives the layer loop + output head + argmax sampler until either
//! `n_predict` tokens are emitted or the model produces `eos_token`.
//!
//! Ports the shape of three C routines in `ds4.c`:
//!
//! * `forward_first_token_cpu`            (`ds4.c` ~L7891) — embed token,
//!    replicate into HC streams, run every layer once, leave the final HC in
//!    `cur`.
//! * `forward_token_raw_swa_cpu`          (`ds4.c` ~L7649) — same data flow
//!    but for a decode step; in the Rust slice we collapse the prefill and
//!    decode paths into the same per-token routine because the per-layer
//!    closure abstracts away whether a KV cache update is happening.
//! * `ds4_engine_generate_argmax` / `generate_raw_swa_cpu` (`ds4.c` ~L16298
//!    and ~L14884) — the top-level driver: prefill, then loop sampling argmax
//!    tokens and feeding each one back through the layer stack.
//!
//! The C version owns a `ds4_kv_cache` and a `ds4_cpu_decode_scratch` and
//! threads them through `layer_forward_raw_swa_one`. In Rust we keep this
//! module free of weight-binder dependencies by accepting a `layer_apply`
//! closure that the caller wires up to whatever per-layer compute it owns
//! (typically `crate::layer_forward::layer_forward_self_one` plus a
//! `CpuScratch`). This keeps `cpu_generate` decoupled from the attention /
//! MoE specifics and makes it trivially unit-testable with an identity
//! layer.

use crate::api::{TokenSink, Tokens};
use crate::hc::{embed_token_f16, hc_from_plain};
use crate::output_head::output_logits_one;
use crate::sampler::argmax;
use crate::shape::{N_EMBD, N_HC, N_LAYER, N_VOCAB};

/// Reusable scratch and state buffers for [`generate_argmax_cpu`].
///
/// Mirrors the role of `ds4_cpu_decode_scratch` plus the per-call `cur` /
/// `next` / `logits` allocations in `forward_first_token_cpu` and
/// `generate_raw_swa_cpu` (`ds4.c` ~L7891 and ~L14884). All buffers are
/// allocated at full DS4 dimensions on construction so the generation loop
/// is allocation-free.
pub struct CpuGenerateCtx {
    /// Current HC state, laid out as `[N_HC, N_EMBD]` row-major.
    pub cur: Vec<f32>,
    /// Scratch HC state used as the destination of each `layer_apply` call;
    /// swapped with `cur` after every layer.
    pub next: Vec<f32>,
    /// Output-head logits buffer, length `N_VOCAB`.
    pub logits: Vec<f32>,
    /// Token-embedding scratch, length `N_EMBD`.
    pub plain: Vec<f32>,
}

impl CpuGenerateCtx {
    /// Allocate every buffer at full DS4 dimensions.
    ///
    /// `cur` and `next` are `N_HC * N_EMBD` (HC state for one token), `logits`
    /// is `N_VOCAB`, and `plain` is `N_EMBD` — matching the explicit
    /// `xmalloc` calls in `forward_first_token_cpu` (`ds4.c` ~L7896).
    pub fn new() -> Self {
        let n_embd = N_EMBD as usize;
        let n_hc = N_HC as usize;
        let n_vocab = N_VOCAB as usize;
        Self {
            cur: vec![0.0; n_hc * n_embd],
            next: vec![0.0; n_hc * n_embd],
            logits: vec![0.0; n_vocab],
            plain: vec![0.0; n_embd],
        }
    }
}

impl Default for CpuGenerateCtx {
    fn default() -> Self {
        Self::new()
    }
}

/// Embed `token`, fan it out into HC streams, and run every layer once,
/// leaving the final HC in `ctx.cur`. Used both for prompt prefill (one call
/// per prompt token) and for the decode step (one call per emitted token).
///
/// Mirrors the per-token body shared by `forward_first_token_cpu` (`ds4.c`
/// ~L7891) and `forward_token_raw_swa_cpu_decode_scratch` (`ds4.c` ~L7614):
/// `embed_token_f16` → `hc_from_plain_embedding` → `N_LAYER` iterations of
/// `layer_forward_*_one` with cur/next swapping.
fn run_token_through_layers(
    token: i32,
    pos: u64,
    tok_embed: &[u16],
    ctx: &mut CpuGenerateCtx,
    layer_apply: &mut dyn FnMut(&mut [f32], &[f32], u32, u64, i32),
) {
    let n_embd = N_EMBD as usize;
    embed_token_f16(&mut ctx.plain, tok_embed, n_embd, token as usize);
    hc_from_plain(&mut ctx.cur, &ctx.plain);
    for il in 0..N_LAYER {
        layer_apply(&mut ctx.next, &ctx.cur, il, pos, token);
        std::mem::swap(&mut ctx.cur, &mut ctx.next);
    }
}

/// Drive end-to-end CPU generation with argmax sampling.
///
/// Steps:
///
/// 1. **Prefill** — walk the prompt one token at a time, running each token
///    through the embedding, HC fan-out, and full `N_LAYER` layer stack via
///    `layer_apply`. After the loop, `ctx.cur` holds the final HC state for
///    the last prompt token.
/// 2. **Decode loop** — repeat at most `n_predict` times:
///    a. Compute logits from `ctx.cur` via [`output_logits_one`].
///    b. Argmax-sample a token.
///    c. If it equals `eos_token`, stop.
///    d. Emit it via `sink`.
///    e. If this is the last iteration, stop (no need to run the layers for
///       a token we will never sample from).
///    f. Otherwise, embed the new token and run it through the layers,
///       updating `ctx.cur` in place.
///
/// Returns the number of tokens emitted.
///
/// Mirrors `ds4_engine_generate_argmax` (`ds4.c` ~L16298) dispatched to
/// `generate_raw_swa_cpu` (`ds4.c` ~L14884). In the C source the prefill is
/// done in layer-major order to expose batch-matmul opportunities; here we
/// do per-token prefill because the `layer_apply` closure is opaque and may
/// own its own batching strategy.
///
/// `layer_apply(next, cur, il, pos, token)` is the caller-supplied per-layer
/// compute: it reads `cur` (the HC state coming into layer `il`) and writes
/// `next` (the HC state leaving layer `il`) for the token at absolute
/// position `pos`. The caller threads its own KV cache / weights through the
/// closure; this driver does not assume any particular state.
pub fn generate_argmax_cpu(
    prompt: &Tokens,
    tok_embed: &[u16],
    output_norm: &[f32],
    output: &[u8],
    output_hc_fn: &[u16],
    output_hc_scale: &[f32],
    output_hc_base: &[f32],
    n_predict: i32,
    eos_token: i32,
    ctx: &mut CpuGenerateCtx,
    mut layer_apply: impl FnMut(&mut [f32], &[f32], u32, u64, i32),
    sink: &mut dyn TokenSink,
) -> Result<i32, String> {
    if prompt.is_empty() {
        return Err("generate_argmax_cpu: prompt is empty".to_string());
    }

    // Prefill: run every prompt token through the full layer stack. The C
    // driver's `prefill_layer_major_cpu` does this layer-major; we do it
    // token-major because the closure is opaque. Either order produces the
    // same final HC for the last prompt token assuming a causal layer.
    let prompt_slice = prompt.as_slice();
    for (i, &tok) in prompt_slice.iter().enumerate() {
        run_token_through_layers(tok, i as u64, tok_embed, ctx, &mut layer_apply);
    }

    let mut pos = prompt_slice.len() as u64;
    let mut n_generated: i32 = 0;

    for i in 0..n_predict {
        // Output head: HC collapse + RMSNorm + Q8_0 vocab projection.
        output_logits_one(
            &mut ctx.logits,
            &ctx.cur,
            output_hc_fn,
            output_hc_scale,
            output_hc_base,
            output_norm,
            output,
        );

        let token = argmax(&ctx.logits);
        if token == eos_token {
            break;
        }

        sink.emit(token);
        n_generated += 1;

        // If this is the last predicted token, there is no need to push it
        // back through the layer stack — its successor will never be sampled.
        if i == n_predict - 1 {
            break;
        }

        run_token_through_layers(token, pos, tok_embed, ctx, &mut layer_apply);
        pos += 1;
    }

    sink.done();
    Ok(n_generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;
    use crate::matvec::Q8_0_BLOCK_BYTES;

    /// (a) `CpuGenerateCtx::new` allocates each buffer at full DS4 dimensions.
    #[test]
    fn ctx_new_allocates_full_dims() {
        let ctx = CpuGenerateCtx::new();
        assert_eq!(ctx.cur.len(), (N_HC * N_EMBD) as usize);
        assert_eq!(ctx.next.len(), (N_HC * N_EMBD) as usize);
        assert_eq!(ctx.logits.len(), N_VOCAB as usize);
        assert_eq!(ctx.plain.len(), N_EMBD as usize);
    }

    /// (b) With a frozen logit distribution (output head sees the same `cur`
    /// at every decode step because `layer_apply` is the identity), argmax
    /// must return the same token on every step. We rig the Q8_0 output
    /// matrix so that exactly one row has a non-zero scale — that row's
    /// index is the deterministic argmax. The sink records emissions and we
    /// verify the emitted sequence is exactly `[winner; n_predict]`.
    #[test]
    fn generate_with_identity_layer_emits_same_token_each_step() {
        let n_embd = N_EMBD as usize;
        let n_hc = N_HC as usize;
        let n_vocab = N_VOCAB as usize;

        // tok_embed: a single-row table is enough — every prompt/decoded
        // token id we use is 0, so we only need row 0. We size it to one row
        // because `embed_token_f16` only reads `row * n_embd .. (row+1) *
        // n_embd`. Row 0 is a small non-zero pattern so RMSNorm has signal.
        let mut tok_embed = vec![0u16; n_embd];
        for i in 0..n_embd {
            tok_embed[i] = f32_to_f16(0.01 * ((i % 17) as f32 - 8.0));
        }

        // Output-head weights: norm = ones, hc_fn = zeros, hc_scale = [1],
        // hc_base = zeros. With zero `hc_fn` the sigmoid-gated head weights
        // are sigmoid(0) + eps for every stream, so the HC collapse is a
        // simple uniform weighted sum. The exact value of `cur` does not
        // matter for the test — only that the final argmax is deterministic.
        let output_norm = vec![1.0f32; n_embd];
        let output_hc_fn = vec![0u16; n_hc * n_hc * n_embd];
        let output_hc_scale = vec![1.0f32];
        let output_hc_base = vec![0.0f32; n_hc];

        // Q8_0 output matrix: every row is all-zero quants with scale 1.0,
        // *except* row `WINNER`, whose first quant byte is +127 (so the dot
        // product against any positive-leaning vector is > 0) and whose
        // scale is also 1.0. Concretely, every other row dot-products to
        // exactly 0, so `WINNER` is the unique argmax irrespective of the
        // exact contents of `cur`.
        // Keep WINNER == prompt-token so the 1-row tok_embed covers every
        // step (initial prompt + each decoded copy of itself).
        const WINNER: usize = 0;
        let blocks = (n_embd + 31) / 32;
        let row_bytes = blocks * Q8_0_BLOCK_BYTES;
        let mut output = vec![0u8; n_vocab * row_bytes];
        let scale_bytes = f32_to_f16(1.0).to_le_bytes();
        for r in 0..n_vocab {
            for b in 0..blocks {
                let base = r * row_bytes + b * Q8_0_BLOCK_BYTES;
                output[base] = scale_bytes[0];
                output[base + 1] = scale_bytes[1];
            }
        }
        // Give the winning row a strictly positive quant for every block.
        // Even if some `norm` entries are negative, the sum of 32 i8=+127
        // dot products across many blocks dominates the other rows' zero
        // dot products with overwhelming margin in the common case; and to
        // make the test 100% deterministic we also bias `cur` to a constant
        // positive vector below, so RMSNorm produces a constant positive
        // `norm` and the dot product is strictly positive.
        for b in 0..blocks {
            let base = WINNER * row_bytes + b * Q8_0_BLOCK_BYTES;
            for q in 0..32 {
                output[base + 2 + q] = 127;
            }
        }

        // Build a context whose `cur` we will preload with all-ones so the
        // identity layer keeps it all-ones forever, the HC collapse stays
        // all-ones (uniform head weights), RMSNorm of all-ones is all-ones,
        // and the matvec sees a strictly positive input. We do this by
        // making `layer_apply` ignore the input and write all-ones into
        // `next` on every layer. That mimics a frozen-state identity for
        // the purposes of this test.
        let mut ctx = CpuGenerateCtx::new();
        let layer_apply = |next: &mut [f32], _cur: &[f32], _il: u32, _pos: u64, _tok: i32| {
            for v in next.iter_mut() {
                *v = 1.0;
            }
        };

        // Prompt is a single token (id 0). `eos_token` is something we will
        // never sample, so it cannot terminate the loop early.
        let prompt = Tokens::from_vec(vec![0]);
        let n_predict = 5;
        let eos_token = -1;

        struct CollectSink {
            out: Vec<i32>,
        }
        impl TokenSink for CollectSink {
            fn emit(&mut self, t: i32) {
                self.out.push(t);
            }
        }
        let mut sink = CollectSink { out: Vec::new() };

        let n = generate_argmax_cpu(
            &prompt,
            &tok_embed,
            &output_norm,
            &output,
            &output_hc_fn,
            &output_hc_scale,
            &output_hc_base,
            n_predict,
            eos_token,
            &mut ctx,
            layer_apply,
            &mut sink,
        )
        .expect("generation should succeed");

        assert_eq!(n, n_predict);
        assert_eq!(sink.out.len(), n_predict as usize);
        for (i, &t) in sink.out.iter().enumerate() {
            assert_eq!(t, WINNER as i32, "step {i}: expected token {WINNER}, got {t}");
        }
    }
}
