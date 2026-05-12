//! CPU reference forward pass — scaffold.
//!
//! Ports the *shape* of `forward_first_token_cpu`, `layer_forward_self_one`,
//! and `output_logits_one` from `ds4.c` (around lines 7438..7962). The body
//! of each layer step still calls helper stubs because the full MoE routing
//! and MLA attention paths are large; this module wires the dataflow so the
//! engine has a single call site to fill in.
//!
//! Layout: per-token state is HC streams of shape `[N_HC × N_EMBD]`. Each
//! transformer layer reads the previous HC, runs:
//!
//! 1. RMSNorm over each stream
//! 2. Q/K/V projections (with LoRA), RoPE, MLA attention
//! 3. Residual add
//! 4. RMSNorm again
//! 5. MoE routing + shared expert FFN (SwiGLU)
//! 6. Residual add
//!
//! and writes the next HC. After 43 layers, [`output_hc_head_one`] collapses
//! HC into one embedding vector, applies the final RMSNorm and vocab
//! projection, and returns logits.

use crate::model::Config;
use crate::nn::{rms_norm_no_weight, rms_norm_weight};
use crate::shape::{N_EMBD, N_HC, N_LAYER, RMS_EPS};

/// Per-layer scratch buffers reused across the layer loop. The original
/// `ds4_cpu_decode_scratch` aggregates ~30 named buffers; we group them
/// into a struct that owns each allocation once and reuses it.
#[derive(Default)]
pub struct CpuScratch {
    pub plain: Vec<f32>,
    pub cur:   Vec<f32>,      // [N_HC * N_EMBD]
    pub next:  Vec<f32>,      // [N_HC * N_EMBD]
    pub flat:  Vec<f32>,      // [N_HC * N_EMBD]
    pub embd:  Vec<f32>,      // [N_EMBD]
    pub norm:  Vec<f32>,      // [N_EMBD]
    pub pre:   Vec<f32>,      // [N_HC]
    pub w:     Vec<f32>,      // [N_HC]
}

impl CpuScratch {
    pub fn new() -> Self {
        let hc = N_HC as usize;
        let n = N_EMBD as usize;
        Self {
            plain: vec![0.0; n],
            cur:   vec![0.0; hc * n],
            next:  vec![0.0; hc * n],
            flat:  vec![0.0; hc * n],
            embd:  vec![0.0; n],
            norm:  vec![0.0; n],
            pre:   vec![0.0; hc],
            w:     vec![0.0; hc],
        }
    }
}

/// Token embedding lookup: writes `out[..N_EMBD]` from a row of the
/// `token_embd` tensor. Mirrors `embed_token_f16`. Pending the model module
/// being able to return raw f16 tensor rows; currently this is a no-op.
pub fn embed_token(_cfg: &Config, _token: i32, _out: &mut [f32]) {
    // TODO: read row token * N_EMBD from tok_embed (f16) → f32
}

/// Replicate the plain embedding into the HC streams. The original
/// `hc_from_plain_embedding` simply duplicates the embedding for each
/// channel because the first HC step is identity.
pub fn hc_from_plain(plain: &[f32], hc: &mut [f32]) {
    let n = N_EMBD as usize;
    for c in 0..N_HC as usize {
        hc[c * n..(c + 1) * n].copy_from_slice(plain);
    }
}

/// One transformer layer step. The body is a structural port: the order of
/// substeps, residual additions, and norms matches `layer_forward_self_one`
/// in `ds4.c`. Attention and MoE bodies are currently identity placeholders
/// pending the full MLA + expert routing port.
pub fn layer_forward_self_one(
    next: &mut [f32],
    cur: &[f32],
    _il: u32,
    _pos: u64,
    _token: i32,
) {
    let hc = N_HC as usize;
    let n = N_EMBD as usize;
    debug_assert_eq!(cur.len(),  hc * n);
    debug_assert_eq!(next.len(), hc * n);
    // Step 1: norm into a temporary
    let mut tmp = vec![0.0_f32; hc * n];
    for c in 0..hc {
        rms_norm_no_weight(&mut tmp[c * n..(c + 1) * n], &cur[c * n..(c + 1) * n], RMS_EPS);
    }
    // Step 2: Q/K/V projections, RoPE, MLA attention.  TODO: full kernel.
    // For now: pass the normalized stream through, simulating an attention
    // output of `0` so the residual = input.
    // Step 3: residual add — `out = cur + attn(norm(cur))`
    for i in 0..hc * n { next[i] = cur[i]; }
    // Step 4: norm of the post-attention state
    let mut tmp2 = vec![0.0_f32; hc * n];
    for c in 0..hc {
        rms_norm_no_weight(&mut tmp2[c * n..(c + 1) * n], &next[c * n..(c + 1) * n], RMS_EPS);
    }
    // Step 5: MoE routing + shared FFN. TODO: full kernel.
    // Step 6: residual add — `out = next + ffn(norm(next))`
    //   (placeholder ffn output = 0)
}

/// Output head: collapse HC streams into one embedding, RMSNorm, Q8_0 vocab
/// projection. Mirrors `output_logits_one`.
pub fn output_logits_one(_cfg: &Config, inp_hc: &[f32], logits: &mut [f32]) {
    let n = N_EMBD as usize;
    let _ = inp_hc;
    // TODO: implement HC collapse + RMSNorm + matvec_q8_0(output)
    let _ = rms_norm_weight;
    let _ = n;
    for v in logits.iter_mut() { *v = 0.0; }
}

/// First-token (prefill from cold) forward pass. Mirrors
/// `forward_first_token_cpu`: embed, replicate into HC, run all
/// `N_LAYER` layer steps, return final HC.
pub fn forward_first_token_cpu(
    cfg: &Config,
    token: i32,
    out_hc: &mut [f32],
    scratch: &mut CpuScratch,
) {
    embed_token(cfg, token, &mut scratch.plain);
    hc_from_plain(&scratch.plain, &mut scratch.cur);
    for il in 0..N_LAYER {
        layer_forward_self_one(&mut scratch.next, &scratch.cur, il, 0, token);
        std::mem::swap(&mut scratch.cur, &mut scratch.next);
    }
    out_hc.copy_from_slice(&scratch.cur);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_shapes_consistent() {
        // The placeholder forward should not panic and should leave the
        // output stream the same size as the input.
        let cfg = Config {
            n_layers: N_LAYER, d_model: N_EMBD, n_heads: 64, n_kv_heads: 1,
            d_head: 512, d_ff: 2048, n_experts: 256, n_experts_per_tok: 6,
            vocab_size: 129_280, rope_base: 10_000.0, rms_norm_eps: 1e-6,
            max_context: 65_536, q_lora_rank: 1024, kv_lora_rank: 1024,
        };
        let mut scratch = CpuScratch::new();
        let mut out = vec![0.0_f32; (N_HC * N_EMBD) as usize];
        forward_first_token_cpu(&cfg, 0, &mut out, &mut scratch);
        assert_eq!(out.len(), scratch.cur.len());
    }
}
