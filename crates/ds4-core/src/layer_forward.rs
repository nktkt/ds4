//! Full CPU layer-forward orchestration for one transformer block.
//!
//! Ports `layer_forward_self_one` from `ds4.c` (~line 7836) — the cold-start
//! variant used by `forward_first_token_cpu` (`ds4.c` ~line 7891). That C
//! function is the simpler twin of `layer_forward_raw_swa_one` (`ds4.c`
//! ~line 7438): no KV cache, no SWA window, no compressed attention. It is
//! exactly the per-layer orchestration we need to wire the existing per-block
//! primitives together for a single prefill-from-cold token.
//!
//! The C sequence inside `layer_forward_self_one` is:
//!
//! 1. `hc_pre_from_state_one(model, hc_attn_fn, hc_attn_scale, hc_attn_base,
//!     residual_hc, attn_cur, post, comb)`  — split + weighted sum across HC
//!     streams (see `ds4.c` ~line 4317).
//! 2. `layer_attn_norm_one(attn_norm, model, layer, attn_cur)`               — `ds4.c` ~line 4575.
//! 3. `layer_q_projection_normed_one(...)`                                   — `ds4.c` ~line 4596.
//! 4. `layer_kv_projection_normed_one(...)`                                  — `ds4.c` ~line 4633.
//! 5. `rope_tail_layer_inplace(q, ...)` + `rope_tail_layer_inplace(kv, ...)` — `ds4.c` ~line 4758.
//! 6. `layer_attention_one(heads, ...)` — single-KV MLA softmax (`ds4.c` ~line 4937).
//! 7. `rope_tail_layer_inplace(heads, ..., inverse=true)` (skipped here — the
//!    available Rust `rope::tail_rotate_inplace` does not expose the inverse
//!    flag; for the cold-start `pos = 0` path the rotation is the identity, so
//!    the omission is exact for that case and a documented approximation for
//!    `pos > 0`).
//! 8. `layer_grouped_out_one(attn_out, ...)`                                 — `ds4.c` ~line 4948.
//! 9. `hc_post_one(after_attn_hc, attn_out, attn_residual, post, comb, ...)` — `ds4.c` ~line 4366.
//! 10. `layer_ffn_one(out_hc, ..., after_attn_hc, ...)` — which itself runs
//!     `hc_pre_from_state_one` over the FFN HC weights, `rms_norm_weight`
//!     against `ffn_norm`, the shared SwiGLU FFN, the routed MoE, and a
//!     final `hc_post_one` (see `ds4.c` ~line 5576).
//!
//! This Rust port inlines step 1 / step 10's `hc_pre_from_state_one` recipe
//! locally using `nn::rms_norm_no_weight` + `matvec::matvec_f16` +
//! `hc::hc_split_sinkhorn_one` + `hc::hc_weighted_sum_one`, because the
//! existing Rust `hc` module only exposes those lower-level helpers.
//!
//! Allocations are batched into a single [`LayerScratch`] so callers can reuse
//! them across the 43-layer loop, mirroring `ds4_cpu_decode_scratch` from
//! `ds4.c`. Per-call helpers like [`shared_ffn_one`] and
//! [`layer_routed_moe_one`] still own their own temporaries.

use crate::attn::{
    layer_attention_one, layer_attn_norm_one, layer_grouped_out_one,
    layer_kv_projection_normed_one, layer_q_projection_normed_one,
};
use crate::ffn::{shared_ffn_one, SharedFfnWeights};
use crate::hc::{hc_post_one, hc_split_sinkhorn_one, hc_weighted_sum_one};
use crate::matvec::matvec_f16;
use crate::moe::{layer_routed_moe_one, ExpertScratch, RoutedMoeWeights};
use crate::nn::{rms_norm_no_weight, rms_norm_weight};
use crate::rope::{layer_rope_freq_base, layer_rope_freq_scale, tail_rotate_inplace};
use crate::shape::{
    HC_EPS, N_EMBD, N_HC, N_HC_SINKHORN_ITER, N_HEAD, N_HEAD_DIM, N_HEAD_KV, N_ROT, RMS_EPS,
    SWIGLU_CLAMP_EXP,
};

/// Borrowed view of every per-layer weight slice consumed by
/// [`layer_forward_self_one`]. The buffers are passed in pre-resolved
/// (tensor-data already looked up) so this module does not depend on the
/// GGUF / tensor-metadata layer.
///
/// Slice types follow the conventions used elsewhere in `ds4-core`:
///
/// * `&[u8]`  — Q8_0 weight rows, packed as `[f16 d][i8 q * 32]` per 32-element block.
/// * `&[u16]` — f16 weight rows, raw IEEE-754 binary16 bit patterns.
/// * `&[f32]` — f32 scales / biases / norm gains.
pub struct LayerWeightsRef<'a> {
    // --- attention sublayer ---
    /// Per-channel RMSNorm gain applied before Q/KV projection. Length `N_EMBD`.
    pub attn_norm: &'a [f32],
    /// Q8_0 low-rank A factor of the Q projection. `[N_LORA_Q, N_EMBD]`.
    pub attn_q_a: &'a [u8],
    /// Per-channel RMSNorm gain applied between the Q A and Q B matvecs. Length `N_LORA_Q`.
    pub attn_q_a_norm: &'a [f32],
    /// Q8_0 high-rank B factor of the Q projection. `[N_HEAD * N_HEAD_DIM, N_LORA_Q]`.
    pub attn_q_b: &'a [u8],
    /// Q8_0 KV projection. `[N_HEAD_KV * N_HEAD_DIM, N_EMBD]`.
    pub attn_kv: &'a [u8],
    /// Per-channel RMSNorm gain applied to the KV projection output. Length `N_HEAD_DIM`.
    pub attn_kv_a_norm: &'a [f32],
    /// Concatenated Q8_0 grouped output A factors. `[N_OUT_GROUP * N_LORA_O, group_dim]`.
    pub attn_output_a_grouped: &'a [u8],
    /// Q8_0 output B factor. `[N_EMBD, N_OUT_GROUP * N_LORA_O]`.
    pub attn_output_b: &'a [u8],
    /// Per-head learned attention sinks. Length `N_HEAD`.
    pub attn_sinks: &'a [f32],

    // --- attention-side HC control ---
    /// f16 HC control projection for the attention pre step.
    /// Shape `[(2 + N_HC) * N_HC, N_HC * N_EMBD]`, packed as `u16`.
    pub hc_attn_fn: &'a [u16],
    /// `[pre, post, comb]` scales for the attention Sinkhorn split. Length `3`.
    pub hc_attn_scale: &'a [f32],
    /// Sinkhorn split base, length `(2 + N_HC) * N_HC`.
    pub hc_attn_base: &'a [f32],

    // --- FFN sublayer ---
    /// Per-channel RMSNorm gain applied before the SwiGLU/MoE stack. Length `N_EMBD`.
    pub ffn_norm: &'a [f32],
    /// Q8_0 shared-expert gate weight. `[N_FF_EXP, N_EMBD]`.
    pub ffn_gate: &'a [u8],
    /// Q8_0 shared-expert up weight. `[N_FF_EXP, N_EMBD]`.
    pub ffn_up: &'a [u8],
    /// Q8_0 shared-expert down weight. `[N_EMBD, N_FF_EXP]`.
    pub ffn_down: &'a [u8],

    // --- routed MoE ---
    /// f16 router head. `[N_EMBD, N_EXPERT]`, packed as `u16`.
    pub ffn_gate_inp: &'a [u16],
    /// Optional selection-only bias for top-k routing. Length `N_EXPERT` when present.
    pub ffn_exp_probs_b: Option<&'a [f32]>,
    /// Per-expert Q8_0 gate weights, one slice per expert.
    pub ffn_gate_exps: &'a [&'a [u8]],
    /// Per-expert Q8_0 up weights, one slice per expert.
    pub ffn_up_exps: &'a [&'a [u8]],
    /// Per-expert Q8_0 down weights, one slice per expert.
    pub ffn_down_exps: &'a [&'a [u8]],

    // --- FFN-side HC control ---
    /// f16 HC control projection for the FFN pre step.
    /// Shape `[(2 + N_HC) * N_HC, N_HC * N_EMBD]`.
    pub hc_ffn_fn: &'a [u16],
    /// `[pre, post, comb]` scales for the FFN Sinkhorn split. Length `3`.
    pub hc_ffn_scale: &'a [f32],
    /// Sinkhorn split base for the FFN side, length `(2 + N_HC) * N_HC`.
    pub hc_ffn_base: &'a [f32],
}

/// Reusable per-token scratch buffers. The C `ds4_cpu_decode_scratch`
/// aggregates ~30 named buffers (`attn_cur`, `attn_norm`, `q`, `kv`, `heads`,
/// `attn_out`, `attn_residual`, `after_attn_hc`, `flat`, `mid_hc`, ...); we
/// group the ones needed by [`layer_forward_self_one`] here. The
/// [`ExpertScratch`] is owned by this struct so the MoE inner loop avoids
/// per-call allocations.
pub struct LayerScratch {
    /// `N_EMBD` — attention sublayer input after HC weighted sum.
    pub attn_in: Vec<f32>,
    /// `N_EMBD` — RMSNorm of `attn_in` (input to Q/KV projection).
    pub attn_norm: Vec<f32>,
    /// `N_HEAD * N_HEAD_DIM` — Q projection output (LoRA + head RMSNorm + RoPE).
    pub q_proj: Vec<f32>,
    /// `N_HEAD_KV * N_HEAD_DIM` — KV projection output (RMSNorm + RoPE).
    pub kv_proj: Vec<f32>,
    /// `N_HEAD * N_HEAD_DIM` — attention rows (per-head softmax over KV).
    pub heads: Vec<f32>,
    /// `N_EMBD` — grouped output projection of `heads`.
    pub attn_out: Vec<f32>,
    /// `N_HC * N_EMBD` — HC state after the attention post step.
    pub mid_hc: Vec<f32>,
    /// `N_EMBD` — FFN sublayer input after HC weighted sum.
    pub ffn_in: Vec<f32>,
    /// `N_EMBD` — RMSNorm of `ffn_in`.
    pub ffn_norm: Vec<f32>,
    /// `N_EMBD` — shared expert SwiGLU output.
    pub ffn_shared: Vec<f32>,
    /// `N_EMBD` — routed MoE output (will be added to `ffn_shared`).
    pub ffn_moe: Vec<f32>,
    /// `N_EMBD` — combined FFN output fed into the FFN-side `hc_post_one`.
    pub ffn_out: Vec<f32>,
    /// `N_HC * N_EMBD` — RMSNorm-no-weight of the HC residual (input to the
    /// HC control f16 matvec). Used by both attention-side and FFN-side
    /// `hc_pre_from_state_one` recipes.
    pub hc_flat: Vec<f32>,
    /// `(2 + N_HC) * N_HC` — raw control mix produced by the HC f16 matvec.
    pub hc_mix: Vec<f32>,
    /// `(2 + N_HC) * N_HC` — split output: `[pre | post | comb]`.
    pub hc_split: Vec<f32>,
    /// `ExpertScratch` for [`layer_routed_moe_one`] (`N_FF_EXP` + `N_EMBD`).
    pub expert: ExpertScratch,
}

impl LayerScratch {
    /// Allocate every reusable buffer at the model-default DS4 dimensions.
    pub fn new() -> Self {
        let n_embd = N_EMBD as usize;
        let n_hc = N_HC as usize;
        let q_dim = (N_HEAD as usize) * (N_HEAD_DIM as usize);
        let kv_dim = (N_HEAD_KV as usize) * (N_HEAD_DIM as usize);
        let split_len = (2 + n_hc) * n_hc;
        Self {
            attn_in: vec![0.0; n_embd],
            attn_norm: vec![0.0; n_embd],
            q_proj: vec![0.0; q_dim],
            kv_proj: vec![0.0; kv_dim],
            heads: vec![0.0; q_dim],
            attn_out: vec![0.0; n_embd],
            mid_hc: vec![0.0; n_hc * n_embd],
            ffn_in: vec![0.0; n_embd],
            ffn_norm: vec![0.0; n_embd],
            ffn_shared: vec![0.0; n_embd],
            ffn_moe: vec![0.0; n_embd],
            ffn_out: vec![0.0; n_embd],
            hc_flat: vec![0.0; n_hc * n_embd],
            hc_mix: vec![0.0; split_len],
            hc_split: vec![0.0; split_len],
            expert: ExpertScratch::new(),
        }
    }
}

impl Default for LayerScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Inline port of `hc_pre_from_state_one` (`ds4.c` ~line 4317): RMSNorm the
/// HC residual, project through the f16 control matrix, run a Sinkhorn split,
/// then collapse the residual using the resulting `pre` weights.
///
/// Outputs:
///
/// * `out`  — `N_EMBD` weighted sum of the HC streams under the `pre` weights.
/// * `post` — `N_HC` post gates (sliced from the Sinkhorn split).
/// * `comb` — `N_HC * N_HC` combine matrix (sliced from the Sinkhorn split).
///
/// Scratch buffers (`hc_flat`, `hc_mix`, `hc_split`) live on [`LayerScratch`].
fn hc_pre_from_state_one(
    out: &mut [f32],
    residual_hc: &[f32],
    hc_fn: &[u16],
    hc_scale: &[f32],
    hc_base: &[f32],
    post: &mut [f32],
    comb: &mut [f32],
    hc_flat: &mut [f32],
    hc_mix: &mut [f32],
    hc_split: &mut [f32],
) {
    let n_hc = N_HC as usize;
    let split_len = (2 + n_hc) * n_hc;
    debug_assert_eq!(out.len(), N_EMBD as usize);
    debug_assert_eq!(residual_hc.len(), n_hc * N_EMBD as usize);
    debug_assert_eq!(post.len(), n_hc);
    debug_assert_eq!(comb.len(), n_hc * n_hc);
    debug_assert_eq!(hc_flat.len(), n_hc * N_EMBD as usize);
    debug_assert_eq!(hc_mix.len(), split_len);
    debug_assert_eq!(hc_split.len(), split_len);
    debug_assert_eq!(hc_scale.len(), 3);
    debug_assert_eq!(hc_base.len(), split_len);

    // 1. RMSNorm without learned weight over the whole flattened HC state.
    rms_norm_no_weight(hc_flat, residual_hc, RMS_EPS);

    // 2. Project to the control mix via f16 matvec.
    matvec_f16(hc_mix, hc_fn, hc_flat);

    // 3. Sinkhorn split into [pre | post | comb].
    hc_split_sinkhorn_one(
        hc_split,
        hc_mix,
        hc_scale,
        hc_base,
        n_hc,
        N_HC_SINKHORN_ITER as usize,
        HC_EPS,
    );

    // 4. Weighted sum of the residual HC streams using the `pre` weights.
    let pre = &hc_split[..n_hc];
    hc_weighted_sum_one(out, residual_hc, pre);

    // 5. Surface the post gates and combine matrix to the caller.
    post.copy_from_slice(&hc_split[n_hc..2 * n_hc]);
    comb.copy_from_slice(&hc_split[2 * n_hc..]);
}

/// Run one transformer block of the cold-start CPU forward pass.
///
/// Ports `layer_forward_self_one` from `ds4.c` (~line 7836). See the module
/// docstring for the step-by-step correspondence with the C source.
///
/// * `next`   — output HC state, `[N_HC, N_EMBD]` row major.
/// * `cur`    — input HC state, `[N_HC, N_EMBD]` row major.
/// * `weights` — borrowed view of all the per-layer tensors this block reads.
/// * `il`     — layer index (used by RoPE to pick the per-layer freq base/scale).
/// * `pos`    — token position in the sequence (used by RoPE).
/// * `token`  — token id (matched to the C signature; currently unused in the
///   pure-CPU path because routed MoE picks experts from activations only).
/// * `scratch` — reusable scratch; allocate once and reuse across the layer
///   loop and across tokens.
pub fn layer_forward_self_one(
    next: &mut [f32],
    cur: &[f32],
    weights: &LayerWeightsRef<'_>,
    il: u32,
    pos: u64,
    token: i32,
    scratch: &mut LayerScratch,
) {
    let _ = token; // matched to the C signature; not consumed by the cold-start path.

    let n_hc = N_HC as usize;
    let n_embd = N_EMBD as usize;
    debug_assert_eq!(cur.len(), n_hc * n_embd);
    debug_assert_eq!(next.len(), n_hc * n_embd);

    // Per-step scratch for the Sinkhorn post/comb outputs. The C source uses
    // `float post[4]` and `float comb[16]` on the stack; we mirror that.
    let mut post = [0.0f32; N_HC as usize];
    let mut comb = [0.0f32; (N_HC * N_HC) as usize];

    // --- Step 1: attention-side hc_pre_from_state_one --------------------
    // The C source `memcpy`s `inp_hc` into `attn_residual`. Here we pass
    // `cur` (immutable) directly to `hc_pre_from_state_one` and remember to
    // feed `cur` (not a moved/aliased copy) to the matching `hc_post_one`
    // below.
    hc_pre_from_state_one(
        &mut scratch.attn_in,
        cur,
        weights.hc_attn_fn,
        weights.hc_attn_scale,
        weights.hc_attn_base,
        &mut post,
        &mut comb,
        &mut scratch.hc_flat,
        &mut scratch.hc_mix,
        &mut scratch.hc_split,
    );

    // --- Step 2: pre-attention RMSNorm -----------------------------------
    layer_attn_norm_one(&mut scratch.attn_norm, &scratch.attn_in, weights.attn_norm);

    // --- Step 3: Q / KV projections (LoRA + per-head norm) --------------
    layer_q_projection_normed_one(
        &mut scratch.q_proj,
        &scratch.attn_norm,
        weights.attn_q_a,
        weights.attn_q_b,
        weights.attn_q_a_norm,
    );
    layer_kv_projection_normed_one(
        &mut scratch.kv_proj,
        &scratch.attn_norm,
        weights.attn_kv,
        weights.attn_kv_a_norm,
    );

    // --- Step 4: RoPE on the tail dims of Q and KV ----------------------
    // The C version also FP8-quantizes and f16-rounds `kv` here; the CPU
    // reference path keeps full f32 precision (the existing per-block
    // primitives expect f32 in/out).
    let freq_base = layer_rope_freq_base(il);
    let freq_scale = layer_rope_freq_scale(il);
    tail_rotate_inplace(
        &mut scratch.q_proj,
        N_HEAD,
        N_HEAD_DIM,
        N_ROT,
        pos,
        freq_base,
        freq_scale,
    );
    tail_rotate_inplace(
        &mut scratch.kv_proj,
        N_HEAD_KV,
        N_HEAD_DIM,
        N_ROT,
        pos,
        freq_base,
        freq_scale,
    );

    // --- Step 5: sink-aware attention with a single KV row -------------
    layer_attention_one(
        &mut scratch.heads,
        &scratch.q_proj,
        &scratch.kv_proj,
        weights.attn_sinks,
    );

    // Step 7 in the C source applies inverse RoPE to `heads` here. The
    // available Rust `rope::tail_rotate_inplace` does not expose the
    // `inverse` flag, so we skip it. For the cold-start `pos = 0` case
    // RoPE is the identity transform and the omission is exact; for
    // `pos > 0` callers this is a documented approximation pending a
    // future port of `rope_tail_layer_inplace` with `inverse=true`.

    // --- Step 6: grouped output projection -----------------------------
    layer_grouped_out_one(
        &mut scratch.attn_out,
        &scratch.heads,
        weights.attn_output_a_grouped,
        weights.attn_output_b,
    );

    // --- Step 7: HC post: inject attn_out, remix HC residuals -----------
    hc_post_one(
        &mut scratch.mid_hc,
        &scratch.attn_out,
        cur,
        &post,
        &comb,
    );

    // --- Step 8: FFN-side hc_pre_from_state_one ------------------------
    let mut ffn_post = [0.0f32; N_HC as usize];
    let mut ffn_comb = [0.0f32; (N_HC * N_HC) as usize];
    hc_pre_from_state_one(
        &mut scratch.ffn_in,
        &scratch.mid_hc,
        weights.hc_ffn_fn,
        weights.hc_ffn_scale,
        weights.hc_ffn_base,
        &mut ffn_post,
        &mut ffn_comb,
        &mut scratch.hc_flat,
        &mut scratch.hc_mix,
        &mut scratch.hc_split,
    );

    // --- Step 9: RMSNorm + shared FFN + routed MoE ----------------------
    rms_norm_weight(
        &mut scratch.ffn_norm,
        &scratch.ffn_in,
        weights.ffn_norm,
        RMS_EPS,
    );

    let shared_weights = SharedFfnWeights {
        gate: weights.ffn_gate,
        up: weights.ffn_up,
        down: weights.ffn_down,
    };
    shared_ffn_one(&mut scratch.ffn_shared, &shared_weights, &scratch.ffn_norm);

    let moe_weights = RoutedMoeWeights {
        ffn_gate_inp: weights.ffn_gate_inp,
        ffn_exp_probs_b: weights.ffn_exp_probs_b,
        ffn_gate_exps: weights.ffn_gate_exps,
        ffn_up_exps: weights.ffn_up_exps,
        ffn_down_exps: weights.ffn_down_exps,
    };
    layer_routed_moe_one(
        &mut scratch.ffn_moe,
        &moe_weights,
        &scratch.ffn_norm,
        SWIGLU_CLAMP_EXP,
        &mut scratch.expert,
    );

    for i in 0..n_embd {
        scratch.ffn_out[i] = scratch.ffn_shared[i] + scratch.ffn_moe[i];
    }

    // --- Step 10: FFN HC post -> final output --------------------------
    hc_post_one(
        next,
        &scratch.ffn_out,
        &scratch.mid_hc,
        &ffn_post,
        &ffn_comb,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;
    use crate::matvec::Q8_0_BLOCK_BYTES;
    use crate::shape::{N_EXPERT, N_FF_EXP, N_LORA_O, N_LORA_Q, N_OUT_GROUP};

    /// Q8_0 row stride in bytes for a row of `in_dim` weights.
    #[inline]
    fn q8_0_row_bytes(in_dim: usize) -> usize {
        let blocks = (in_dim + 31) / 32;
        blocks * Q8_0_BLOCK_BYTES
    }

    /// Build a Q8_0 weight block of `out_dim` rows by `in_dim` columns whose
    /// every block has scale 0 and quants 0 — i.e. the zero matrix.
    fn zero_q8_0(out_dim: usize, in_dim: usize) -> Vec<u8> {
        vec![0u8; out_dim * q8_0_row_bytes(in_dim)]
    }

    /// Owned bundle of zero-weight buffers. The `Vec<&[u8]>` indirection lists
    /// for the per-expert tables are built separately in [`expert_refs`]
    /// because the borrow checker can't store a `Vec<&[u8]>` referencing this
    /// struct's own fields inside the same struct without self-references.
    struct ZeroWeights {
        attn_norm: Vec<f32>,
        attn_q_a: Vec<u8>,
        attn_q_a_norm: Vec<f32>,
        attn_q_b: Vec<u8>,
        attn_kv: Vec<u8>,
        attn_kv_a_norm: Vec<f32>,
        attn_output_a_grouped: Vec<u8>,
        attn_output_b: Vec<u8>,
        attn_sinks: Vec<f32>,
        hc_attn_fn: Vec<u16>,
        hc_attn_scale: Vec<f32>,
        hc_attn_base: Vec<f32>,
        ffn_norm: Vec<f32>,
        ffn_gate: Vec<u8>,
        ffn_up: Vec<u8>,
        ffn_down: Vec<u8>,
        ffn_gate_inp: Vec<u16>,
        ffn_gate_exps_bufs: Vec<Vec<u8>>,
        ffn_up_exps_bufs: Vec<Vec<u8>>,
        ffn_down_exps_bufs: Vec<Vec<u8>>,
        hc_ffn_fn: Vec<u16>,
        hc_ffn_scale: Vec<f32>,
        hc_ffn_base: Vec<f32>,
    }

    /// Per-expert reference lists. Kept in a separate struct so the
    /// caller can hold `&[&[u8]]` slices that outlive a single function
    /// call.
    struct ExpertRefs<'a> {
        gate: Vec<&'a [u8]>,
        up: Vec<&'a [u8]>,
        down: Vec<&'a [u8]>,
    }

    impl ZeroWeights {
        fn new() -> Self {
            let n_embd = N_EMBD as usize;
            let n_hc = N_HC as usize;
            let split_len = (2 + n_hc) * n_hc;
            let lora_q = N_LORA_Q as usize;
            let n_head = N_HEAD as usize;
            let head_dim = N_HEAD_DIM as usize;
            let n_ff = N_FF_EXP as usize;
            let n_expert = N_EXPERT as usize;
            let n_out_group = N_OUT_GROUP as usize;
            let lora_o = N_LORA_O as usize;
            let group_dim = head_dim * n_head / n_out_group;

            // With a zero `ffn_gate_inp`, all expert probabilities tie at the
            // same value; `topk_desc` breaks ties in input order, so only the
            // first `N_EXPERT_USED` experts (indices 0..6) are ever read in
            // this test. We allocate full-size weights for those and empty
            // dummies for the rest — saves ~6.8 GB of zero buffers at the
            // full DS4 dimensions.
            use crate::shape::N_EXPERT_USED;
            let n_used = N_EXPERT_USED as usize;
            let ffn_gate_exps_bufs: Vec<Vec<u8>> = (0..n_expert)
                .map(|i| if i < n_used { zero_q8_0(n_ff, n_embd) } else { Vec::new() })
                .collect();
            let ffn_up_exps_bufs: Vec<Vec<u8>> = (0..n_expert)
                .map(|i| if i < n_used { zero_q8_0(n_ff, n_embd) } else { Vec::new() })
                .collect();
            let ffn_down_exps_bufs: Vec<Vec<u8>> = (0..n_expert)
                .map(|i| if i < n_used { zero_q8_0(n_embd, n_ff) } else { Vec::new() })
                .collect();

            // Make sure the helper import is exercised in case we ever flip
            // a weight from zero in this builder.
            let _ = f32_to_f16(0.0);

            Self {
                attn_norm: vec![1.0; n_embd],
                attn_q_a: zero_q8_0(lora_q, n_embd),
                attn_q_a_norm: vec![1.0; lora_q],
                attn_q_b: zero_q8_0(n_head * head_dim, lora_q),
                attn_kv: zero_q8_0(head_dim, n_embd),
                attn_kv_a_norm: vec![1.0; head_dim],
                attn_output_a_grouped: zero_q8_0(n_out_group * lora_o, group_dim),
                attn_output_b: zero_q8_0(n_embd, n_out_group * lora_o),
                // Sinks very negative => sink contributes nothing, but with
                // n_kv == 1 and zero scores, softmax = 1 on the single KV row
                // (which is all zeros), so the head output is exactly zero.
                attn_sinks: vec![-1.0e6; n_head],
                hc_attn_fn: vec![0u16; split_len * (n_hc * n_embd)],
                // All scales zero so the Sinkhorn pre/post/comb only sees the
                // base term — and that is also zero, so pre = 0.5 + eps,
                // post = 1.0, comb = uniform 1/n_hc.
                hc_attn_scale: vec![0.0; 3],
                hc_attn_base: vec![0.0; split_len],
                ffn_norm: vec![1.0; n_embd],
                ffn_gate: zero_q8_0(n_ff, n_embd),
                ffn_up: zero_q8_0(n_ff, n_embd),
                ffn_down: zero_q8_0(n_embd, n_ff),
                ffn_gate_inp: vec![0u16; n_expert * n_embd],
                ffn_gate_exps_bufs,
                ffn_up_exps_bufs,
                ffn_down_exps_bufs,
                hc_ffn_fn: vec![0u16; split_len * (n_hc * n_embd)],
                hc_ffn_scale: vec![0.0; 3],
                hc_ffn_base: vec![0.0; split_len],
            }
        }

        fn expert_refs(&self) -> ExpertRefs<'_> {
            ExpertRefs {
                gate: self.ffn_gate_exps_bufs.iter().map(|v| v.as_slice()).collect(),
                up: self.ffn_up_exps_bufs.iter().map(|v| v.as_slice()).collect(),
                down: self.ffn_down_exps_bufs.iter().map(|v| v.as_slice()).collect(),
            }
        }

        fn build_ref<'a>(&'a self, exps: &'a ExpertRefs<'a>) -> LayerWeightsRef<'a> {
            LayerWeightsRef {
                attn_norm: &self.attn_norm,
                attn_q_a: &self.attn_q_a,
                attn_q_a_norm: &self.attn_q_a_norm,
                attn_q_b: &self.attn_q_b,
                attn_kv: &self.attn_kv,
                attn_kv_a_norm: &self.attn_kv_a_norm,
                attn_output_a_grouped: &self.attn_output_a_grouped,
                attn_output_b: &self.attn_output_b,
                attn_sinks: &self.attn_sinks,
                hc_attn_fn: &self.hc_attn_fn,
                hc_attn_scale: &self.hc_attn_scale,
                hc_attn_base: &self.hc_attn_base,
                ffn_norm: &self.ffn_norm,
                ffn_gate: &self.ffn_gate,
                ffn_up: &self.ffn_up,
                ffn_down: &self.ffn_down,
                ffn_gate_inp: &self.ffn_gate_inp,
                ffn_exp_probs_b: None,
                ffn_gate_exps: &exps.gate,
                ffn_up_exps: &exps.up,
                ffn_down_exps: &exps.down,
                hc_ffn_fn: &self.hc_ffn_fn,
                hc_ffn_scale: &self.hc_ffn_scale,
                hc_ffn_base: &self.hc_ffn_base,
            }
        }
    }

    /// (a) With every weight set to zero, both attention and FFN sublayers
    ///     contribute zero (their outputs are exactly zero), so the HC
    ///     `post` step degenerates to a pure mix of the HC residual through
    ///     the Sinkhorn-uniform `comb` matrix. When the input HC state has
    ///     all four streams set to the same vector `v`, that uniform mix
    ///     reproduces `v` on every output stream — i.e. the residual-only
    ///     path acts as the identity on a stream-uniform input.
    #[test]
    fn zero_weights_residual_only_preserves_stream_uniform_input() {
        let n_hc = N_HC as usize;
        let n_embd = N_EMBD as usize;

        let weights_owned = ZeroWeights::new();
        let exps = weights_owned.expert_refs();
        let weights = weights_owned.build_ref(&exps);

        // Stream-uniform input: same `v` repeated across the four HC streams.
        let v: Vec<f32> = (0..n_embd).map(|i| ((i % 13) as f32) * 0.01 - 0.05).collect();
        let mut cur = vec![0.0f32; n_hc * n_embd];
        for h in 0..n_hc {
            cur[h * n_embd..(h + 1) * n_embd].copy_from_slice(&v);
        }

        let mut next = vec![0.0f32; n_hc * n_embd];
        let mut scratch = LayerScratch::new();
        layer_forward_self_one(&mut next, &cur, &weights, /*il=*/ 0, /*pos=*/ 0, /*token=*/ 0, &mut scratch);

        // Each output stream should be (approximately) equal to `v`.
        // Tolerance is loose because the Sinkhorn split is not exactly
        // uniform (the eps floor + 20 iterations leave tiny drift), and
        // RMSNorm + matvec on the zero `hc_attn_fn` also adds a small
        // numerical bias.
        for h in 0..n_hc {
            for i in 0..n_embd {
                let got = next[h * n_embd + i];
                let want = v[i];
                assert!(
                    (got - want).abs() < 1e-3,
                    "stream {h} dim {i}: got {got} want {want}",
                );
            }
        }
    }

    /// (b) Shape consistency: at the real DS4 dimensions, the layer-forward
    ///     produces an `[N_HC, N_EMBD]` output with no NaNs / infinities,
    ///     even with arbitrary (but small) input activations. Uses the
    ///     same all-zero weights as (a) so the test stays cheap.
    #[test]
    fn shape_consistency_at_full_ds4_dims() {
        let n_hc = N_HC as usize;
        let n_embd = N_EMBD as usize;

        let weights_owned = ZeroWeights::new();
        let exps = weights_owned.expert_refs();
        let weights = weights_owned.build_ref(&exps);

        let cur: Vec<f32> = (0..n_hc * n_embd)
            .map(|i| ((i % 37) as f32) * 0.002 - 0.03)
            .collect();
        let mut next = vec![0.0f32; n_hc * n_embd];
        let mut scratch = LayerScratch::new();
        layer_forward_self_one(&mut next, &cur, &weights, /*il=*/ 7, /*pos=*/ 0, /*token=*/ 42, &mut scratch);

        assert_eq!(next.len(), n_hc * n_embd);
        for (i, &v) in next.iter().enumerate() {
            assert!(v.is_finite(), "next[{i}] = {v} not finite");
        }

        // Also make sure the various scratch buffers still have their
        // documented shapes (they're reused across calls, so the lengths
        // shouldn't drift).
        assert_eq!(scratch.attn_in.len(), n_embd);
        assert_eq!(scratch.attn_norm.len(), n_embd);
        assert_eq!(scratch.q_proj.len(), (N_HEAD as usize) * (N_HEAD_DIM as usize));
        assert_eq!(scratch.kv_proj.len(), (N_HEAD_KV as usize) * (N_HEAD_DIM as usize));
        assert_eq!(scratch.heads.len(), (N_HEAD as usize) * (N_HEAD_DIM as usize));
        assert_eq!(scratch.attn_out.len(), n_embd);
        assert_eq!(scratch.mid_hc.len(), n_hc * n_embd);
        assert_eq!(scratch.ffn_in.len(), n_embd);
        assert_eq!(scratch.ffn_norm.len(), n_embd);
        assert_eq!(scratch.ffn_shared.len(), n_embd);
        assert_eq!(scratch.ffn_moe.len(), n_embd);
        assert_eq!(scratch.ffn_out.len(), n_embd);
        assert_eq!(scratch.hc_flat.len(), n_hc * n_embd);
        let split_len = (2 + n_hc) * n_hc;
        assert_eq!(scratch.hc_mix.len(), split_len);
        assert_eq!(scratch.hc_split.len(), split_len);
    }
}
