//! Multi-head Latent Attention (MLA) — CPU reference helpers.
//!
//! Ports the per-token attention slice of `ds4.c` lines ~4575..4944:
//!
//! * `layer_attn_norm_one`           (~line 4575)
//! * `layer_q_projection_normed_one` (~line 4596) — Q8_0 LoRA Q projection
//! * `layer_kv_projection_normed_one`(~line 4633) — Q8_0 KV projection
//! * `layer_attention_rows_one`      (~line 4897) — sink-aware MLA softmax
//! * `layer_grouped_out_one`         (~line 4948) — grouped output projection
//!
//! These mirror the C control flow exactly: f32 storage, f64 accumulators
//! where the C source uses them, identical pre/post norm placement, and the
//! 1/sqrt(head_dim) scaling that ships with DeepSeek V4 Flash. The helpers
//! are deliberately allocation-light (no scratch struct) so they slot into
//! the existing `forward.rs` scaffold without binding decisions baked in.
//!
//! Notes on quantization:
//!
//! * Q8_0 weight rows arrive as raw byte slices laid out as
//!   `[[f16 d_b][i8 q_b * 32]] * blocks`. The row count is implied by the
//!   shape constants (`N_LORA_Q`, `N_HEAD * N_HEAD_DIM`, etc.).
//! * The C source pulls weights via `tensor_data(model, layer->attn_q_a)`
//!   and friends; in Rust we accept the already-resolved byte slice so the
//!   tensor metadata layer can live in another module.

use crate::matvec::{matvec_q8_0, Q8_0_BLOCK_BYTES};
use crate::nn::{head_rms_norm_inplace, rms_norm_weight};
use crate::shape::{N_HEAD, N_HEAD_DIM, N_LORA_O, N_LORA_Q, RMS_EPS};

/// Pre-attention RMSNorm with a learned per-channel scale. Mirrors
/// `layer_attn_norm_one` (ds4.c ~line 4575). The C wrapper exists just to
/// fetch the `attn_norm` tensor and forward to `rms_norm_weight`; we pass
/// the weight slice directly because tensor lookup happens upstream here.
pub fn layer_attn_norm_one(out: &mut [f32], x: &[f32], attn_norm_w: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), attn_norm_w.len());
    rms_norm_weight(out, x, attn_norm_w, RMS_EPS);
}

/// Low-rank Q projection plus per-head RMSNorm. Mirrors
/// `layer_q_projection_normed_one` (ds4.c ~line 4596):
///
/// 1. `qr      = attn_q_a @ norm`           (Q8_0, output dim `N_LORA_Q`)
/// 2. `qr_norm = rms_norm_weight(qr, q_a_norm, N_LORA_Q)`
/// 3. `q       = attn_q_b @ qr_norm`        (Q8_0, output dim `N_HEAD * N_HEAD_DIM`)
/// 4. `head_rms_norm_inplace(q, N_HEAD, N_HEAD_DIM)`
///
/// `attn_q_a` and `attn_q_b` are Q8_0-packed byte rows; their lengths are
/// implied by `N_EMBD → N_LORA_Q` and `N_LORA_Q → N_HEAD * N_HEAD_DIM`.
pub fn layer_q_projection_normed_one(
    q: &mut [f32],
    norm: &[f32],
    attn_q_a: &[u8],
    attn_q_b: &[u8],
    q_a_norm_w: &[f32],
) {
    let lora = N_LORA_Q as usize;
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    debug_assert_eq!(q.len(), n_head * head_dim);
    debug_assert_eq!(q_a_norm_w.len(), lora);
    debug_assert_eq!(attn_q_a.len(), lora * q8_0_row_bytes(norm.len()));
    debug_assert_eq!(attn_q_b.len(), n_head * head_dim * q8_0_row_bytes(lora));

    let mut qr = vec![0.0_f32; lora];
    let mut qr_norm = vec![0.0_f32; lora];

    matvec_q8_0(&mut qr, attn_q_a, norm);
    rms_norm_weight(&mut qr_norm, &qr, q_a_norm_w, RMS_EPS);
    matvec_q8_0(q, attn_q_b, &qr_norm);
    head_rms_norm_inplace(q, N_HEAD, N_HEAD_DIM, RMS_EPS);
}

/// KV projection: one Q8_0 matvec of width `N_HEAD_DIM` followed by a
/// learned RMSNorm. Mirrors `layer_kv_projection_normed_one`
/// (ds4.c ~line 4633). DS4 ships with a single shared KV head
/// (`N_HEAD_KV == 1`), so the result is one `N_HEAD_DIM`-length vector.
pub fn layer_kv_projection_normed_one(
    kv: &mut [f32],
    normed: &[f32],
    attn_kv: &[u8],
    kv_a_norm_w: &[f32],
) {
    let head_dim = N_HEAD_DIM as usize;
    debug_assert_eq!(kv.len(), head_dim);
    debug_assert_eq!(kv_a_norm_w.len(), head_dim);
    debug_assert_eq!(attn_kv.len(), head_dim * q8_0_row_bytes(normed.len()));

    let mut raw = vec![0.0_f32; head_dim];
    matvec_q8_0(&mut raw, attn_kv, normed);
    rms_norm_weight(kv, &raw, kv_a_norm_w, RMS_EPS);
}

/// Sink-aware attention over a window of KV rows. Mirrors
/// `layer_attention_rows_one` (ds4.c ~line 4897).
///
/// MLA shares one latent KV vector across all heads, so `kv_rows` is
/// laid out as `[n_kv, N_HEAD_DIM]` (not per-head). For each head:
///
/// * compute the raw scores `q . kv * 1/sqrt(N_HEAD_DIM)`
/// * include the learned per-head `sinks[h]` logit in the softmax
///   denominator (but **not** the numerator — it has no value vector)
/// * weighted-sum the KV rows to produce that head's output
///
/// `out_heads` is `[N_HEAD, N_HEAD_DIM]`, `q` is `[N_HEAD, N_HEAD_DIM]`,
/// `sinks` is `[N_HEAD]`. The numerical recipe matches the C version
/// bit-for-bit modulo platform float ordering: scores accumulate in f64
/// in the dot product (the C `dot_f32` is a scalar accumulator but the
/// max + softmax reductions promote, so we follow suit).
pub fn layer_attention_rows_one(
    out_heads: &mut [f32],
    q: &[f32],
    kv_rows: &[f32],
    sinks: &[f32],
    n_kv: usize,
) {
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    debug_assert_eq!(out_heads.len(), n_head * head_dim);
    debug_assert_eq!(q.len(), n_head * head_dim);
    debug_assert_eq!(sinks.len(), n_head);
    debug_assert_eq!(kv_rows.len(), n_kv * head_dim);

    let kq_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut score = vec![0.0_f32; n_kv];

    for h in 0..n_head {
        let qh = &q[h * head_dim..(h + 1) * head_dim];

        // First pass: scores + max for numerical stability.
        let mut max_score = sinks[h];
        for r in 0..n_kv {
            let kv = &kv_rows[r * head_dim..(r + 1) * head_dim];
            let mut acc: f64 = 0.0;
            for i in 0..head_dim {
                acc += qh[i] as f64 * kv[i] as f64;
            }
            let s = (acc as f32) * kq_scale;
            score[r] = s;
            if s > max_score {
                max_score = s;
            }
        }

        // Second pass: softmax + weighted sum of KV rows.
        let oh = &mut out_heads[h * head_dim..(h + 1) * head_dim];
        for v in oh.iter_mut() {
            *v = 0.0;
        }

        // Sink contributes to the denominator but not the numerator.
        let mut denom: f64 = (sinks[h] - max_score).exp() as f64;
        for r in 0..n_kv {
            let weight = (score[r] - max_score).exp();
            let kv = &kv_rows[r * head_dim..(r + 1) * head_dim];
            denom += weight as f64;
            for i in 0..head_dim {
                oh[i] += weight * kv[i];
            }
        }

        let inv = if denom > 0.0 { (1.0_f64 / denom) as f32 } else { 0.0 };
        for v in oh.iter_mut() {
            *v *= inv;
        }
    }
}

/// Convenience wrapper for the single-KV case used by the cold-start
/// (`forward_first_token_cpu`) path. Mirrors `layer_attention_one`
/// (ds4.c ~line 4937), which is just `layer_attention_rows_one` with
/// `n_kv == 1`.
pub fn layer_attention_one(out_heads: &mut [f32], q: &[f32], kv: &[f32], sinks: &[f32]) {
    layer_attention_rows_one(out_heads, q, kv, sinks, 1);
}

/// Grouped output projection. Mirrors `layer_grouped_out_one`
/// (ds4.c ~line 4948).
///
/// The C source splits the 64 heads into 8 groups of 8 heads; each
/// group has its own Q8_0 low-rank `attn_output_a` projection of
/// `group_dim → N_LORA_O`, and then a shared Q8_0 `attn_output_b`
/// fans the concatenated `[N_OUT_GROUP * N_LORA_O]` low vector back
/// out to `N_EMBD`.
///
/// `attn_output_a_grouped` is the flat concatenation of all 8 group
/// row buffers (the C path calls `matvec_q8_0_grouped_rows`, which
/// in scalar form is just 8 independent Q8_0 matvecs against disjoint
/// head slices). `attn_output_b` is one Q8_0 matrix of shape
/// `[N_EMBD, N_OUT_GROUP * N_LORA_O]`.
pub fn layer_grouped_out_one(
    out: &mut [f32],
    heads: &[f32],
    attn_output_a_grouped: &[u8],
    attn_output_b: &[u8],
) {
    const N_GROUPS: usize = 8;
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    let rank = N_LORA_O as usize;

    let group_heads = n_head / N_GROUPS;
    let group_dim = head_dim * group_heads;
    let group_row_bytes = q8_0_row_bytes(group_dim);
    let group_bytes = rank * group_row_bytes;

    debug_assert_eq!(heads.len(), n_head * head_dim);
    debug_assert_eq!(out.len(), crate::shape::N_EMBD as usize);
    debug_assert_eq!(attn_output_a_grouped.len(), N_GROUPS * group_bytes);
    debug_assert_eq!(
        attn_output_b.len(),
        out.len() * q8_0_row_bytes(N_GROUPS * rank)
    );

    let mut low = vec![0.0_f32; N_GROUPS * rank];
    for g in 0..N_GROUPS {
        let h0 = g * group_heads;
        let heads_slice = &heads[h0 * head_dim..(h0 + group_heads) * head_dim];
        let w = &attn_output_a_grouped[g * group_bytes..(g + 1) * group_bytes];
        let low_slice = &mut low[g * rank..(g + 1) * rank];
        matvec_q8_0(low_slice, w, heads_slice);
    }

    matvec_q8_0(out, attn_output_b, &low);
}

/// Q8_0 row stride in bytes for an input vector of length `in_dim`.
/// Pulled from `matvec::Q8_0_BLOCK_BYTES`; kept here as a one-liner so the
/// debug_asserts above read cleanly.
#[inline]
fn q8_0_row_bytes(in_dim: usize) -> usize {
    let blocks = (in_dim + 31) / 32;
    blocks * Q8_0_BLOCK_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{N_EMBD, N_HEAD, N_HEAD_DIM};

    /// `layer_attn_norm_one` with a unit-scale weight collapses to a
    /// plain (no-weight) RMSNorm. Mirrors the trivial identity-like
    /// check that the C wrapper does no extra arithmetic.
    #[test]
    fn attn_norm_unit_weight_matches_rms() {
        let n = N_EMBD as usize;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001) - 2.0).collect();
        let w = vec![1.0_f32; n];
        let mut out = vec![0.0_f32; n];
        layer_attn_norm_one(&mut out, &x, &w);
        // Check the RMS-normalized magnitude is close to 1 modulo eps.
        let ss: f64 = out.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = (ss / n as f64).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "rms = {rms}");
    }

    /// Sink-aware attention with `n_kv == 1` and a very negative sink
    /// degenerates to "copy the KV row to every head" — softmax weight
    /// on the single KV row → 1, sink contribution → 0. This pins the
    /// numerical recipe without depending on weight files.
    #[test]
    fn attention_single_kv_with_dead_sink_copies_kv() {
        let n_head = N_HEAD as usize;
        let head_dim = N_HEAD_DIM as usize;
        let q = vec![0.5_f32; n_head * head_dim];
        let kv: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.01).collect();
        let sinks = vec![-1e6_f32; n_head]; // effectively zero
        let mut out = vec![0.0_f32; n_head * head_dim];
        layer_attention_one(&mut out, &q, &kv, &sinks);
        for h in 0..n_head {
            for i in 0..head_dim {
                let got = out[h * head_dim + i];
                let want = kv[i];
                assert!(
                    (got - want).abs() < 1e-4,
                    "head {h} dim {i}: got {got}, want {want}"
                );
            }
        }
    }

    /// Sink-aware attention output should never produce NaNs / Infs for
    /// finite inputs, even when the sink and scores have wildly different
    /// magnitudes. Also asserts the per-head output shape.
    #[test]
    fn attention_rows_shape_and_finite() {
        let n_head = N_HEAD as usize;
        let head_dim = N_HEAD_DIM as usize;
        let n_kv = 7;
        let q: Vec<f32> = (0..n_head * head_dim).map(|i| ((i % 13) as f32) * 0.01).collect();
        let kv_rows: Vec<f32> = (0..n_kv * head_dim).map(|i| ((i % 7) as f32) * 0.02 - 0.05).collect();
        let sinks: Vec<f32> = (0..n_head).map(|h| (h as f32) * 0.1 - 1.0).collect();
        let mut out = vec![0.0_f32; n_head * head_dim];
        layer_attention_rows_one(&mut out, &q, &kv_rows, &sinks, n_kv);
        assert_eq!(out.len(), n_head * head_dim);
        for &v in out.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
