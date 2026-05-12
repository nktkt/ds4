//! CPU reference output head: HC collapse, final RMSNorm, and Q8_0 vocab
//! projection.
//!
//! Ports the language-model head from `ds4.c`:
//!
//! * `hc_weighted_sum_one`           (ds4.c ~L4267)
//! * `output_hc_head_one`            (ds4.c ~L7919)
//! * `output_logits_one`             (ds4.c ~L7947)
//! * `output_logits_one_decode_scratch` (ds4.c ~L7965) — same math as
//!   `output_logits_one` but allocation-free; in this Rust slice we fold the
//!   two into a single function because the scratch buffers are local Vecs.
//!
//! Layout conventions match `ds4.c`:
//!
//! * `inp_hc` is `[N_HC, N_EMBD]` row-major: head `h` occupies
//!   `inp_hc[h * N_EMBD .. (h + 1) * N_EMBD]`.
//! * `out_hc_fn` is an f16 weight matrix of shape `[N_HC, N_HC * N_EMBD]`,
//!   row-major, stored as packed `u16` (IEEE-754 binary16 bit patterns).
//! * `out_hc_scale` is a single f32 broadcast scalar (length-1 in the C tensor
//!   but we accept any non-empty slice and read `[0]`).
//! * `out_hc_base` is an f32 vector of length `N_HC`.
//! * `output_norm` is the f32 per-channel RMSNorm gain of length `N_EMBD`.
//! * `output_w` is the Q8_0 vocab projection weight matrix laid out exactly as
//!   `matvec_q8_0` expects: `N_VOCAB` rows of `ceil(N_EMBD/32) * 34` bytes each
//!   (`[f16 d][i8 q * 32]` per 32-element block).

use crate::matvec::{matvec_f16, matvec_q8_0};
use crate::nn::{rms_norm_no_weight, rms_norm_weight, sigmoid_stable};
use crate::shape::{HC_EPS, N_EMBD, N_HC, N_VOCAB, RMS_EPS};

/// Weighted sum across the HC heads.
///
/// `out[d] = sum_h x[h * n_embd + d] * weights[h]`
///
/// Mirrors `hc_weighted_sum_one` in `ds4.c` (~L4267). The number of heads is
/// inferred from `weights.len()` and `n_embd` from `out.len()`.
pub fn hc_weighted_sum_one(out: &mut [f32], x: &[f32], weights: &[f32]) {
    let n_embd = out.len();
    let n_hc = weights.len();
    assert_eq!(
        x.len(),
        n_hc * n_embd,
        "hc_weighted_sum_one: x.len() must equal weights.len() * out.len()",
    );
    for d in 0..n_embd {
        let mut acc = 0.0f32;
        for h in 0..n_hc {
            acc += x[h * n_embd + d] * weights[h];
        }
        out[d] = acc;
    }
}

/// HC collapse head: normalize the HC state, project through the f16 `fn`
/// matrix, build the per-head sigmoid weights, and weighted-sum the HC slots
/// into a single embedding-sized vector.
///
/// Mirrors `output_hc_head_one` in `ds4.c` (~L7919). Layouts:
///
/// * `out`           — `[N_EMBD]`
/// * `inp_hc`        — `[N_HC, N_EMBD]` row-major
/// * `out_hc_fn`     — f16 weight matrix `[N_HC, N_HC * N_EMBD]` row-major
/// * `out_hc_scale`  — single f32 broadcast scalar (read `[0]`)
/// * `out_hc_base`   — `[N_HC]`
pub fn output_hc_head_one(
    out: &mut [f32],
    inp_hc: &[f32],
    out_hc_fn: &[u16],
    out_hc_scale: &[f32],
    out_hc_base: &[f32],
) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    let hc_dim = n_embd * n_hc;

    assert_eq!(out.len(), n_embd, "output_hc_head_one: out must be N_EMBD");
    assert_eq!(inp_hc.len(), hc_dim, "output_hc_head_one: inp_hc must be N_HC * N_EMBD");
    assert_eq!(
        out_hc_fn.len(),
        n_hc * hc_dim,
        "output_hc_head_one: out_hc_fn must be [N_HC, N_HC * N_EMBD]",
    );
    assert!(!out_hc_scale.is_empty(), "output_hc_head_one: out_hc_scale must have at least one element");
    assert_eq!(out_hc_base.len(), n_hc, "output_hc_head_one: out_hc_base must be N_HC");

    let mut flat = vec![0.0f32; hc_dim];
    rms_norm_no_weight(&mut flat, inp_hc, RMS_EPS);

    let mut pre = vec![0.0f32; n_hc];
    matvec_f16(&mut pre, out_hc_fn, &flat);

    let scale0 = out_hc_scale[0];
    let mut w = vec![0.0f32; n_hc];
    for i in 0..n_hc {
        w[i] = sigmoid_stable(pre[i] * scale0 + out_hc_base[i]) + HC_EPS;
    }

    hc_weighted_sum_one(out, inp_hc, &w);
}

/// Full language-model head: HC collapse, final RMSNorm with learned gain,
/// then Q8_0 vocab projection.
///
/// Mirrors `output_logits_one` in `ds4.c` (~L7947).  The decode-scratch variant
/// (`output_logits_one_decode_scratch`, ~L7965) computes the same values; in
/// this Rust slice we collapse them into one allocating function because the
/// scratch buffers are stack/heap-local `Vec`s.
///
/// Layouts:
///
/// * `logits`        — `[N_VOCAB]`
/// * `inp_hc`        — `[N_HC, N_EMBD]` row-major
/// * `out_hc_fn`     — f16 weight matrix `[N_HC, N_HC * N_EMBD]` row-major
/// * `out_hc_scale`  — single f32 broadcast scalar
/// * `out_hc_base`   — `[N_HC]`
/// * `output_norm`   — `[N_EMBD]` per-channel RMSNorm gain
/// * `output_w`      — Q8_0 vocab projection, `N_VOCAB` rows by
///                      `ceil(N_EMBD/32) * 34` bytes
pub fn output_logits_one(
    logits: &mut [f32],
    inp_hc: &[f32],
    out_hc_fn: &[u16],
    out_hc_scale: &[f32],
    out_hc_base: &[f32],
    output_norm: &[f32],
    output_w: &[u8],
) {
    let n_embd = N_EMBD as usize;
    let n_vocab = N_VOCAB as usize;

    assert_eq!(logits.len(), n_vocab, "output_logits_one: logits must be N_VOCAB");
    assert_eq!(output_norm.len(), n_embd, "output_logits_one: output_norm must be N_EMBD");

    let mut embd = vec![0.0f32; n_embd];
    output_hc_head_one(&mut embd, inp_hc, out_hc_fn, out_hc_scale, out_hc_base);

    let mut norm = vec![0.0f32; n_embd];
    rms_norm_weight(&mut norm, &embd, output_norm, RMS_EPS);

    matvec_q8_0(logits, output_w, &norm);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;
    use crate::matvec::Q8_0_BLOCK_BYTES;

    /// With unit weights, `hc_weighted_sum_one` should just sum the heads
    /// element-wise.
    #[test]
    fn weighted_sum_unit_weights_is_sum() {
        let n_embd = 5usize;
        let n_hc = 3usize;
        // x[h, d] = h * 10 + d
        let mut x = vec![0.0f32; n_hc * n_embd];
        for h in 0..n_hc {
            for d in 0..n_embd {
                x[h * n_embd + d] = (h * 10 + d) as f32;
            }
        }
        let w = vec![1.0f32; n_hc];
        let mut out = vec![0.0f32; n_embd];
        hc_weighted_sum_one(&mut out, &x, &w);
        for d in 0..n_embd {
            let expected: f32 = (0..n_hc).map(|h| (h * 10 + d) as f32).sum();
            assert!((out[d] - expected).abs() < 1e-5, "d={d} got {} expected {}", out[d], expected);
        }
    }

    /// One-hot selector: weights = [0, 1, 0, ...] should pluck head 1 verbatim.
    #[test]
    fn weighted_sum_one_hot_selects_head() {
        let n_embd = 4usize;
        let n_hc = 4usize;
        let mut x = vec![0.0f32; n_hc * n_embd];
        for h in 0..n_hc {
            for d in 0..n_embd {
                x[h * n_embd + d] = (h as f32) + 0.25 * d as f32;
            }
        }
        let mut w = vec![0.0f32; n_hc];
        w[1] = 1.0;
        let mut out = vec![0.0f32; n_embd];
        hc_weighted_sum_one(&mut out, &x, &w);
        for d in 0..n_embd {
            let expected = 1.0 + 0.25 * d as f32;
            assert!((out[d] - expected).abs() < 1e-6);
        }
    }

    /// `output_logits_one` shape consistency: with sane-shaped inputs the head
    /// runs without panicking and produces `N_VOCAB` finite logits. We use the
    /// real shape constants (and therefore allocate the real-sized buffers) but
    /// fill the Q8_0 matrix with zero blocks (scale 1.0, qs 0) so every logit
    /// is exactly zero — this checks the layout-stride contract end-to-end
    /// without depending on real weights.
    #[test]
    fn output_logits_zero_weights_gives_zero_logits() {
        let n_embd = N_EMBD as usize;
        let n_hc = N_HC as usize;
        let n_vocab = N_VOCAB as usize;
        let hc_dim = n_embd * n_hc;

        // Modest non-zero HC state so RMSNorm has something to chew on.
        let inp_hc: Vec<f32> = (0..hc_dim).map(|i| ((i % 7) as f32) * 0.01 - 0.03).collect();

        // out_hc_fn all-zeros in f16 (== f32 0.0); pre will be all-zero, so the
        // sigmoid argument is just out_hc_base.
        let out_hc_fn = vec![0u16; n_hc * hc_dim];
        let out_hc_scale = vec![1.0f32];
        let out_hc_base = vec![0.0f32; n_hc];
        let output_norm = vec![1.0f32; n_embd];

        // Q8_0 zero weight matrix: per 32-element block, f16 scale = 1.0
        // followed by 32 zero i8 quants. Any scale works since the qs are 0.
        let blocks = (n_embd + 31) / 32;
        let row_bytes = blocks * Q8_0_BLOCK_BYTES;
        let mut output_w = vec![0u8; n_vocab * row_bytes];
        let scale_bytes = f32_to_f16(1.0).to_le_bytes();
        for r in 0..n_vocab {
            for b in 0..blocks {
                let base = r * row_bytes + b * Q8_0_BLOCK_BYTES;
                output_w[base] = scale_bytes[0];
                output_w[base + 1] = scale_bytes[1];
                // qs already zero from vec! initialization.
            }
        }

        let mut logits = vec![f32::NAN; n_vocab];
        output_logits_one(
            &mut logits,
            &inp_hc,
            &out_hc_fn,
            &out_hc_scale,
            &out_hc_base,
            &output_norm,
            &output_w,
        );

        // Every logit is exactly 0.0 (no NaNs, no infinities) since every Q8_0
        // weight is zero.
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] = {v} is not finite");
            assert_eq!(v, 0.0, "logit[{i}] = {v}, expected 0.0");
        }
    }
}
