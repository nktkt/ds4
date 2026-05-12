//! CPU reference shared-expert SwiGLU FFN.
//!
//! Ports the Q8_0 shared expert and the FFN sublayer driver from `ds4.c`:
//!
//! * `swiglu`                 (ds4.c ~line 5022)
//! * `layer_shared_ffn_one`   (ds4.c ~line 5029)
//! * `layer_ffn_one`          (ds4.c ~line 5576)
//! * `layer_ffn_one_decode_scratch` (ds4.c ~line 5681) — fold-in: this is the
//!   allocation-free twin of `layer_ffn_one`. The Rust port collapses the two
//!   into a single function because Rust's `Vec` already gives us cheap
//!   scratch storage and any caller that wants to reuse buffers can do so
//!   without us re-typing every signature.
//!
//! Scope deliberately covers only the shared expert (Q8_0 gate / up / down +
//! SwiGLU) and the RMSNorm + shared-FFN flow. The routed MoE sub-layer is
//! supplied via a caller-provided closure so this module stays free of the
//! router / expert-table plumbing.

use crate::matvec::{dot_q8_0_row, quantize_q8_0_activation, Q8_0_BLOCK_BYTES};
use crate::nn::{rms_norm_weight, swiglu};
use crate::shape::{N_EMBD, N_FF_EXP, RMS_EPS};

/// Q8_0 row stride in bytes for a row of `in_dim` weights.
#[inline]
fn q8_0_row_bytes(in_dim: usize) -> usize {
    let blocks = (in_dim + 31) / 32;
    blocks * Q8_0_BLOCK_BYTES
}

/// Q8_0 gate/up/down packed weights for the shared expert of one layer.
///
/// Each slice is laid out row-major `[out_dim, in_dim]` with the standard
/// `[f16 scale][i8 qs * 32]` block striding used throughout `ds4.c`.
pub struct SharedFfnWeights<'a> {
    /// `[N_FF_EXP, N_EMBD]` Q8_0 — gate projection.
    pub gate: &'a [u8],
    /// `[N_FF_EXP, N_EMBD]` Q8_0 — up projection.
    pub up: &'a [u8],
    /// `[N_EMBD, N_FF_EXP]` Q8_0 — down projection.
    pub down: &'a [u8],
}

/// Full FFN-sublayer weights for one transformer block.
///
/// Mirrors the subset of `ds4_layer_weights` that `layer_ffn_one` reads.
pub struct LayerFfnWeights<'a> {
    /// Per-channel RMSNorm scale, length `N_EMBD`.
    pub ffn_norm: &'a [f32],
    pub shared: SharedFfnWeights<'a>,
}

/// Run the Q8_0 shared-expert SwiGLU MLP for a single token.
///
/// Ports `layer_shared_ffn_one` (ds4.c ~line 5029):
///
/// ```text
/// xq, xscale = quantize_q8_0(x)
/// gate = W_gate @ x         (Q8_0 matvec)
/// up   = W_up   @ x         (Q8_0 matvec)
/// mid  = silu(gate) * up    (SwiGLU)
/// out  = W_down @ mid       (Q8_0 matvec)
/// ```
///
/// `out.len()` must equal `N_EMBD`; `x.len()` must equal `N_EMBD` (the shared
/// expert's input dim). The intermediate FFN width is `N_FF_EXP`.
pub fn shared_ffn_one(
    out: &mut [f32],
    weights: &SharedFfnWeights<'_>,
    x: &[f32],
) {
    let n_embd = N_EMBD as usize;
    let n_ff = N_FF_EXP as usize;
    assert_eq!(out.len(), n_embd, "shared_ffn_one: out dim must be N_EMBD");
    assert_eq!(x.len(), n_embd, "shared_ffn_one: in dim must be N_EMBD");

    let gate_up_row_bytes = q8_0_row_bytes(n_embd);
    let down_row_bytes = q8_0_row_bytes(n_ff);
    assert_eq!(
        weights.gate.len(),
        n_ff * gate_up_row_bytes,
        "shared_ffn_one: gate weight has wrong size"
    );
    assert_eq!(
        weights.up.len(),
        n_ff * gate_up_row_bytes,
        "shared_ffn_one: up weight has wrong size"
    );
    assert_eq!(
        weights.down.len(),
        n_embd * down_row_bytes,
        "shared_ffn_one: down weight has wrong size"
    );

    // Quantize the activation once and reuse for both gate and up matvecs —
    // matches the C `quantize_q8_0_activation` + `matvec_q8_0_pair_prequant`
    // flow.
    let blocks_in = (n_embd + 31) / 32;
    let mut xq = vec![0i8; blocks_in * 32];
    let mut xscale = vec![0.0f32; blocks_in];
    quantize_q8_0_activation(x, &mut xq, &mut xscale);

    // gate = W_gate @ x, up = W_up @ x
    let mut gate = vec![0.0f32; n_ff];
    let mut up = vec![0.0f32; n_ff];
    for r in 0..n_ff {
        let g_row = &weights.gate[r * gate_up_row_bytes..(r + 1) * gate_up_row_bytes];
        let u_row = &weights.up[r * gate_up_row_bytes..(r + 1) * gate_up_row_bytes];
        gate[r] = dot_q8_0_row(g_row, &xq, &xscale, n_embd);
        up[r] = dot_q8_0_row(u_row, &xq, &xscale, n_embd);
    }

    // mid = silu(gate) * up
    let mut mid = vec![0.0f32; n_ff];
    swiglu(&mut mid, &gate, &up);

    // out = W_down @ mid — quantize the intermediate, then dot per row.
    let blocks_mid = (n_ff + 31) / 32;
    let mut midq = vec![0i8; blocks_mid * 32];
    let mut midscale = vec![0.0f32; blocks_mid];
    quantize_q8_0_activation(&mid, &mut midq, &mut midscale);
    for r in 0..n_embd {
        let d_row = &weights.down[r * down_row_bytes..(r + 1) * down_row_bytes];
        out[r] = dot_q8_0_row(d_row, &midq, &midscale, n_ff);
    }
}

/// Apply RMSNorm + shared expert (+ optional routed MoE) for one token.
///
/// Ports the inner part of `layer_ffn_one` (ds4.c ~line 5576) — specifically
/// the
///
/// ```text
/// rms_norm_weight(norm, ffn_cur, ffn_norm, ...)
/// layer_routed_moe_one(moe, ..., norm, ...)        // optional
/// layer_shared_ffn_one(shared, ..., norm)
/// ffn_out = moe + shared
/// ```
///
/// slice. The hadamard-coded pre/post hooks (`hc_pre_from_state_one` and
/// `hc_post_one`) live one level up in the C code and are *not* part of the
/// FFN sublayer per se, so they are out of scope here.
///
/// `moe_hook` is an optional callback that receives the normalized input and
/// writes the routed-MoE contribution into `out_moe`. When `None`, the routed
/// path is skipped (out = shared expert only).
pub fn layer_ffn_one<F>(
    out: &mut [f32],
    weights: &LayerFfnWeights<'_>,
    x: &[f32],
    moe_hook: Option<F>,
) where
    F: FnOnce(&mut [f32], &[f32]),
{
    let n_embd = N_EMBD as usize;
    assert_eq!(out.len(), n_embd, "layer_ffn_one: out dim must be N_EMBD");
    assert_eq!(x.len(), n_embd, "layer_ffn_one: in dim must be N_EMBD");
    assert_eq!(
        weights.ffn_norm.len(),
        n_embd,
        "layer_ffn_one: ffn_norm length must be N_EMBD"
    );

    // norm = rms_norm(x) * ffn_norm
    let mut norm = vec![0.0f32; n_embd];
    rms_norm_weight(&mut norm, x, weights.ffn_norm, RMS_EPS);

    // shared expert
    let mut shared = vec![0.0f32; n_embd];
    shared_ffn_one(&mut shared, &weights.shared, &norm);

    // optional routed MoE
    if let Some(hook) = moe_hook {
        let mut moe = vec![0.0f32; n_embd];
        hook(&mut moe, &norm);
        for i in 0..n_embd {
            out[i] = shared[i] + moe[i];
        }
    } else {
        out.copy_from_slice(&shared);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;

    /// Build a Q8_0 weight matrix `[out_dim, in_dim]` whose every block has
    /// scale 0 and qs = 0. Such a matrix is the all-zero matrix: every
    /// matvec against it returns 0 regardless of input.
    fn zero_q8_0(out_dim: usize, in_dim: usize) -> Vec<u8> {
        let row_bytes = q8_0_row_bytes(in_dim);
        // f16 0.0 == 0x0000, qs = 0; vec![0u8; ..] already gives us that.
        vec![0u8; out_dim * row_bytes]
    }

    /// Q8_0 weight matrix that encodes `c * I` for a square shape. Each row
    /// has a single block whose scale is `c / 1` and whose only non-zero
    /// quant lane is the diagonal one with value 1. (Only valid when
    /// `in_dim <= 32`, which is fine for the small tests below.)
    fn diag_q8_0(dim: usize, c: f32) -> Vec<u8> {
        assert!(dim <= 32);
        let row_bytes = q8_0_row_bytes(dim);
        let mut w = vec![0u8; dim * row_bytes];
        let scale_bits = f32_to_f16(c).to_le_bytes();
        for r in 0..dim {
            let row = &mut w[r * row_bytes..(r + 1) * row_bytes];
            row[0] = scale_bits[0];
            row[1] = scale_bits[1];
            // qs[r] = 1 (as i8); other lanes stay 0.
            row[2 + r] = 1u8;
        }
        w
    }

    #[test]
    fn shared_ffn_zero_weights_yields_zero() {
        // With all weights = 0, the FFN output must be exactly zero
        // regardless of input. This exercises the full plumbing
        // (quantize -> matvec -> swiglu -> matvec) at full DS4 shapes.
        let n_embd = N_EMBD as usize;
        let n_ff = N_FF_EXP as usize;
        let gate = zero_q8_0(n_ff, n_embd);
        let up = zero_q8_0(n_ff, n_embd);
        let down = zero_q8_0(n_embd, n_ff);
        let w = SharedFfnWeights { gate: &gate, up: &up, down: &down };

        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut out = vec![1.0f32; n_embd]; // non-zero sentinel
        shared_ffn_one(&mut out, &w, &x);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "out[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn shared_ffn_shape_matches_n_embd() {
        // Quick smoke test: feed random-ish input through zero weights, check
        // shape contract (length unchanged, no panic on the standard DS4
        // dims).
        let n_embd = N_EMBD as usize;
        let n_ff = N_FF_EXP as usize;
        let gate = zero_q8_0(n_ff, n_embd);
        let up = zero_q8_0(n_ff, n_embd);
        let down = zero_q8_0(n_embd, n_ff);
        let w = SharedFfnWeights { gate: &gate, up: &up, down: &down };
        let x = vec![0.5f32; n_embd];
        let mut out = vec![0.0f32; n_embd];
        shared_ffn_one(&mut out, &w, &x);
        assert_eq!(out.len(), n_embd);
    }

    #[test]
    fn layer_ffn_one_routes_moe_hook() {
        // Use a tiny shape via a contrived test: layer_ffn_one is hard-wired
        // to N_EMBD, so we instead exercise just the routing logic by
        // wrapping shared_ffn_one with zero weights (-> 0) and feeding a
        // hook that writes a known constant.
        let n_embd = N_EMBD as usize;
        let n_ff = N_FF_EXP as usize;
        let gate = zero_q8_0(n_ff, n_embd);
        let up = zero_q8_0(n_ff, n_embd);
        let down = zero_q8_0(n_embd, n_ff);
        let ffn_norm = vec![1.0f32; n_embd];
        let weights = LayerFfnWeights {
            ffn_norm: &ffn_norm,
            shared: SharedFfnWeights { gate: &gate, up: &up, down: &down },
        };
        let x = vec![1.0f32; n_embd];
        let mut out = vec![0.0f32; n_embd];

        // With shared expert weights all zero, only the MoE hook should
        // contribute. We write a constant `7.0` into every output lane.
        let hook = |moe: &mut [f32], _norm: &[f32]| {
            for v in moe.iter_mut() { *v = 7.0; }
        };
        layer_ffn_one(&mut out, &weights, &x, Some(hook));
        for &v in out.iter() {
            assert_eq!(v, 7.0);
        }

        // And without a hook, output must be exactly the (zero) shared
        // expert.
        let mut out2 = vec![1.0f32; n_embd];
        layer_ffn_one::<fn(&mut [f32], &[f32])>(&mut out2, &weights, &x, None);
        for &v in out2.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn diag_q8_0_helper_smoke() {
        // Sanity-check our diag_q8_0 builder against dot_q8_0_row directly.
        // (Pure unit-level: makes sure the test helpers themselves work,
        // so the larger tests above are trustworthy.)
        let dim = 8usize;
        let w = diag_q8_0(dim, 2.0);
        let row_bytes = q8_0_row_bytes(dim);
        // Activation = [1, 2, 3, ..., 8].
        let x: Vec<f32> = (1..=dim as i32).map(|i| i as f32).collect();
        let blocks = (dim + 31) / 32;
        let mut xq = vec![0i8; blocks * 32];
        let mut xscale = vec![0.0f32; blocks];
        // Pad x to a full block for quantize_q8_0_activation.
        let mut padded = x.clone();
        padded.resize(blocks * 32, 0.0);
        quantize_q8_0_activation(&padded, &mut xq, &mut xscale);
        for r in 0..dim {
            let row = &w[r * row_bytes..(r + 1) * row_bytes];
            let got = dot_q8_0_row(row, &xq, &xscale, dim);
            let want = 2.0 * x[r];
            // Q8_0 introduces some quant error; allow a generous tolerance.
            assert!((got - want).abs() < 0.5, "row {r}: got {got}, want {want}");
        }
    }
}
