//! Forward-pass integration smoke tests.
//!
//! These chain together a handful of the primitives exposed by `ds4-core`
//! into tiny synthetic transformer-state pipelines so we catch wiring-level
//! regressions cheaply. None of the production fixed-shape entry points
//! (`output_logits_one`, `shared_ffn_one`, `layer_routed_moe_one`) are used
//! here because they hard-code the DS4 dimensions; instead we re-use the
//! lower-level primitives at tiny dims (`n_embd <= 64`, `n_vocab <= 32`,
//! `n_hc = 4`) so the suite runs in milliseconds.

use ds4_core::half::{f16_to_f32, f32_to_f16};
use ds4_core::hc::{hc_from_plain, hc_post_one, hc_weighted_sum_one};
use ds4_core::matvec::{
    dot_q8_0_row, matvec_f16, matvec_f32, matvec_q8_0, quantize_q8_0_activation,
    Q8_0_BLOCK_BYTES,
};
use ds4_core::moe::topk_desc;
use ds4_core::nn::{rms_norm_no_weight, rms_norm_weight, softmax_inplace, swiglu};
use ds4_core::sampler::{argmax, sample};
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// Tiny test helpers
// ---------------------------------------------------------------------------

/// Pack an `[out_dim, in_dim]` row-major f32 matrix into f16 (`u16`) bits.
fn f32_matrix_to_f16(w: &[f32]) -> Vec<u16> {
    w.iter().map(|&v| f32_to_f16(v)).collect()
}

/// Number of bytes per Q8_0 row for `in_dim` features.
fn q8_0_row_bytes(in_dim: usize) -> usize {
    let blocks = (in_dim + 31) / 32;
    blocks * Q8_0_BLOCK_BYTES
}

/// Build a Q8_0 weight matrix `[out_dim, in_dim]` by quantizing each row
/// independently from a dense f32 row. `in_dim` may exceed 32 — the routine
/// emits one block per 32-element chunk, matching the layout consumed by
/// [`dot_q8_0_row`] and [`matvec_q8_0`].
fn quantize_q8_0_matrix(rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
    let row_bytes = q8_0_row_bytes(in_dim);
    let blocks = (in_dim + 31) / 32;
    let padded = blocks * 32;
    let mut out = vec![0u8; rows.len() * row_bytes];
    let mut row_buf = vec![0.0f32; padded];
    let mut qs = vec![0i8; padded];
    let mut scales = vec![0.0f32; blocks];
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), in_dim);
        row_buf.iter_mut().for_each(|v| *v = 0.0);
        row_buf[..in_dim].copy_from_slice(row);
        quantize_q8_0_activation(&row_buf, &mut qs, &mut scales);
        let row_out = &mut out[r * row_bytes..(r + 1) * row_bytes];
        for b in 0..blocks {
            let base = b * Q8_0_BLOCK_BYTES;
            let scale_bits = f32_to_f16(scales[b]).to_le_bytes();
            row_out[base] = scale_bits[0];
            row_out[base + 1] = scale_bits[1];
            // i8 -> u8 reinterpret is just a bit copy.
            for i in 0..32 {
                row_out[base + 2 + i] = qs[b * 32 + i] as u8;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 1. matvec_f16_identity_logits
// ---------------------------------------------------------------------------

/// Set the output projection to a (near-)identity matrix in f16, feed an
/// activation whose argmax is at a known index, and verify
/// `argmax(out)` picks that index.
#[test]
fn matvec_f16_identity_logits() {
    let n_embd = 16usize;
    let n_vocab = 16usize;

    // f16 identity: w[r * n_embd + r] = 1.0, rest 0.0.
    let mut w_f32 = vec![0.0f32; n_vocab * n_embd];
    for i in 0..n_vocab {
        w_f32[i * n_embd + i] = 1.0;
    }
    let w = f32_matrix_to_f16(&w_f32);

    // Activation whose peak is at index 7.
    let mut x = vec![0.1f32; n_embd];
    x[7] = 5.0;

    let mut logits = vec![0.0f32; n_vocab];
    matvec_f16(&mut logits, &w, &x);

    // Round-trip through f16 introduces no error for these exact values; the
    // identity row picks out `x[r]` per output channel.
    for i in 0..n_vocab {
        assert!(
            (logits[i] - x[i]).abs() < 1e-3,
            "logits[{i}] = {} (want {})", logits[i], x[i],
        );
    }
    assert_eq!(argmax(&logits), 7);
}

// ---------------------------------------------------------------------------
// 2. swiglu_then_matvec_f32
// ---------------------------------------------------------------------------

/// Chain `swiglu(gate, up) -> matvec_f32(W_down, .)` and compare against a
/// scalar reference implementation that recomputes the same math directly.
#[test]
fn swiglu_then_matvec_f32() {
    let n_ff = 16usize;
    let n_embd = 8usize;

    // Deterministic gate/up vectors with a mix of signs.
    let gate: Vec<f32> = (0..n_ff).map(|i| (i as f32 - 7.5) * 0.3).collect();
    let up:   Vec<f32> = (0..n_ff).map(|i| ((i as f32) * 0.21).cos()).collect();

    let mut mid = vec![0.0f32; n_ff];
    swiglu(&mut mid, &gate, &up);

    // Down projection W_down: shape [n_embd, n_ff], deterministic values.
    let mut w_down = vec![0.0f32; n_embd * n_ff];
    for r in 0..n_embd {
        for c in 0..n_ff {
            w_down[r * n_ff + c] = ((r * 13 + c * 7) as f32).sin() * 0.1;
        }
    }

    let mut out = vec![0.0f32; n_embd];
    matvec_f32(&mut out, &w_down, &mid);

    // Reference: silu(g) * u, then dense matvec, computed independently.
    fn silu(x: f32) -> f32 {
        let s = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let z = x.exp();
            z / (1.0 + z)
        };
        x * s
    }
    let ref_mid: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect();
    let mut ref_out = vec![0.0f32; n_embd];
    for r in 0..n_embd {
        let mut acc = 0.0f32;
        for c in 0..n_ff {
            acc += w_down[r * n_ff + c] * ref_mid[c];
        }
        ref_out[r] = acc;
    }

    for i in 0..n_embd {
        assert!(
            (out[i] - ref_out[i]).abs() < 1e-5,
            "out[{i}] = {} (ref {})", out[i], ref_out[i],
        );
        assert!(out[i].is_finite());
    }

    // Sanity: mid should match the reference too.
    for i in 0..n_ff {
        assert!((mid[i] - ref_mid[i]).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// 3. softmax_then_sample_concentrates_at_top
// ---------------------------------------------------------------------------

/// With one extremely large logit, `sample` should converge to that index
/// across many seeds even with a non-trivial (large) temperature.
#[test]
fn softmax_then_sample_concentrates_at_top() {
    let n_vocab = 32usize;
    // Build a flat-ish logit vector with one huge peak that survives even
    // when divided by a large temperature.
    let peak = 17usize;
    let base_logits: Vec<f32> = (0..n_vocab).map(|i| 0.01 * i as f32).collect();
    let mut peaked = base_logits.clone();
    peaked[peak] = 1.0e6; // gigantic — survives any tempering used here.

    // First, confirm a plain `softmax_inplace` of the peaked vector
    // assigns essentially all mass to `peak`.
    let mut probs = peaked.clone();
    softmax_inplace(&mut probs);
    let s: f32 = probs.iter().sum();
    assert!((s - 1.0).abs() < 1e-5, "softmax must sum to ~1 (got {s})");
    assert!(probs[peak] > 0.999, "peak mass too low: {}", probs[peak]);

    // Now drive the full sampler at temperature = 2.0 (well above the
    // argmax-fallback threshold) across many seeds and ensure every draw
    // lands on `peak`.
    let trials = 64;
    for seed in 0..trials {
        let mut logits = peaked.clone();
        let mut rng = rand_xoshiro::SplitMix64::seed_from_u64(seed as u64 + 1);
        let tok = sample(
            &mut logits,
            /* temperature */ 2.0,
            /* top_k */ 0,
            /* top_p */ 1.0,
            /* min_p */ 0.0,
            &mut rng,
        );
        assert_eq!(
            tok, peak as i32,
            "seed {seed}: sampler picked {tok}, expected {peak}",
        );
    }

    // Sanity: with temperature = 0, sample falls through to argmax.
    let mut logits = peaked.clone();
    let mut rng = rand_xoshiro::SplitMix64::seed_from_u64(42);
    let tok = sample(&mut logits, 0.0, 0, 1.0, 0.0, &mut rng);
    assert_eq!(tok, peak as i32);
}

// ---------------------------------------------------------------------------
// 4. q8_0_round_trip_through_matvec
// ---------------------------------------------------------------------------

/// Quantize an activation to Q8_0, dot it against a Q8_0-quantized weight
/// row, and compare the result to a plain f32 reference. Q8_0 has at most
/// roughly one quant of error per lane (per scale), so we allow a slack
/// tolerance scaled by the row magnitude.
#[test]
fn q8_0_round_trip_through_matvec() {
    let in_dim = 64usize; // exactly two Q8_0 blocks.
    let out_dim = 4usize;

    // Activation: smooth values across [-1.5, 1.5].
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i as f32) * 0.13 - 4.0).sin() * 1.5)
        .collect();

    // Weight rows: simple deterministic patterns covering positive,
    // negative, and signed mixed.
    let rows: Vec<Vec<f32>> = (0..out_dim)
        .map(|r| {
            (0..in_dim)
                .map(|c| ((r * 7 + c * 3) as f32 * 0.05).cos())
                .collect::<Vec<f32>>()
        })
        .collect();

    let w_q = quantize_q8_0_matrix(&rows, in_dim);
    let mut out_q = vec![0.0f32; out_dim];
    matvec_q8_0(&mut out_q, &w_q, &x);

    // f32 reference.
    let mut out_ref = vec![0.0f32; out_dim];
    for r in 0..out_dim {
        let mut acc = 0.0f32;
        for c in 0..in_dim {
            acc += rows[r][c] * x[c];
        }
        out_ref[r] = acc;
    }

    // Tolerance: Q8_0 has a worst-case relative quant error of ~1/127. We
    // also pay it twice (one quant on activations, one on weights), and the
    // row sum compounds; a generous absolute slack proportional to the
    // sum-of-abs of x and rows is plenty.
    for r in 0..out_dim {
        let mag: f32 = rows[r].iter().zip(x.iter()).map(|(w, v)| w.abs() * v.abs()).sum();
        let tol = (mag / 127.0).max(1e-3) * 4.0;
        assert!(
            (out_q[r] - out_ref[r]).abs() < tol,
            "row {r}: q8_0 = {} ref = {} tol = {}", out_q[r], out_ref[r], tol,
        );
        assert!(out_q[r].is_finite());
    }

    // Also exercise `dot_q8_0_row` directly with the in-place quantizer so
    // the per-row API matches `matvec_q8_0`.
    let blocks = (in_dim + 31) / 32;
    let mut xq = vec![0i8; blocks * 32];
    let mut xscale = vec![0.0f32; blocks];
    let mut padded = x.clone();
    padded.resize(blocks * 32, 0.0);
    quantize_q8_0_activation(&padded, &mut xq, &mut xscale);
    let row_bytes = q8_0_row_bytes(in_dim);
    for r in 0..out_dim {
        let row = &w_q[r * row_bytes..(r + 1) * row_bytes];
        let got = dot_q8_0_row(row, &xq, &xscale, in_dim);
        assert!(
            (got - out_q[r]).abs() < 1e-4,
            "row {r}: dot_q8_0_row {got} disagreed with matvec_q8_0 {}", out_q[r],
        );
    }

    // Confirm f16 round-trip identity on a scale we know fits: f32 -> f16
    // -> f32 must preserve integer powers of two exactly. (Sanity-check the
    // half module used above to pack Q8_0 scale bits.)
    for v in [1.0f32, 0.5, -2.0, 16.0, -0.125] {
        let back = f16_to_f32(f32_to_f16(v));
        assert!((back - v).abs() < 1e-6, "f16 roundtrip drifted on {v} -> {back}");
    }
}

// ---------------------------------------------------------------------------
// 5. output_head_shape_invariants
// ---------------------------------------------------------------------------

/// Mini HC-collapse + RMSNorm + Q8_0 vocab projection at toy dimensions.
/// The production `output_logits_one` hard-codes the full DS4 shapes
/// (N_EMBD = 4096, N_VOCAB = 129_280), so this test rebuilds the same
/// pipeline from primitives at `n_embd = 8`, `n_vocab = 8`, `n_hc = 4`. It
/// also exercises `hc_from_plain`, `hc_post_one`, and `topk_desc` to verify
/// every output is finite and the logits vector has the right length.
#[test]
fn output_head_shape_invariants() {
    let n_embd = 8usize;
    let n_vocab = 8usize;
    let n_hc = 4usize; // must match shape::N_HC for hc_from_plain.
    assert_eq!(n_hc, ds4_core::shape::N_HC as usize);

    // Plain embedding: a single token's residual.
    let plain: Vec<f32> = (0..n_embd).map(|i| (i as f32 - 3.5) * 0.2).collect();

    // Replicate into all HC streams.
    let mut hc_state = vec![0.0f32; n_hc * n_embd];
    hc_from_plain(&mut hc_state, &plain);
    for h in 0..n_hc {
        for d in 0..n_embd {
            assert_eq!(hc_state[h * n_embd + d], plain[d]);
        }
    }

    // Push a fake "block output" through `hc_post_one` to verify the
    // residual remix path. Combine matrix = identity-ish so the residual
    // streams come through unchanged before the post bias is added.
    let block_out: Vec<f32> = plain.iter().map(|v| v * 0.5).collect();
    let post = vec![0.25f32; n_hc];
    let mut comb = vec![0.0f32; n_hc * n_hc];
    for h in 0..n_hc {
        comb[h + h * n_hc] = 1.0; // [dst + src * n_hc] identity.
    }
    let mut hc_after = vec![0.0f32; n_hc * n_embd];
    hc_post_one(&mut hc_after, &block_out, &hc_state, &post, &comb);
    for &v in &hc_after {
        assert!(v.is_finite());
    }

    // HC weighted sum: collapse streams with uniform weights summing to 1.
    let weights = vec![1.0f32 / n_hc as f32; n_hc];
    let mut collapsed = vec![0.0f32; n_embd];
    hc_weighted_sum_one(&mut collapsed, &hc_after, &weights);

    // Final RMSNorm with a per-channel gain.
    let gain: Vec<f32> = (0..n_embd).map(|i| 0.5 + (i as f32) * 0.1).collect();
    let mut normed = vec![0.0f32; n_embd];
    rms_norm_weight(&mut normed, &collapsed, &gain, 1e-6);
    for &v in &normed {
        assert!(v.is_finite());
    }

    // Sanity check `rms_norm_no_weight` on the same input.
    let mut normed_plain = vec![0.0f32; n_embd];
    rms_norm_no_weight(&mut normed_plain, &collapsed, 1e-6);
    for &v in &normed_plain {
        assert!(v.is_finite());
    }

    // Build a tiny Q8_0 vocab projection: each row is a different
    // smooth pattern of `n_embd` weights.
    let rows: Vec<Vec<f32>> = (0..n_vocab)
        .map(|r| {
            (0..n_embd)
                .map(|c| ((r as f32) * 0.7 + (c as f32) * 0.15).sin() * 0.5)
                .collect()
        })
        .collect();
    let output_w = quantize_q8_0_matrix(&rows, n_embd);

    let mut logits = vec![f32::NAN; n_vocab];
    matvec_q8_0(&mut logits, &output_w, &normed);

    // Shape invariants.
    assert_eq!(logits.len(), n_vocab);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logits[{i}] = {v} is not finite");
    }

    // Sampler agreement: `argmax` and `topk_desc` (k = 1) should pick the
    // same token.
    let mut top1 = [-1i32; 1];
    topk_desc(&logits, 1, &mut top1);
    assert_eq!(top1[0], argmax(&logits));
    assert!(top1[0] >= 0 && (top1[0] as usize) < n_vocab);

    // A second pass through the full sampler at temperature 0 must agree
    // with `argmax` as well.
    let mut logits_copy = logits.clone();
    let mut rng = rand_xoshiro::SplitMix64::seed_from_u64(1234);
    let picked = sample(&mut logits_copy, 0.0, 0, 1.0, 0.0, &mut rng);
    assert_eq!(picked, argmax(&logits));
}
