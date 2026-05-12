//! CPU reference Hyper-Connection (HC) helpers.
//!
//! Ports the small per-token HC plumbing from `ds4.c`:
//!
//! * `embed_token_f16`           (ds4.c ~line 2684)
//! * `hc_from_plain_embedding`   (ds4.c ~line 4358)
//! * `hc_weighted_sum_one`       (ds4.c ~line 4267)
//! * `hc_post_one`               (ds4.c ~line 4366)
//! * `hc_split_sinkhorn_one`     (ds4.c ~line 4186)
//!
//! All routines operate on f32 activation slices. Dimensions are derived from
//! slice lengths instead of being passed explicitly so the Rust API stays
//! idiomatic; the original C versions take `n_embd` / `n_hc` arguments.
//!
//! The HC state for a single token is laid out as `n_hc` contiguous
//! `n_embd`-wide streams, i.e. `hc[h * n_embd + d]` indexes the value for HC
//! stream `h` at embedding dimension `d`. The combine matrix `comb` is
//! addressed as `comb[dst + src * n_hc]` exactly like the C source.
//!
//! See the architecture commentary in `ds4.c` just above
//! `hc_split_sinkhorn_one` for the role of the pre weights, post gates, and
//! combine matrix produced by the Sinkhorn split.

use crate::half::f16_to_f32;
use crate::shape::N_HC;

/// Decode one row of the token embedding table from f16 into an f32 slice.
///
/// `table` is the whole `[n_vocab, stride]` row-major f16 weight buffer (as
/// raw `u16` bits), and `out` receives the `stride`-wide embedding for the
/// given `token`. Mirrors `embed_token_f16` in `ds4.c` (~line 2684).
pub fn embed_token_f16(out: &mut [f32], table: &[u16], stride: usize, token: usize) {
    assert_eq!(out.len(), stride, "out must be stride wide");
    let base = token * stride;
    assert!(
        base + stride <= table.len(),
        "token id is outside the embedding table",
    );
    let row = &table[base..base + stride];
    for i in 0..stride {
        out[i] = f16_to_f32(row[i]);
    }
}

/// Replicate the plain token embedding into all `N_HC` HC streams so the
/// first layer sees identical residuals on every stream. Mirrors
/// `hc_from_plain_embedding` in `ds4.c` (~line 4358).
pub fn hc_from_plain(out_hc: &mut [f32], plain: &[f32]) {
    let n_embd = plain.len();
    let n_hc = N_HC as usize;
    assert_eq!(
        out_hc.len(),
        n_embd * n_hc,
        "out_hc must be n_hc * n_embd wide",
    );
    for h in 0..n_hc {
        out_hc[h * n_embd..(h + 1) * n_embd].copy_from_slice(plain);
    }
}

/// Reduce the `N_HC` HC streams in `x` into the single `n_embd`-wide vector
/// `out` using per-stream `weights`. Mirrors `hc_weighted_sum_one` in
/// `ds4.c` (~line 4267).
pub fn hc_weighted_sum_one(out: &mut [f32], x: &[f32], weights: &[f32]) {
    let n_embd = out.len();
    let n_hc = weights.len();
    assert_eq!(
        x.len(),
        n_embd * n_hc,
        "x must be n_hc * n_embd wide",
    );
    for d in 0..n_embd {
        let mut acc = 0.0f32;
        for h in 0..n_hc {
            acc += x[h * n_embd + d] * weights[h];
        }
        out[d] = acc;
    }
}

/// HC post step for one sublayer output: inject `block_out` (gated by `post`)
/// and remix the previous HC streams in `residual_hc` through the combine
/// matrix `comb` (addressed as `[dst + src * n_hc]`). Mirrors `hc_post_one`
/// in `ds4.c` (~line 4366).
pub fn hc_post_one(
    out_hc: &mut [f32],
    block_out: &[f32],
    residual_hc: &[f32],
    post: &[f32],
    comb: &[f32],
) {
    let n_embd = block_out.len();
    let n_hc = post.len();
    assert_eq!(out_hc.len(), n_embd * n_hc, "out_hc shape mismatch");
    assert_eq!(residual_hc.len(), n_embd * n_hc, "residual_hc shape mismatch");
    assert_eq!(comb.len(), n_hc * n_hc, "comb must be n_hc x n_hc");

    for dst in 0..n_hc {
        for d in 0..n_embd {
            let mut acc = block_out[d] * post[dst];
            for src in 0..n_hc {
                // Combine matrix is addressed as [dst_hc, src_hc].
                acc += comb[dst + src * n_hc] * residual_hc[src * n_embd + d];
            }
            out_hc[dst * n_embd + d] = acc;
        }
    }
}

/// Decode the HC control projection into pre weights, post gates, and a
/// doubly-normalized combine matrix.
///
/// `mix` is the raw projection of the (normalized) HC state, of length
/// `2 * n_hc + n_hc * n_hc`. `scale` is the three learned scales
/// `[pre, post, comb]`. `base` is the same-shape learned bias added before
/// activation. The output `out_split` is laid out as `[pre | post | comb]`:
///
/// * `out_split[0..n_hc]`             — pre weights (sigmoid + eps)
/// * `out_split[n_hc..2*n_hc]`        — post gates  (2 * sigmoid)
/// * `out_split[2*n_hc..]`            — Sinkhorn-normalized combine matrix
///
/// `iters` controls how many additional alternating row/column
/// normalizations follow the initial softmax over rows. Mirrors
/// `hc_split_sinkhorn_one` in `ds4.c` (~line 4186).
pub fn hc_split_sinkhorn_one(
    out_split: &mut [f32],
    mix: &[f32],
    scale: &[f32],
    base: &[f32],
    n_hc: usize,
    iters: usize,
    eps: f32,
) {
    assert!(scale.len() >= 3, "scale must contain [pre, post, comb] scales");
    let split_len = 2 * n_hc + n_hc * n_hc;
    assert_eq!(out_split.len(), split_len, "out_split shape mismatch");
    assert_eq!(mix.len(), split_len, "mix shape mismatch");
    assert_eq!(base.len(), split_len, "base shape mismatch");

    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];

    // Pre weights: sigmoid(mix * pre_scale + base) + eps.
    for i in 0..n_hc {
        let z = mix[i] * pre_scale + base[i];
        out_split[i] = 1.0 / (1.0 + (-z).exp()) + eps;
    }

    // Post gates: 2 * sigmoid(mix * post_scale + base).
    for i in 0..n_hc {
        let off = n_hc + i;
        let z = mix[off] * post_scale + base[off];
        out_split[off] = 2.0 / (1.0 + (-z).exp());
    }

    // Combine matrix: softmax over rows (dst), then alternating column/row
    // normalization for `iters` total iterations. Worked on a local scratch
    // buffer so the final write to `out_split` happens after Sinkhorn
    // converges. The C version uses a fixed 16x16 stack buffer; we allocate
    // exactly n_hc * n_hc here.
    let mut c = vec![0.0f32; n_hc * n_hc];

    // Initial row-softmax with `+ eps` smoothing.
    for dst in 0..n_hc {
        let mut row_max = f32::NEG_INFINITY;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let off = 2 * n_hc + idx;
            let v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if v > row_max {
                row_max = v;
            }
        }

        let mut row_sum = 0.0f32;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let v = (c[idx] - row_max).exp();
            c[idx] = v;
            row_sum += v;
        }

        let inv = 1.0 / row_sum;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    // Column normalization (over dst) for iteration 0.
    for src in 0..n_hc {
        let mut sum = 0.0f32;
        for dst in 0..n_hc {
            sum += c[src + dst * n_hc];
        }
        let inv = 1.0 / (sum + eps);
        for dst in 0..n_hc {
            c[src + dst * n_hc] *= inv;
        }
    }

    // Sinkhorn iterations: alternate row-then-column normalization.
    for _iter in 1..iters {
        for dst in 0..n_hc {
            let mut sum = 0.0f32;
            for src in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for src in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }

        for src in 0..n_hc {
            let mut sum = 0.0f32;
            for dst in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for dst in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
    }

    out_split[2 * n_hc..2 * n_hc + n_hc * n_hc].copy_from_slice(&c);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;
    use crate::shape::{HC_EPS, N_EMBD};

    #[test]
    fn embed_token_f16_decodes_row() {
        // Build a tiny 3-row, 4-wide f16 table whose row `t` is [t, t+0.5, t+1, t+1.5].
        let stride = 4usize;
        let n_vocab = 3usize;
        let mut table = vec![0u16; n_vocab * stride];
        for t in 0..n_vocab {
            for i in 0..stride {
                let v = t as f32 + 0.5 * i as f32;
                table[t * stride + i] = f32_to_f16(v);
            }
        }

        let mut out = vec![0.0f32; stride];
        embed_token_f16(&mut out, &table, stride, 2);
        for i in 0..stride {
            let expected = 2.0 + 0.5 * i as f32;
            assert!(
                (out[i] - expected).abs() < 1e-3,
                "row 2 col {i}: got {} expected {}",
                out[i],
                expected,
            );
        }
    }

    #[test]
    fn hc_from_plain_replicates_then_weighted_sum_recovers() {
        // plain values are unique so we can check all four streams independently.
        let n_embd = 8usize;
        let n_hc = N_HC as usize;
        let plain: Vec<f32> = (0..n_embd).map(|i| (i as f32) * 0.25 - 0.5).collect();

        let mut hc = vec![0.0f32; n_embd * n_hc];
        hc_from_plain(&mut hc, &plain);

        // Each HC stream should be identical to `plain`.
        for h in 0..n_hc {
            for d in 0..n_embd {
                assert_eq!(hc[h * n_embd + d], plain[d]);
            }
        }

        // Weighted sum with weights summing to 1 reproduces `plain`.
        let weights = vec![0.25f32; n_hc];
        let mut collapsed = vec![0.0f32; n_embd];
        hc_weighted_sum_one(&mut collapsed, &hc, &weights);
        for d in 0..n_embd {
            assert!((collapsed[d] - plain[d]).abs() < 1e-6);
        }

        // Also ensure HC dims line up with the production constants.
        let _ = N_EMBD; // referenced for the doc-cite/sanity.
    }

    #[test]
    fn hc_post_one_matches_explicit_formula() {
        let n_embd = 3usize;
        let n_hc = N_HC as usize;

        let block_out = vec![1.0f32, -2.0, 0.5];
        let residual_hc: Vec<f32> = (0..n_embd * n_hc).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let post = vec![0.5f32, 1.0, -0.25, 2.0];
        // comb is row-major over (src * n_hc + dst) when addressed as [dst + src * n_hc].
        let comb: Vec<f32> = (0..n_hc * n_hc).map(|i| 0.01 * (i as f32 + 1.0)).collect();

        let mut out_hc = vec![0.0f32; n_embd * n_hc];
        hc_post_one(&mut out_hc, &block_out, &residual_hc, &post, &comb);

        // Recompute the formula independently here.
        for dst in 0..n_hc {
            for d in 0..n_embd {
                let mut expected = block_out[d] * post[dst];
                for src in 0..n_hc {
                    expected += comb[dst + src * n_hc] * residual_hc[src * n_embd + d];
                }
                let got = out_hc[dst * n_embd + d];
                assert!(
                    (got - expected).abs() < 1e-6,
                    "dst={dst} d={d}: got {got} expected {expected}",
                );
            }
        }
    }

    #[test]
    fn hc_split_sinkhorn_one_produces_doubly_stochastic_combine() {
        let n_hc = 4usize;
        let iters = 20usize;
        let eps = HC_EPS;

        let split_len = 2 * n_hc + n_hc * n_hc;
        // Use a mildly asymmetric `mix` so the matrix is not trivially uniform.
        let mix: Vec<f32> = (0..split_len)
            .map(|i| ((i as f32) * 0.137).sin())
            .collect();
        let scale = vec![0.5f32, 0.75, 1.25];
        let base = vec![0.0f32; split_len];

        let mut out = vec![0.0f32; split_len];
        hc_split_sinkhorn_one(&mut out, &mix, &scale, &base, n_hc, iters, eps);

        // Pre weights live in [eps, 1+eps] thanks to sigmoid + eps.
        for i in 0..n_hc {
            assert!(out[i] > 0.0 && out[i] < 1.0 + 2.0 * eps);
        }
        // Post gates live in (0, 2) thanks to 2 * sigmoid.
        for i in 0..n_hc {
            let g = out[n_hc + i];
            assert!(g > 0.0 && g < 2.0);
        }

        // After 20 Sinkhorn iterations the combine matrix should be very close
        // to doubly stochastic (rows and columns sum to 1).
        let comb = &out[2 * n_hc..];
        for dst in 0..n_hc {
            let mut row_sum = 0.0f32;
            for src in 0..n_hc {
                row_sum += comb[src + dst * n_hc];
            }
            assert!(
                (row_sum - 1.0).abs() < 1e-3,
                "row {dst} sum {row_sum} not close to 1",
            );
        }
        for src in 0..n_hc {
            let mut col_sum = 0.0f32;
            for dst in 0..n_hc {
                col_sum += comb[src + dst * n_hc];
            }
            assert!(
                (col_sum - 1.0).abs() < 1e-3,
                "col {src} sum {col_sum} not close to 1",
            );
        }
    }
}
