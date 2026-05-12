//! CPU reference Mixture-of-Experts (MoE) routing.
//!
//! This module ports the single-token routed MoE path from `ds4.c`. The C
//! production code dispatches IQ2_XXS gate/up and Q2_K down projections through
//! dedicated quantized matvec kernels; here we use the f16 / Q8_0 primitives
//! already available in `crate::matvec` to provide a CPU reference that
//! mirrors the C control flow.
//!
//! Cross-reference table:
//!
//! * [`hc_split_sinkhorn_one`] — `ds4.c` `hc_split_sinkhorn_one` (~line 4186)
//! * [`topk_desc`]             — `ds4.c` `topk_desc` (~line 5211)
//! * [`router_probs_one`]      — `ds4.c` `layer_router_probs_one` (~line 5169)
//! * [`topk_selected_experts_from_probs`]
//!                             — `ds4.c` `layer_topk_selected_experts_from_probs` (~line 5246)
//! * [`expert_swiglu_one`]     — body of the `trace` branch of
//!                                `layer_routed_moe_one` (~line 5327)
//! * [`layer_routed_moe_one`]  — `ds4.c` `layer_routed_moe_one` (~line 5278)
//!
//! This file deliberately only depends on items re-exported from `crate::nn`,
//! `crate::matvec`, and `crate::shape` plus `std`.

use crate::matvec::{matvec_f16, matvec_q8_0};
use crate::nn::{silu, softplus_stable};
use crate::shape::{
    EXPERT_WEIGHT_SCALE, HC_EPS, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_HC,
    N_HC_SINKHORN_ITER,
};

/// Maximum number of hyper-connection streams supported by the stack-allocated
/// Sinkhorn buffer. The C code uses `float c[16 * 16]`; we mirror that.
const SINKHORN_MAX_HC: usize = 16;

/// Run one Sinkhorn iteration block that splits the control mix into pre /
/// post / comb streams.
///
/// Direct port of `hc_split_sinkhorn_one` in `ds4.c` (~line 4186).
///
/// Layout of the inputs/outputs follows the C version exactly:
///
/// * `mix` and `base` are `(2 + n_hc) * n_hc` floats:
///   - `[0 .. n_hc]`                       — pre stream (post sigmoid)
///   - `[n_hc .. 2*n_hc]`                  — post stream (post `2*sigmoid`)
///   - `[2*n_hc .. (2 + n_hc)*n_hc]`       — comb matrix `(src, dst)` row-major
/// * `scale[0]` is the pre scale, `scale[1]` the post scale, `scale[2]` the
///   comb scale.
/// * `out` has the same layout as `mix`/`base`.
///
/// The comb matrix is first row-softmaxed (dst-major rows, `n_hc` entries per
/// row over `src`), then renormalized columnwise; the remaining `iters - 1`
/// iterations alternate dst- and src-normalization, matching Sinkhorn's
/// doubly-stochastic projection.
pub fn hc_split_sinkhorn_one(
    out: &mut [f32],
    mix: &[f32],
    scale: &[f32; 3],
    base: &[f32],
    n_hc: usize,
    iters: usize,
    eps: f32,
) {
    assert!(n_hc <= SINKHORN_MAX_HC, "n_hc exceeds SINKHORN_MAX_HC");
    let dim = (2 + n_hc) * n_hc;
    assert_eq!(mix.len(), dim);
    assert_eq!(base.len(), dim);
    assert_eq!(out.len(), dim);
    assert!(iters >= 1, "Sinkhorn iters must be >= 1");

    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];

    // pre stream: sigmoid(mix * pre_scale + base) + eps
    for i in 0..n_hc {
        let z = mix[i] * pre_scale + base[i];
        out[i] = 1.0 / (1.0 + (-z).exp()) + eps;
    }

    // post stream: 2 * sigmoid(mix * post_scale + base)
    for i in 0..n_hc {
        let off = n_hc + i;
        let z = mix[off] * post_scale + base[off];
        out[off] = 2.0 / (1.0 + (-z).exp());
    }

    // comb stream: row softmax then iterated Sinkhorn normalization.
    let mut c = [0.0f32; SINKHORN_MAX_HC * SINKHORN_MAX_HC];

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

    // First half-step: column (src-major) normalization.
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

    // Remaining iterations: alternate dst / src normalization.
    for _ in 1..iters {
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

    for i in 0..n_hc * n_hc {
        out[2 * n_hc + i] = c[i];
    }
}

/// Convenience wrapper that runs the Sinkhorn split with the model-default
/// `N_HC` / `N_HC_SINKHORN_ITER` / `HC_EPS`.
///
/// The `out` / `mix` / `base` buffers must be sized for `N_HC`, i.e.
/// `(2 + N_HC) * N_HC` floats.
pub fn hc_split_sinkhorn_default(
    out: &mut [f32],
    mix: &[f32],
    scale: &[f32; 3],
    base: &[f32],
) {
    hc_split_sinkhorn_one(
        out,
        mix,
        scale,
        base,
        N_HC as usize,
        N_HC_SINKHORN_ITER as usize,
        HC_EPS,
    );
}

/// Descending top-k indices of `score`. Mirrors `topk_desc` in `ds4.c`
/// (~line 5211). The implementation is the C insertion-sort variant so ties
/// are broken in input order (earlier index wins), matching the C code
/// bit-for-bit on the integer index output.
pub fn topk_desc(score: &[f32], k: usize, idx: &mut [i32]) {
    assert_eq!(idx.len(), k);
    for v in idx.iter_mut() {
        *v = -1;
    }
    let n = score.len();
    for i in 0..n {
        for j in 0..k {
            let beats = if idx[j] < 0 {
                true
            } else {
                score[i] > score[idx[j] as usize]
            };
            if beats {
                let mut m = k - 1;
                while m > j {
                    idx[m] = idx[m - 1];
                    m -= 1;
                }
                idx[j] = i as i32;
                break;
            }
        }
    }
}

/// Apply DeepSeek's router head: project the residual through the gate-input
/// matrix and squash with `sqrt(softplus(.))`.
///
/// Mirrors `layer_router_probs_one` in `ds4.c` (~line 5169). `gate_inp_w` is
/// the row-major f16 router weight, packed as `u16` half floats (same
/// convention used by [`matvec_f16`]).
pub fn router_probs_one(probs: &mut [f32], gate_inp_w: &[u16], x: &[f32]) {
    assert_eq!(probs.len(), N_EXPERT as usize);
    matvec_f16(probs, gate_inp_w, x);
    for p in probs.iter_mut() {
        *p = softplus_stable(*p).sqrt();
    }
}

/// Pick `N_EXPERT_USED` experts and their normalized routing weights from a
/// raw `probs` vector and an optional bias term used only for selection.
///
/// Mirrors `layer_topk_selected_experts_from_probs` (~line 5246). The
/// expert weights are renormalized to sum to `EXPERT_WEIGHT_SCALE`, with the
/// same floor (`6.103515625e-5`) as the C source.
pub fn topk_selected_experts_from_probs(
    selected: &mut [i32; N_EXPERT_USED as usize],
    expert_weight: &mut [f32; N_EXPERT_USED as usize],
    probs: &[f32],
    exp_probs_bias: Option<&[f32]>,
) {
    assert_eq!(probs.len(), N_EXPERT as usize);

    let mut selection = [0.0f32; N_EXPERT as usize];
    selection.copy_from_slice(probs);

    if let Some(bias) = exp_probs_bias {
        assert_eq!(bias.len(), N_EXPERT as usize);
        for i in 0..N_EXPERT as usize {
            selection[i] += bias[i];
        }
    }

    topk_desc(&selection, N_EXPERT_USED as usize, &mut selected[..]);

    let mut sum = 0.0f32;
    for i in 0..N_EXPERT_USED as usize {
        let e = selected[i];
        assert!(
            e >= 0 && (e as u32) < N_EXPERT,
            "top-k produced an out-of-range expert id"
        );
        expert_weight[i] = probs[e as usize];
        sum += expert_weight[i];
    }
    // Match the C floor (~ smallest positive normal f16).
    if sum < 6.103_515_625e-5 {
        sum = 6.103_515_625e-5;
    }
    for w in expert_weight.iter_mut() {
        *w = *w / sum * EXPERT_WEIGHT_SCALE;
    }
}

/// Convenience wrapper that runs the router and top-k together. Mirrors
/// `layer_topk_selected_experts` (~line 5234).
pub fn topk_selected_experts(
    selected: &mut [i32; N_EXPERT_USED as usize],
    expert_weight: &mut [f32; N_EXPERT_USED as usize],
    gate_inp_w: &[u16],
    exp_probs_bias: Option<&[f32]>,
    x: &[f32],
) {
    let mut probs = [0.0f32; N_EXPERT as usize];
    router_probs_one(&mut probs, gate_inp_w, x);
    topk_selected_experts_from_probs(selected, expert_weight, &probs, exp_probs_bias);
}

/// Run one expert's gate / up / down projection for a single token.
///
/// Mirrors the inner per-expert body of `layer_routed_moe_one` (the `trace`
/// branch in `ds4.c` around line 5327): IQ2_XXS gate/up paired matvec
/// (replaced here by Q8_0 matvecs), SwiGLU clamp + router weight, then Q2_K
/// down (also replaced by Q8_0).
///
/// Buffers:
///
/// * `out`        — `N_EMBD` floats; the expert's contribution is **added**
///                   on top of whatever's already there, matching the C
///                   accumulation in `out[j] += down[j]`.
/// * `gate_w`     — Q8_0 packed bytes for this expert's gate matrix
///                   (`[N_EMBD, N_FF_EXP]` row major).
/// * `up_w`       — Q8_0 packed bytes for this expert's up matrix.
/// * `down_w`     — Q8_0 packed bytes for this expert's down matrix
///                   (`[N_FF_EXP, N_EMBD]` row major).
/// * `x`          — `N_EMBD` input residual after pre-norm.
/// * `clamp`      — SwiGLU clamp (use `crate::shape::SWIGLU_CLAMP_EXP` for the
///                   model default); values `<= 1e-6` disable clamping.
/// * `expert_weight` — router-side weight applied to the SwiGLU activation
///                   before the down projection.
/// * `scratch`    — reusable scratch with three slots of `N_FF_EXP` and one
///                   slot of `N_EMBD`. See [`ExpertScratch`].
pub fn expert_swiglu_one(
    out: &mut [f32],
    gate_w: &[u8],
    up_w: &[u8],
    down_w: &[u8],
    x: &[f32],
    clamp: f32,
    expert_weight: f32,
    scratch: &mut ExpertScratch,
) {
    assert_eq!(out.len(), N_EMBD as usize);
    assert_eq!(x.len(), N_EMBD as usize);
    assert_eq!(scratch.gate.len(), N_FF_EXP as usize);
    assert_eq!(scratch.up.len(), N_FF_EXP as usize);
    assert_eq!(scratch.mid.len(), N_FF_EXP as usize);
    assert_eq!(scratch.down.len(), N_EMBD as usize);

    matvec_q8_0(&mut scratch.gate, gate_w, x);
    matvec_q8_0(&mut scratch.up, up_w, x);

    let limit_active = clamp > 1.0e-6;
    for j in 0..N_FF_EXP as usize {
        let mut g = scratch.gate[j];
        let mut u = scratch.up[j];
        if limit_active {
            if g > clamp {
                g = clamp;
            }
            if u > clamp {
                u = clamp;
            }
            if u < -clamp {
                u = -clamp;
            }
        }
        scratch.mid[j] = silu(g) * u * expert_weight;
    }

    matvec_q8_0(&mut scratch.down, down_w, &scratch.mid);
    for j in 0..N_EMBD as usize {
        out[j] += scratch.down[j];
    }
}

/// Reusable per-token scratch for [`expert_swiglu_one`] and
/// [`layer_routed_moe_one`]. The C code mallocs these per call when `trace`
/// is on; we let callers reuse them across tokens or layers.
pub struct ExpertScratch {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub mid: Vec<f32>,
    pub down: Vec<f32>,
}

impl ExpertScratch {
    /// Allocate scratch sized for the model defaults (`N_FF_EXP` / `N_EMBD`).
    pub fn new() -> Self {
        Self {
            gate: vec![0.0; N_FF_EXP as usize],
            up: vec![0.0; N_FF_EXP as usize],
            mid: vec![0.0; N_FF_EXP as usize],
            down: vec![0.0; N_EMBD as usize],
        }
    }
}

impl Default for ExpertScratch {
    fn default() -> Self { Self::new() }
}

/// Bundle of per-layer weights consumed by [`layer_routed_moe_one`].
///
/// The C `ds4_layer_weights` carries every tensor a layer needs; the CPU
/// reference here only needs the routing head, the optional selection-time
/// bias, and the per-expert gate/up/down blocks. Per-expert slices are
/// indexed `[expert]` and contain that expert's whole weight block.
pub struct RoutedMoeWeights<'a> {
    /// f16 router head (`[N_EMBD, N_EXPERT]`, row major), packed as `u16`.
    pub ffn_gate_inp: &'a [u16],
    /// Optional bias added only for selection, not for weighting.
    pub ffn_exp_probs_b: Option<&'a [f32]>,
    /// Per-expert Q8_0 gate weights.
    pub ffn_gate_exps: &'a [&'a [u8]],
    /// Per-expert Q8_0 up weights.
    pub ffn_up_exps: &'a [&'a [u8]],
    /// Per-expert Q8_0 down weights.
    pub ffn_down_exps: &'a [&'a [u8]],
}

/// Single-token routed MoE for one layer.
///
/// Mirrors `layer_routed_moe_one` in `ds4.c` (~line 5278). The C version
/// picks between hash routing (when `ffn_gate_tid2eid` is non-null) and
/// top-k routing; the CPU reference here implements only the top-k path,
/// which is the one used by the default DeepSeek V4 Flash GGUF.
///
/// The output is written, **not** accumulated — the C function `memset`s
/// `out` to zero before summing the per-expert down projections.
pub fn layer_routed_moe_one(
    out: &mut [f32],
    weights: &RoutedMoeWeights<'_>,
    x: &[f32],
    clamp: f32,
    scratch: &mut ExpertScratch,
) {
    assert_eq!(out.len(), N_EMBD as usize);
    assert_eq!(x.len(), N_EMBD as usize);
    let n_used = N_EXPERT_USED as usize;
    let n_exp = N_EXPERT as usize;
    assert_eq!(weights.ffn_gate_exps.len(), n_exp);
    assert_eq!(weights.ffn_up_exps.len(), n_exp);
    assert_eq!(weights.ffn_down_exps.len(), n_exp);

    for v in out.iter_mut() {
        *v = 0.0;
    }

    let mut selected = [-1i32; N_EXPERT_USED as usize];
    let mut expert_weight = [0.0f32; N_EXPERT_USED as usize];
    topk_selected_experts(
        &mut selected,
        &mut expert_weight,
        weights.ffn_gate_inp,
        weights.ffn_exp_probs_b,
        x,
    );

    for i in 0..n_used {
        let e = selected[i] as usize;
        expert_swiglu_one(
            out,
            weights.ffn_gate_exps[e],
            weights.ffn_up_exps[e],
            weights.ffn_down_exps[e],
            x,
            clamp,
            expert_weight[i],
            scratch,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sinkhorn comb output should be (approximately) doubly stochastic:
    /// each row sums to 1 and each column sums to 1, modulo the small `eps`
    /// floor that the C code adds.
    #[test]
    fn sinkhorn_converges_to_doubly_stochastic() {
        let n_hc = 4usize;
        let dim = (2 + n_hc) * n_hc;
        // Mix has some arbitrary but deterministic pattern.
        let mut mix = vec![0.0f32; dim];
        for i in 0..dim {
            // mild values so exp() stays in range.
            mix[i] = ((i as f32) * 0.137).sin();
        }
        let base = vec![0.0f32; dim];
        let scale = [1.0f32, 1.0, 1.0];
        let mut out = vec![0.0f32; dim];
        hc_split_sinkhorn_one(&mut out, &mix, &scale, &base, n_hc, 20, 1.0e-6);

        let comb = &out[2 * n_hc..];

        // Row sums (over src for each dst): close to 1.
        for dst in 0..n_hc {
            let mut s = 0.0f32;
            for src in 0..n_hc {
                s += comb[src + dst * n_hc];
            }
            assert!((s - 1.0).abs() < 1e-3, "row {dst} sum drifted: {s}");
        }
        // Column sums (over dst for each src): close to 1.
        for src in 0..n_hc {
            let mut s = 0.0f32;
            for dst in 0..n_hc {
                s += comb[src + dst * n_hc];
            }
            assert!((s - 1.0).abs() < 1e-3, "col {src} sum drifted: {s}");
        }
    }

    /// One Sinkhorn pass on an all-zero mix + zero base must produce the
    /// uniform distribution exactly.
    #[test]
    fn sinkhorn_uniform_on_zero_input() {
        let n_hc = 4usize;
        let dim = (2 + n_hc) * n_hc;
        let mix = vec![0.0f32; dim];
        let base = vec![0.0f32; dim];
        let scale = [1.0f32, 1.0, 1.0];
        let mut out = vec![0.0f32; dim];
        hc_split_sinkhorn_one(&mut out, &mix, &scale, &base, n_hc, 20, 1.0e-6);

        // Pre stream: sigmoid(0) + eps = 0.5 + 1e-6.
        for i in 0..n_hc {
            assert!((out[i] - (0.5 + 1.0e-6)).abs() < 1e-5);
        }
        // Post stream: 2 * sigmoid(0) = 1.0.
        for i in 0..n_hc {
            assert!((out[n_hc + i] - 1.0).abs() < 1e-5);
        }
        // Comb stream: uniform 1/n_hc on every entry.
        let expected = 1.0 / n_hc as f32;
        for i in 0..n_hc * n_hc {
            assert!(
                (out[2 * n_hc + i] - expected).abs() < 1e-4,
                "comb[{i}] = {} not uniform",
                out[2 * n_hc + i]
            );
        }
    }

    /// `topk_desc` should pick the indices with the largest scores and
    /// order them by descending score, breaking ties in input order.
    #[test]
    fn topk_picks_largest_scores() {
        let scores = [0.1f32, 5.0, -2.0, 4.5, 4.5, 9.0, 0.0, 3.0];
        let mut idx = [-1i32; 4];
        topk_desc(&scores, 4, &mut idx);
        // Sorted desc: 9.0 @5, 5.0 @1, 4.5 @3 (first 4.5), 4.5 @4.
        assert_eq!(idx, [5, 1, 3, 4]);

        // Sanity: the chosen scores are non-increasing.
        for w in idx.windows(2) {
            let a = scores[w[0] as usize];
            let b = scores[w[1] as usize];
            assert!(a >= b);
        }
    }

    /// `topk_selected_experts_from_probs` should pick the top-k probability
    /// indices and yield router weights summing to `EXPERT_WEIGHT_SCALE`.
    #[test]
    fn topk_selected_normalizes_to_expert_weight_scale() {
        let mut probs = [0.0f32; N_EXPERT as usize];
        // Plant a handful of distinctive peaks.
        let peaks: [(usize, f32); 6] = [
            (3,   0.9),
            (17,  0.8),
            (42,  0.7),
            (100, 0.6),
            (200, 0.5),
            (250, 0.4),
        ];
        for &(i, v) in &peaks {
            probs[i] = v;
        }

        let mut selected = [-1i32; N_EXPERT_USED as usize];
        let mut weights = [0.0f32; N_EXPERT_USED as usize];
        topk_selected_experts_from_probs(&mut selected, &mut weights, &probs, None);

        let mut sel_sorted = selected;
        sel_sorted.sort();
        // The six planted peaks are exactly the selected experts.
        let mut expected: Vec<i32> = peaks.iter().map(|&(i, _)| i as i32).collect();
        expected.sort();
        assert_eq!(sel_sorted.to_vec(), expected);

        // Weights renormalized so the sum matches EXPERT_WEIGHT_SCALE.
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - EXPERT_WEIGHT_SCALE).abs() < 1e-4,
            "weights sum {sum} != {EXPERT_WEIGHT_SCALE}"
        );
        for &w in &weights {
            assert!(w > 0.0);
        }
    }
}
