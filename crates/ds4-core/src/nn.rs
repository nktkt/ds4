//! Neural-network primitives shared between CPU reference and verification
//! paths. Ports of small, well-isolated helpers from `ds4.c`:
//!
//! * `sigmoid_stable`, `silu`, `softplus_stable`
//! * `swiglu`
//! * `rms_norm_no_weight`, `rms_norm_weight`
//! * `head_rms_norm_inplace`
//! * `softmax_inplace`
//!
//! These all run on f32 slices and use f64 accumulators where the C version
//! does, so numeric outputs match bit-for-bit modulo platform float ordering.

/// Numerically stable sigmoid. Mirrors `sigmoid_stable` (see ds4.c around the
/// SwiGLU section).
#[inline]
pub fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[inline]
pub fn silu(x: f32) -> f32 { x * sigmoid_stable(x) }

/// `log(1+exp(x))` with the same branchpoints as the C version
/// (`softplus_stable`).
#[inline]
pub fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 { x }
    else if x < -20.0 { x.exp() }
    else { (-(-x).abs()).exp().ln_1p() + x.max(0.0) }
}

/// In-place SwiGLU. `out[i] = silu(gate[i]) * up[i]`.
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    assert_eq!(out.len(), gate.len());
    assert_eq!(out.len(), up.len());
    for i in 0..out.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// RMSNorm without learned scale. Mirrors `rms_norm_no_weight`. Uses f64
/// accumulators to match the C code.
pub fn rms_norm_no_weight(out: &mut [f32], x: &[f32], eps: f32) {
    assert_eq!(out.len(), x.len());
    let n = x.len() as f64;
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let scale = 1.0 / ((ss / n + eps as f64).sqrt() as f32);
    for i in 0..out.len() {
        out[i] = x[i] * scale;
    }
}

/// RMSNorm with learned per-channel scale.
pub fn rms_norm_weight(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    assert_eq!(out.len(), x.len());
    assert_eq!(out.len(), weight.len());
    let n = x.len() as f64;
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let scale = 1.0 / ((ss / n + eps as f64).sqrt() as f32);
    for i in 0..out.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Per-head RMSNorm: normalize each `head_dim`-length chunk of `x`
/// independently. Mirrors `head_rms_norm_inplace`.
pub fn head_rms_norm_inplace(x: &mut [f32], n_head: u32, head_dim: u32, eps: f32) {
    let head_dim = head_dim as usize;
    let n = head_dim as f64;
    for h in 0..n_head as usize {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        let ss: f64 = head.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let scale = 1.0 / ((ss / n + eps as f64).sqrt() as f32);
        for v in head.iter_mut() { *v *= scale; }
    }
}

/// In-place numerically stable softmax (subtract max first).
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() { *v /= sum; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silu_zero_is_zero() { assert_eq!(silu(0.0), 0.0); }
    #[test]
    fn rms_norm_matches_formula() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        rms_norm_no_weight(&mut out, &x, 1e-6);
        let n = x.len() as f32;
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let scale = 1.0 / ((ss / n + 1e-6).sqrt());
        for i in 0..4 {
            assert!((out[i] - x[i] * scale).abs() < 1e-5);
        }
    }
    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0_f32, 2.0, 3.0];
        softmax_inplace(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }
    #[test]
    fn swiglu_smoke() {
        let g = [0.0_f32, 1.0, 2.0, -1.0];
        let u = [1.0_f32; 4];
        let mut o = [0.0_f32; 4];
        swiglu(&mut o, &g, &u);
        // silu(0) = 0, silu(>0) > 0, silu(<0) < 0 (but small magnitude)
        assert_eq!(o[0], 0.0);
        assert!(o[1] > 0.0);
        assert!(o[2] > o[1]);
        assert!(o[3] < 0.0);
    }
}
