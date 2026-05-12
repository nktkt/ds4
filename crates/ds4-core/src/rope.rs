//! Rotary position embedding (RoPE) — YaRN variant used by DeepSeek V4 Flash.
//!
//! Ports `rope_yarn_*`, `rope_tail_ext_inplace`, `layer_rope_freq_base`, and
//! `layer_rope_freq_scale` from `ds4.c`. Only the host-side variant is here;
//! the GPU paths reuse the same numerical recipe inside their kernels.

use crate::shape::{
    COMPRESS_ROPE_FREQ_BASE, ROPE_FREQ_BASE, ROPE_ORIG_CTX, ROPE_SCALE_FACTOR,
    ROPE_YARN_BETA_FAST, ROPE_YARN_BETA_SLOW,
};

#[inline]
pub fn yarn_ramp(low: f32, high: f32, i0: i32) -> f32 {
    let v = (i0 as f32 - low) / (high - low).max(1e-3);
    v.clamp(0.0, 1.0)
}

#[inline]
pub fn yarn_corr_dim(n_dims: i32, n_ctx_orig: u64, n_rot: f32, base: f32) -> f32 {
    // C: log(n_ctx_orig / (n_rot * 2*pi)) / (2*log(base))
    let pi = std::f32::consts::PI;
    let num = (n_ctx_orig as f32 / (n_rot * 2.0 * pi)).ln();
    let den = 2.0 * base.ln();
    let _ = n_dims;
    num / den
}

pub fn yarn_corr_dims(n_dims: i32, n_ctx_orig: u64, freq_base: f32, beta_fast: f32, beta_slow: f32) -> [f32; 2] {
    let lo = yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base);
    let hi = yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base);
    [lo.floor(), hi.ceil()]
}

/// Per-layer RoPE base. For DS4 the early dense layers (`il < 2`) and the
/// non-compressed-attention layers use the standard base; compressed
/// attention layers use a different base. Mirrors `layer_rope_freq_base`.
pub fn layer_rope_freq_base(il: u32) -> f32 {
    use crate::layer;
    if layer::compress_ratio(il) == 0 { ROPE_FREQ_BASE } else { COMPRESS_ROPE_FREQ_BASE }
}

/// Per-layer RoPE scale. Set to 1/ROPE_SCALE_FACTOR for compressed layers.
pub fn layer_rope_freq_scale(il: u32) -> f32 {
    use crate::layer;
    if layer::compress_ratio(il) == 0 { 1.0 } else { 1.0 / ROPE_SCALE_FACTOR }
}

/// Apply YaRN-style RoPE rotation to the trailing `n_rot` dimensions of a
/// per-head Q or K tensor at position `pos`. `x` is laid out as
/// `[n_head, head_dim]`; rotations happen over pairs `(x[2k], x[2k+1])` of
/// the last `n_rot` lanes of each head.
pub fn tail_rotate_inplace(
    x: &mut [f32],
    n_head: u32,
    head_dim: u32,
    n_rot: u32,
    pos: u64,
    freq_base: f32,
    freq_scale: f32,
) {
    assert!(n_rot % 2 == 0, "RoPE n_rot must be even");
    let head_dim = head_dim as usize;
    let n_rot = n_rot as usize;
    let pos_f = pos as f32 * freq_scale;
    let extrap = yarn_corr_dims(n_rot as i32, ROPE_ORIG_CTX, freq_base, ROPE_YARN_BETA_FAST, ROPE_YARN_BETA_SLOW);
    for h in 0..n_head as usize {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        let base_off = head_dim - n_rot;
        for k in 0..n_rot / 2 {
            let i0 = (2 * k) as i32;
            let freq = 1.0 / freq_base.powf(2.0 * k as f32 / n_rot as f32);
            let ramp = yarn_ramp(extrap[0], extrap[1], i0);
            let theta = pos_f * freq * (1.0 - ramp);
            let (s, c) = theta.sin_cos();
            let a = head[base_off + 2 * k];
            let b = head[base_off + 2 * k + 1];
            head[base_off + 2 * k]     = a * c - b * s;
            head[base_off + 2 * k + 1] = a * s + b * c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pos_zero_is_identity() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let orig = x.clone();
        tail_rotate_inplace(&mut x, 1, 4, 4, 0, ROPE_FREQ_BASE, 1.0);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
