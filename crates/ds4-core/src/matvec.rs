//! Matrix-vector multiplies for DS4 weight types.
//!
//! Ports of `matvec_f16`, `matvec_f32`, and the q8_0 activation quantization
//! helper. The quantized variants for Q2_K / IQ2_XXS get their own module
//! ([`crate::dequant`]) because they need the per-block scale handling.

use crate::half::f16_to_f32;

/// Dot product of one f16 weight row against an f32 activation vector.
/// Mirrors `dot_f16_row` from `ds4.c` (scalar path).
#[inline]
pub fn dot_f16_row(row: &[u16], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len(), x.len());
    let mut acc = 0.0f32;
    for i in 0..row.len() {
        acc += f16_to_f32(row[i]) * x[i];
    }
    acc
}

/// `out = W @ x` for f16 weights `W` laid out row-major `[out_dim, in_dim]`.
pub fn matvec_f16(out: &mut [f32], w: &[u16], x: &[f32]) {
    let out_dim = out.len();
    let in_dim = x.len();
    assert_eq!(w.len(), out_dim * in_dim);
    for r in 0..out_dim {
        out[r] = dot_f16_row(&w[r * in_dim..(r + 1) * in_dim], x);
    }
}

/// f32 weights variant. Mirrors `matvec_f32`.
pub fn matvec_f32(out: &mut [f32], w: &[f32], x: &[f32]) {
    let out_dim = out.len();
    let in_dim = x.len();
    assert_eq!(w.len(), out_dim * in_dim);
    for r in 0..out_dim {
        let row = &w[r * in_dim..(r + 1) * in_dim];
        let mut acc = 0.0f32;
        for i in 0..in_dim { acc += row[i] * x[i]; }
        out[r] = acc;
    }
}

/// Quantize an activation vector to Q8_0: 32-element blocks, each holding an
/// f32 scale and 32 int8 values. Mirrors `quantize_q8_0_activation`.
pub fn quantize_q8_0_activation(x: &[f32], xq: &mut [i8], scale: &mut [f32]) {
    let n = x.len();
    assert!(n % 32 == 0, "Q8_0 expects n divisible by 32");
    let blocks = n / 32;
    assert_eq!(xq.len(), n);
    assert_eq!(scale.len(), blocks);
    for b in 0..blocks {
        let start = b * 32;
        let block = &x[start..start + 32];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        scale[b] = d;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for i in 0..32 {
            let q = (block[i] * id).round();
            xq[start + i] = q.clamp(-127.0, 127.0) as i8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;

    #[test]
    fn f16_matvec_identity() {
        // 3x3 identity in f16
        let mut w = vec![0u16; 9];
        for i in 0..3 { w[i * 3 + i] = f32_to_f16(1.0); }
        let x = vec![1.0_f32, 2.0, 3.0];
        let mut out = vec![0.0_f32; 3];
        matvec_f16(&mut out, &w, &x);
        for i in 0..3 { assert!((out[i] - x[i]).abs() < 1e-3); }
    }

    #[test]
    fn q8_0_quantize_roundtrip() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let mut xq = vec![0i8; 32];
        let mut scale = vec![0.0_f32; 1];
        quantize_q8_0_activation(&x, &mut xq, &mut scale);
        // Reconstruct
        for i in 0..32 {
            let r = xq[i] as f32 * scale[0];
            assert!((r - x[i]).abs() < scale[0]); // within one quantum
        }
    }
}
