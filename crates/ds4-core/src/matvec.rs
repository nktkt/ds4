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

/// Block layout for a `Q8_0` *weight* block: 2 bytes of f16 scale + 32 bytes of
/// signed int8 quants. Mirrors the `[f16 d][i8 qs * 32]` row striding used
/// inside `dot_q8_0_row`.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Plain int8 dot product over up to 32 lanes. Mirrors `dot_i8_32`.
#[inline]
pub fn dot_i8_n(qs: &[i8], xq: &[i8], n: usize) -> i32 {
    let mut acc = 0i32;
    for i in 0..n { acc += qs[i] as i32 * xq[i] as i32; }
    acc
}

/// Row dot product against a Q8_0 weight row laid out as
/// `[[f16 d_b][i8 q_b * 32]] * blocks`. Returns the f32 dot product.
/// Ports the scalar fallback of `dot_q8_0_row` from `ds4.c`.
pub fn dot_q8_0_row(row: &[u8], xq: &[i8], xscale: &[f32], in_dim: usize) -> f32 {
    let blocks = (in_dim + 31) / 32;
    debug_assert!(row.len() >= blocks * Q8_0_BLOCK_BYTES);
    let mut acc = 0.0f32;
    for b in 0..blocks {
        let base = b * Q8_0_BLOCK_BYTES;
        let scale_bits = u16::from_le_bytes([row[base], row[base + 1]]);
        let qs_start = base + 2;
        let qs = unsafe {
            // SAFETY: u8 and i8 have the same layout; the slice was bounds-checked above.
            std::slice::from_raw_parts(row[qs_start..].as_ptr() as *const i8, 32)
        };
        let i0 = b * 32;
        let n = if in_dim - i0 < 32 { in_dim - i0 } else { 32 };
        acc += crate::half::f16_to_f32(scale_bits) * xscale[b] * dot_i8_n(qs, &xq[i0..i0 + n], n) as f32;
    }
    acc
}

/// `out = W @ x` for `Q8_0` weights with row stride `Q8_0_BLOCK_BYTES * blocks`.
/// Quantizes the activation in one pass (matches the C path's
/// `quantize_q8_0_activation_batch` + `matvec_q8_0_worker` flow).
pub fn matvec_q8_0(out: &mut [f32], w: &[u8], x: &[f32]) {
    let in_dim = x.len();
    let blocks = (in_dim + 31) / 32;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;
    assert_eq!(w.len(), out.len() * row_bytes);
    let mut xq = vec![0i8; blocks * 32];
    let mut xscale = vec![0.0_f32; blocks];
    let mut tail = vec![0.0_f32; blocks * 32 - in_dim];
    let _ = &mut tail;
    quantize_q8_0_activation(
        &{ let mut v = x.to_vec(); v.resize(blocks * 32, 0.0); v },
        &mut xq, &mut xscale,
    );
    for r in 0..out.len() {
        out[r] = dot_q8_0_row(&w[r * row_bytes..(r + 1) * row_bytes], &xq, &xscale, in_dim);
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
    fn dot_q8_0_row_identity_like() {
        // Build a single Q8_0 block with scale 1.0 and qs = [1, 1, ..., 1].
        // Then dot it against an int8 activation of all 1s (also scale 1.0).
        // Result should be 32.
        let mut row = vec![0u8; Q8_0_BLOCK_BYTES];
        let scale = crate::half::f32_to_f16(1.0).to_le_bytes();
        row[0] = scale[0];
        row[1] = scale[1];
        for i in 0..32 { row[2 + i] = 1u8; } // i8 = 1
        let xq = vec![1i8; 32];
        let xscale = vec![1.0_f32; 1];
        let v = dot_q8_0_row(&row, &xq, &xscale, 32);
        assert!((v - 32.0).abs() < 1e-3, "got {v}");
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
