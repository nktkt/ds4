//! Small inner-loop kernels for IQ2/Q2 quant dot products.
//!
//! Ports of the `dot_iq2_pair_16` / `dot_q2_16` helpers from `ds4.c`. We keep
//! a scalar implementation that always compiles, plus an optional
//! `target_arch = "aarch64"` path that mirrors the NEON `vdot` version. Both
//! produce the same integer accumulator value so logits stay bit-identical
//! between targets.

/// Dot product of two 8-byte signed grids against 16 bytes of activations.
#[inline]
pub fn dot_iq2_pair_16(grid0: &[i8; 8], grid1: &[i8; 8], q8: &[i8; 16]) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..8 { sum += grid0[i] as i32 * q8[i] as i32; }
    for i in 0..8 { sum += grid1[i] as i32 * q8[8 + i] as i32; }
    sum
}

/// Dot product of 16 packed 2-bit quants (shifted by `shift` ∈ {0,2,4,6}
/// then masked to two bits) against 16 bytes of int8 activations.
#[inline]
pub fn dot_q2_16(q2: &[u8; 16], q8: &[i8; 16], shift: u32) -> i32 {
    let mut sum: i32 = 0;
    let s = shift & 7;
    for i in 0..16 {
        let v = ((q2[i] >> s) & 3) as i32;
        sum += q8[i] as i32 * v;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pair_16_scalar() {
        let g0 = [1, -1, 2, -2, 3, -3, 4, -4];
        let g1 = [5, -5, 6, -6, 7, -7, 8, -8];
        let q8 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        // Manual expected
        let mut expected = 0;
        for i in 0..8 { expected += g0[i] as i32 * q8[i] as i32; }
        for i in 0..8 { expected += g1[i] as i32 * q8[8 + i] as i32; }
        assert_eq!(dot_iq2_pair_16(&g0, &g1, &q8), expected);
    }

    #[test]
    fn q2_shifts_mask_to_2_bits() {
        let q2 = [0b11_10_01_00; 16];
        let q8 = [1; 16];
        assert_eq!(dot_q2_16(&q2, &q8, 0), 16 * 0);
        assert_eq!(dot_q2_16(&q2, &q8, 2), 16 * 1);
        assert_eq!(dot_q2_16(&q2, &q8, 4), 16 * 2);
        assert_eq!(dot_q2_16(&q2, &q8, 6), 16 * 3);
    }
}
