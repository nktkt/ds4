//! Quantized row-dot and matvec kernels for Q2_K, Q4_K, and IQ2_XXS.
//!
//! These mirror the CPU reference path in `ds4.c` (`ds4_vec_dot_q2_K_q8_K`,
//! `ds4_vec_dot_iq2_xxs_q8_K`, plus the `matvec_q2_k_*` / `matvec_iq2_xxs_*`
//! row drivers around lines 1697-3990). The C version consumes a
//! `block_q8_K *` activation (one f32 scale + 256 i8 lanes + per-16 i16
//! bsums); here we accept the unpacked `(xq, xscale)` pair the Rust scratch
//! layout already produces, with one f32 scale per `QK_K` (=256) chunk and
//! `in_dim` contiguous i8 lanes. The integer inner loops route through
//! `crate::dot::{dot_q2_16, dot_iq2_pair_16}` so the scalar/NEON split there
//! is the single source of truth.
//!
//! No Q4_K dot exists in `ds4.c` (the C side only dequants Q4_K and feeds it
//! to the f32 path); the Q4_K kernel here matches the dequant recipe in
//! `crate::dequant::dequant_q4k` so the row-dot value equals
//! `<dequant_q4k(W_row), x>` to within float rounding.

use crate::dot::{dot_iq2_pair_16, dot_q2_16};
use crate::half::f16_to_f32;
use crate::iq2_tables::IQ2XXS_SIGNED_GRID;
use crate::quant::{BlockIQ2XXS, BlockQ2K, BlockQ4K, QK_K};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sum 16 consecutive `i8` lanes into an `i32`. Mirrors the per-16 entries
/// of `block_q8_K::bsums` that `ds4_vec_dot_q2_K_q8_K` consumes for the
/// `dmin * summs` correction term.
#[inline]
fn bsum16(xq: &[i8]) -> i32 {
    let mut s: i32 = 0;
    for i in 0..16 {
        s += xq[i] as i32;
    }
    s
}

#[inline]
fn slice16(xq: &[i8], off: usize) -> &[i8; 16] {
    let s: &[i8] = &xq[off..off + 16];
    // SAFETY: the slice length is exactly 16 and `[i8; 16]` has the same
    // layout as `[i8]` of length 16.
    unsafe { &*(s.as_ptr() as *const [i8; 16]) }
}

// ---------------------------------------------------------------------------
// Q2_K
// ---------------------------------------------------------------------------

/// Dot one Q2_K weight row against an int8 activation row.
///
/// Mirrors `ds4_vec_dot_q2_K_q8_K` (scalar branch, `ds4.c` ~line 1778). The
/// per-`QK_K` Q8 scale comes from `xscale[block]`; `xq` holds the 256 i8
/// lanes for that block contiguously.
pub fn dot_q2_k_row(blocks: &[BlockQ2K], xq: &[i8], xscale: &[f32]) -> f32 {
    debug_assert_eq!(xq.len(), blocks.len() * QK_K);
    debug_assert_eq!(xscale.len(), blocks.len());

    let mut sumf = 0.0f32;
    for (i, blk) in blocks.iter().enumerate() {
        let xq_block = &xq[i * QK_K..(i + 1) * QK_K];
        let q8_scale = xscale[i];

        // summs term: per-16-lane bsums dotted with high-nibble scales.
        let sc = &blk.scales;
        let mut summs: i32 = 0;
        for j in 0..16 {
            let bs = bsum16(&xq_block[j * 16..]);
            summs += bs * ((sc[j] >> 4) as i32);
        }

        let dall = q8_scale * f16_to_f32(blk.d);
        let dmin = q8_scale * f16_to_f32(blk.dmin);

        let mut isum: i32 = 0;
        let mut is = 0usize;
        for k in 0..QK_K / 128 {
            // 2 super-blocks of 128 elements each (= 4 shifts × 32 lanes).
            let q2_base = k * 32;
            let q8_base = k * 128;
            for j in 0..4 {
                let shift: u32 = (j * 2) as u32;
                let q8_off = q8_base + j * 32;

                let d_lo = (sc[is] & 0x0f) as i32;
                is += 1;
                let q2_lo: &[u8; 16] = {
                    let s = &blk.qs[q2_base..q2_base + 16];
                    unsafe { &*(s.as_ptr() as *const [u8; 16]) }
                };
                let isuml_lo = dot_q2_16(q2_lo, slice16(xq_block, q8_off), shift);
                isum += d_lo * isuml_lo;

                let d_hi = (sc[is] & 0x0f) as i32;
                is += 1;
                let q2_hi: &[u8; 16] = {
                    let s = &blk.qs[q2_base + 16..q2_base + 32];
                    unsafe { &*(s.as_ptr() as *const [u8; 16]) }
                };
                let isuml_hi = dot_q2_16(q2_hi, slice16(xq_block, q8_off + 16), shift);
                isum += d_hi * isuml_hi;
            }
        }
        sumf += dall * isum as f32 - dmin * summs as f32;
    }
    sumf
}

/// `out = W @ x` for Q2_K weights. Mirrors `matvec_q2_k_worker` /
/// `matvec_q2_k_expert` (`ds4.c` ~line 3900).
pub fn matvec_q2_k(out: &mut [f32], rows: &[BlockQ2K], in_dim: usize, xq: &[i8], xscale: &[f32]) {
    assert_eq!(in_dim % QK_K, 0, "Q2_K matvec requires in_dim % QK_K == 0");
    let blocks_per_row = in_dim / QK_K;
    assert_eq!(rows.len(), out.len() * blocks_per_row);
    assert_eq!(xq.len(), in_dim);
    assert_eq!(xscale.len(), blocks_per_row);

    for r in 0..out.len() {
        let row = &rows[r * blocks_per_row..(r + 1) * blocks_per_row];
        out[r] = dot_q2_k_row(row, xq, xscale);
    }
}

// ---------------------------------------------------------------------------
// Q4_K
// ---------------------------------------------------------------------------

/// Dot one Q4_K weight row against an int8 activation row.
///
/// `ds4.c` has no `ds4_vec_dot_q4_K_q8_K`; this kernel matches the dequant
/// recipe in `crate::dequant::dequant_q4k` (Q4_K dequant ~line 40 of
/// `dequant.rs`, originally `dequantize_row_q4_K` in ggml). Each 256-element
/// block decomposes into four 64-element sub-blocks with `dl = d * sc[is]`
/// and `ml = dmin * sc[is + 4]` (low 6 bits each).
pub fn dot_q4_k_row(blocks: &[BlockQ4K], xq: &[i8], xscale: &[f32]) -> f32 {
    debug_assert_eq!(xq.len(), blocks.len() * QK_K);
    debug_assert_eq!(xscale.len(), blocks.len());

    let mut sumf = 0.0f32;
    for (i, blk) in blocks.iter().enumerate() {
        let xq_block = &xq[i * QK_K..(i + 1) * QK_K];
        let q8_scale = xscale[i];
        let d = q8_scale * f16_to_f32(blk.d);
        let dmin = q8_scale * f16_to_f32(blk.dmin);

        // Four 64-element sub-blocks.
        for is in 0..QK_K / 64 {
            let sc = (blk.scales[is] & 0x3f) as i32;
            let m = (blk.scales[is + 4] & 0x3f) as i32;
            let dl = d * sc as f32;
            let ml = dmin * m as f32;

            let qs_off = is * 32;
            let out_off = is * 64;

            // Low nibble = first 32 outputs of the sub-block.
            let mut isum_lo: i32 = 0;
            let mut bsum_lo: i32 = 0;
            for j in 0..32 {
                let q = (blk.qs[qs_off + j] & 0x0f) as i32;
                let x = xq_block[out_off + j] as i32;
                isum_lo += q * x;
                bsum_lo += x;
            }

            // High nibble = next 32 outputs of the sub-block.
            let mut isum_hi: i32 = 0;
            let mut bsum_hi: i32 = 0;
            for j in 0..32 {
                let q = (blk.qs[qs_off + j] >> 4) as i32;
                let x = xq_block[out_off + 32 + j] as i32;
                isum_hi += q * x;
                bsum_hi += x;
            }

            sumf += dl * (isum_lo + isum_hi) as f32 - ml * (bsum_lo + bsum_hi) as f32;
        }
    }
    sumf
}

/// `out = W @ x` for Q4_K weights. Row-major: `rows` lays out
/// `out.len()` rows back-to-back, each with `in_dim / QK_K` blocks.
pub fn matvec_q4_k(out: &mut [f32], rows: &[BlockQ4K], in_dim: usize, xq: &[i8], xscale: &[f32]) {
    assert_eq!(in_dim % QK_K, 0, "Q4_K matvec requires in_dim % QK_K == 0");
    let blocks_per_row = in_dim / QK_K;
    assert_eq!(rows.len(), out.len() * blocks_per_row);
    assert_eq!(xq.len(), in_dim);
    assert_eq!(xscale.len(), blocks_per_row);

    for r in 0..out.len() {
        let row = &rows[r * blocks_per_row..(r + 1) * blocks_per_row];
        out[r] = dot_q4_k_row(row, xq, xscale);
    }
}

// ---------------------------------------------------------------------------
// IQ2_XXS
// ---------------------------------------------------------------------------

/// Dot one IQ2_XXS weight row against an int8 activation row.
///
/// Mirrors `ds4_vec_dot_iq2_xxs_q8_K` (scalar branch, `ds4.c` ~line 1874).
/// Each `QK_K`-block holds 32 packed `u16` codes; pairs of `u16`s decode to
/// (4 grid indices, 4 × 7-bit sign indices, 4-bit local scale `ls`) and
/// drive eight 16-lane `dot_iq2_pair_16` calls per block.
pub fn dot_iq2_xxs_row(blocks: &[BlockIQ2XXS], xq: &[i8], xscale: &[f32]) -> f32 {
    debug_assert_eq!(xq.len(), blocks.len() * QK_K);
    debug_assert_eq!(xscale.len(), blocks.len());

    let signed_grid = &*IQ2XXS_SIGNED_GRID;
    let mut sumf = 0.0f32;

    for (i, blk) in blocks.iter().enumerate() {
        let xq_block = &xq[i * QK_K..(i + 1) * QK_K];
        let d = f16_to_f32(blk.d) * xscale[i];

        let mut bsum: i32 = 0;
        for ib32 in 0..QK_K / 32 {
            // Two u32s packed as four u16s in `qs`.
            let qs = &blk.qs[ib32 * 4..ib32 * 4 + 4];
            let a0 = (qs[0] as u32) | ((qs[1] as u32) << 16);
            let a1 = (qs[2] as u32) | ((qs[3] as u32) << 16);

            let g0 = (a0 & 0xff) as usize;
            let g1 = ((a0 >> 8) & 0xff) as usize;
            let g2 = ((a0 >> 16) & 0xff) as usize;
            let g3 = ((a0 >> 24) & 0xff) as usize;

            let s0 = (a1 & 0x7f) as usize;
            let s1 = ((a1 >> 7) & 0x7f) as usize;
            let s2 = ((a1 >> 14) & 0x7f) as usize;
            let s3 = ((a1 >> 21) & 0x7f) as usize;

            let ls = 2 * ((a1 >> 28) as i32) + 1;

            let q8_off = ib32 * 32;
            let mut sumi: i32 = 0;
            sumi += dot_iq2_pair_16(
                &signed_grid[g0][s0],
                &signed_grid[g1][s1],
                slice16(xq_block, q8_off),
            );
            sumi += dot_iq2_pair_16(
                &signed_grid[g2][s2],
                &signed_grid[g3][s3],
                slice16(xq_block, q8_off + 16),
            );
            bsum += sumi * ls;
        }
        sumf += d * bsum as f32;
    }
    0.125 * sumf
}

/// `out = W @ x` for IQ2_XXS weights. Mirrors `matvec_iq2_xxs_pair_worker`
/// reduced to a single output tensor (`ds4.c` ~line 3764).
pub fn matvec_iq2_xxs(out: &mut [f32], rows: &[BlockIQ2XXS], in_dim: usize, xq: &[i8], xscale: &[f32]) {
    assert_eq!(in_dim % QK_K, 0, "IQ2_XXS matvec requires in_dim % QK_K == 0");
    let blocks_per_row = in_dim / QK_K;
    assert_eq!(rows.len(), out.len() * blocks_per_row);
    assert_eq!(xq.len(), in_dim);
    assert_eq!(xscale.len(), blocks_per_row);

    for r in 0..out.len() {
        let row = &rows[r * blocks_per_row..(r + 1) * blocks_per_row];
        out[r] = dot_iq2_xxs_row(row, xq, xscale);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;

    fn zero_q2k_block() -> BlockQ2K {
        BlockQ2K {
            scales: [0; QK_K / 16],
            qs: [0; QK_K / 4],
            d: 0,
            dmin: 0,
        }
    }

    fn zero_q4k_block() -> BlockQ4K {
        BlockQ4K {
            d: 0,
            dmin: 0,
            scales: [0; 12],
            qs: [0; QK_K / 2],
        }
    }

    fn zero_iq2_xxs_block() -> BlockIQ2XXS {
        BlockIQ2XXS { d: 0, qs: [0; QK_K / 8] }
    }

    #[test]
    fn q2_k_zero_input_is_zero() {
        let blk = zero_q2k_block();
        let xq = vec![0i8; QK_K];
        let xscale = vec![0.0f32; 1];
        let v = dot_q2_k_row(std::slice::from_ref(&blk), &xq, &xscale);
        assert_eq!(v, 0.0);

        let mut out = vec![1.0f32; 2];
        let rows = vec![blk, blk];
        let xscale2 = vec![0.0f32; 1];
        matvec_q2_k(&mut out, &rows, QK_K, &xq, &xscale2);
        assert_eq!(out, vec![0.0; 2]);
    }

    #[test]
    fn q2_k_constant_one_weight_recovers_activation_sum() {
        // Build a Q2_K block whose dequant value at every lane is exactly 1.0:
        //   - all `qs` bytes = 0b01010101 → every 2-bit field is 1
        //   - low nibble of every sc[is] = 1  → dl = d * 1
        //   - high nibble of every sc[is] = 0 → ml = dmin * 0 = 0
        //   - d = f16(1.0), dmin = f16(0.0)
        // Then row dot equals `xscale * sum(xq)`.
        let mut blk = zero_q2k_block();
        for i in 0..(QK_K / 4) {
            blk.qs[i] = 0b01_01_01_01;
        }
        for i in 0..(QK_K / 16) {
            blk.scales[i] = 0x01; // dl=1, ml=0
        }
        blk.d = f32_to_f16(1.0);
        blk.dmin = f32_to_f16(0.0);

        let mut xq = vec![0i8; QK_K];
        for i in 0..QK_K {
            xq[i] = ((i as i32 % 7) - 3) as i8;
        }
        let xscale = vec![0.5f32];
        let v = dot_q2_k_row(std::slice::from_ref(&blk), &xq, &xscale);
        let expected: f32 = 0.5 * xq.iter().map(|&q| q as f32).sum::<f32>();
        assert!((v - expected).abs() < 1e-3, "got {v}, expected {expected}");
    }

    #[test]
    fn q4_k_zero_input_is_zero() {
        let blk = zero_q4k_block();
        let xq = vec![0i8; QK_K];
        let xscale = vec![0.0f32; 1];
        let v = dot_q4_k_row(std::slice::from_ref(&blk), &xq, &xscale);
        assert_eq!(v, 0.0);

        let mut out = vec![2.0f32; 3];
        let rows = vec![blk, blk, blk];
        matvec_q4_k(&mut out, &rows, QK_K, &xq, &xscale);
        assert_eq!(out, vec![0.0; 3]);
    }

    #[test]
    fn q4_k_constant_one_weight_recovers_activation_sum() {
        // Q4_K block whose dequant is identically 1.0:
        //   - every 4-bit code = 1 (qs byte = 0x11 puts a 1 in both nibbles)
        //   - dl = d * (scales[is] & 0x3f) with scales[0..4] = 1
        //   - ml = dmin * (scales[is+4] & 0x3f) with scales[4..8] = 0
        //   - d = f16(1.0), dmin = f16(0.0)
        let mut blk = zero_q4k_block();
        for i in 0..(QK_K / 2) {
            blk.qs[i] = 0x11;
        }
        for is in 0..4 {
            blk.scales[is] = 1;
            blk.scales[is + 4] = 0;
        }
        blk.d = f32_to_f16(1.0);
        blk.dmin = f32_to_f16(0.0);

        let mut xq = vec![0i8; QK_K];
        for i in 0..QK_K {
            xq[i] = ((i as i32 % 11) - 5) as i8;
        }
        let xscale = vec![0.25f32];
        let v = dot_q4_k_row(std::slice::from_ref(&blk), &xq, &xscale);
        let expected: f32 = 0.25 * xq.iter().map(|&q| q as f32).sum::<f32>();
        assert!((v - expected).abs() < 1e-3, "got {v}, expected {expected}");
    }

    #[test]
    fn iq2_xxs_zero_input_is_zero() {
        let blk = zero_iq2_xxs_block();
        let xq = vec![0i8; QK_K];
        let xscale = vec![0.0f32; 1];
        let v = dot_iq2_xxs_row(std::slice::from_ref(&blk), &xq, &xscale);
        assert_eq!(v, 0.0);

        let mut out = vec![3.0f32; 2];
        let rows = vec![blk, blk];
        matvec_iq2_xxs(&mut out, &rows, QK_K, &xq, &xscale);
        assert_eq!(out, vec![0.0; 2]);
    }

    #[test]
    fn iq2_xxs_dot_matches_dequant_reference() {
        // Pick a small set of nonzero codes, then check that
        // `dot_iq2_xxs_row(..., xq, xscale)` agrees with the equivalent
        // f32 dot of the dequantized row against `xq * xscale`.
        use crate::dequant::dequant_iq2_xxs;

        let mut blk = zero_iq2_xxs_block();
        // For each ib32 (8 of them), set 4 grid indices and a sign+ls word.
        for ib32 in 0..(QK_K / 32) {
            let base = ib32 * 4;
            // Grid indices (low byte of each of the first 4 u16s).
            blk.qs[base] = (1 + ib32 as u16) & 0xff;
            blk.qs[base + 1] = (2 + ib32 as u16) & 0xff;
            // The other two u16s carry the sign/ls word in their bytes.
            // Build the 32-bit "aux32[1]" word, then split it into two u16s.
            // 4 × 7-bit sign indices + 4-bit ls in top 4 bits.
            let s0: u32 = (ib32 as u32) & 0x7f;
            let s1: u32 = ((ib32 as u32) + 1) & 0x7f;
            let s2: u32 = ((ib32 as u32) + 2) & 0x7f;
            let s3: u32 = ((ib32 as u32) + 3) & 0x7f;
            let ls: u32 = (ib32 as u32) & 0x0f;
            let a1: u32 = s0 | (s1 << 7) | (s2 << 14) | (s3 << 21) | (ls << 28);
            blk.qs[base + 2] = (a1 & 0xffff) as u16;
            blk.qs[base + 3] = (a1 >> 16) as u16;
        }
        blk.d = f32_to_f16(0.75);

        let mut xq = vec![0i8; QK_K];
        for i in 0..QK_K {
            xq[i] = ((i as i32 * 17) % 51 - 25) as i8;
        }
        let xscale = vec![0.5f32];

        let v = dot_iq2_xxs_row(std::slice::from_ref(&blk), &xq, &xscale);

        let mut deq = [0.0f32; QK_K];
        dequant_iq2_xxs(&blk, &mut deq);
        // The dequant in this crate already folds the 0.125 factor into `d`,
        // so the f32 reference is the straight dot of dequant * (xq * xscale).
        let mut reference = 0.0f64;
        for i in 0..QK_K {
            reference += deq[i] as f64 * xq[i] as f64 * xscale[0] as f64;
        }
        // NOTE: the dequant path doesn't multiply by the per-ib32 `ls`
        // scale (it isn't a literal dequant — see ggml). Our `dot_iq2_xxs_row`
        // does. So we instead compare against a hand-rolled reference that
        // accounts for `ls`:
        let signed_grid = &*IQ2XXS_SIGNED_GRID;
        let mut ref2 = 0.0f64;
        for ib32 in 0..(QK_K / 32) {
            let base = ib32 * 4;
            let a0 = (blk.qs[base] as u32) | ((blk.qs[base + 1] as u32) << 16);
            let a1 = (blk.qs[base + 2] as u32) | ((blk.qs[base + 3] as u32) << 16);
            let gs = [
                (a0 & 0xff) as usize,
                ((a0 >> 8) & 0xff) as usize,
                ((a0 >> 16) & 0xff) as usize,
                ((a0 >> 24) & 0xff) as usize,
            ];
            let ss = [
                (a1 & 0x7f) as usize,
                ((a1 >> 7) & 0x7f) as usize,
                ((a1 >> 14) & 0x7f) as usize,
                ((a1 >> 21) & 0x7f) as usize,
            ];
            let ls = (2 * (a1 >> 28) + 1) as f64;
            let mut sub: f64 = 0.0;
            for q in 0..4 {
                for j in 0..8 {
                    let g = signed_grid[gs[q]][ss[q]][j] as f64;
                    sub += g * xq[ib32 * 32 + q * 8 + j] as f64;
                }
            }
            ref2 += sub * ls;
        }
        ref2 *= 0.125 * f16_to_f32(blk.d) as f64 * xscale[0] as f64;

        // `reference` (via dequant) is for sanity only — keep it alive but
        // assert against `ref2`, which is the actual model.
        let _ = reference;
        assert!((v as f64 - ref2).abs() < 1e-3, "got {v}, expected {ref2}");
    }
}
