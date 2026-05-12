//! Dequantization of DS4's quant block formats into plain f32.
//!
//! Ports the Q2_K, Q4_K, IQ2_XXS, and Q8_K dequant paths from `ds4.c`. These
//! are reference implementations: correct first, fast later. Backends do not
//! call these on the hot path — they dequantize on the GPU during matmul.
//! These routines are useful for tests, the CPU reference path, and the
//! `dump_weights` debug tool.

use crate::half::f16_to_f32;
use crate::iq2_tables::{IQ2XXS_GRID, IQ2XXS_SIGNS, KSIGNS_IQ2XS};
use crate::quant::{BlockIQ2XXS, BlockQ2K, BlockQ4K, BlockQ8K, QK_K};

/// Dequant one Q2_K block of 256 elements into `out[..256]`.
///
/// NOTE: this is a preliminary implementation. The exact `Q2_K` sub-block
/// scale layout in `ds4.c` (4-bit `dl`, 4-bit `ml` packed across 16 bytes)
/// still needs bit-perfect cross-validation; the matmul path does not depend
/// on this dequant entry point.
pub fn dequant_q2k(blk: &BlockQ2K, out: &mut [f32; QK_K]) {
    let d = f16_to_f32(blk.d);
    let dmin = f16_to_f32(blk.dmin);
    // Q2_K packs 16 4-bit sub-block scales into `scales[0..16]`.
    let mut o = 0;
    for is in 0..QK_K / 128 {           // 2 super-blocks of 128
        let sc = &blk.scales[is * 8..is * 8 + 8];
        for n in 0..2 {
            for j in 0..16 {
                let dl = d * (sc[n * 4] & 0xf) as f32;
                let ml = dmin * (sc[n * 4] >> 4) as f32;
                let shift = (n * 2) as u32;
                let q = (blk.qs[is * 32 + 16 * n + j] >> shift) & 3;
                out[o] = dl * q as f32 - ml;
                o += 1;
            }
        }
    }
}

/// Dequant one Q4_K block of 256 elements.
pub fn dequant_q4k(blk: &BlockQ4K, out: &mut [f32; QK_K]) {
    let d = f16_to_f32(blk.d);
    let dmin = f16_to_f32(blk.dmin);
    let mut o = 0;
    for is in 0..QK_K / 64 {
        let sc = blk.scales[is];
        let m = blk.scales[is + 4];
        let dl = d * (sc & 63) as f32;
        let ml = dmin * (m & 63) as f32;
        for j in 0..32 {
            let q = (blk.qs[is * 32 + j] & 0x0f) as f32;
            out[o] = dl * q - ml;
            o += 1;
        }
        for j in 0..32 {
            let q = (blk.qs[is * 32 + j] >> 4) as f32;
            out[o] = dl * q - ml;
            o += 1;
        }
    }
}

/// Dequant one IQ2_XXS block. Each block packs 8 (grid index, sign index)
/// pairs in 66 bytes, with one f16 scale.
pub fn dequant_iq2_xxs(blk: &BlockIQ2XXS, out: &mut [f32; QK_K]) {
    let d = f16_to_f32(blk.d) * 0.125; // matches `ds4.c` IQ2_XXS scale factor
    let mut o = 0;
    for ib in 0..QK_K / 32 {
        // Each ib spans 32 outputs = 4 grid lookups × 8 byte lanes.
        let base = ib * 4;
        for sub in 0..4 {
            let packed = blk.qs[base + sub] as u32;
            let grid_idx = (packed & 0xff) as usize;
            let sign_idx = ((packed >> 8) & 0x7f) as usize;
            let _ = KSIGNS_IQ2XS;
            let grid_bytes = IQ2XXS_GRID[grid_idx].to_le_bytes();
            let signs = IQ2XXS_SIGNS[sign_idx];
            for j in 0..8 {
                let v = grid_bytes[j] as i8 as f32 * signs[j] as f32;
                out[o] = d * v;
                o += 1;
            }
        }
    }
}

/// Dequant Q8_K — straightforward `f32` reconstruction.
pub fn dequant_q8k(blk: &BlockQ8K, out: &mut [f32; QK_K]) {
    for i in 0..QK_K {
        out[i] = blk.d * blk.qs[i] as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;

    #[test]
    fn q8k_round_trip() {
        let mut blk = BlockQ8K { d: 0.5, qs: [0; QK_K], bsums: [0; QK_K / 16] };
        for i in 0..QK_K { blk.qs[i] = (i as i32 - 128) as i8; }
        let mut out = [0.0; QK_K];
        dequant_q8k(&blk, &mut out);
        for i in 0..QK_K {
            assert!((out[i] - 0.5 * (blk.qs[i] as f32)).abs() < 1e-6);
        }
    }

    #[test]
    fn q2k_does_not_panic() {
        // Smoke-test only: the exact Q2_K dequant recipe in ds4.c interleaves
        // sub-block scales in a specific way that we still need to mirror
        // bit-for-bit. This test just verifies that calling the routine
        // doesn't trip a bounds check.
        let _ = f32_to_f16;
        let blk = BlockQ2K {
            scales: [0; QK_K / 16],
            qs: [0; QK_K / 4],
            d: 0,
            dmin: 0,
        };
        let mut out = [0.0; QK_K];
        dequant_q2k(&blk, &mut out);
    }
}
