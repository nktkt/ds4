//! f16 ↔ f32 conversion + DeepSeek V4 E4M3FN microscale dequant.
//!
//! Ports `f16_to_f32`, `f32_to_f16`, `dsv4_e4m3fn_value_cpu`, and
//! `dsv4_e4m3fn_dequant_cpu` from `ds4.c`. The conversions use exactly the
//! same bit recipes as the C version so logits match across implementations.

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let mut exp = ((h >> 10) & 0x1f) as u32;
    let mut mant = (h & 0x03ff) as u32;
    let bits;
    if exp == 0 {
        if mant == 0 {
            bits = sign;
        } else {
            exp = 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp = exp.wrapping_sub(1);
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if exp == 31 {
        bits = sign | 0x7f80_0000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    f32::from_bits(bits)
}

#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fffff;

    if exp <= 0 {
        if exp < -10 { return sign; }
        let mant_full = mant | 0x800000;
        let shift = (14 - exp) as u32;
        let mut half_mant = mant_full >> shift;
        let round_bit = (mant_full >> (shift - 1)) & 1;
        let sticky = mant_full & ((1u32 << (shift - 1)) - 1);
        if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
            half_mant += 1;
        }
        return sign | (half_mant as u16);
    }

    if exp >= 31 {
        if ((bits >> 23) & 0xff) == 0xff && mant != 0 {
            return sign | 0x7e00;
        }
        return sign | 0x7c00;
    }

    let mut half = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
    let round = mant & 0x1fff;
    if round > 0x1000 || (round == 0x1000 && (half & 1) != 0) {
        half += 1;
    }
    half
}

pub fn round_inplace(x: &mut [f32]) {
    for v in x.iter_mut() { *v = f16_to_f32(f32_to_f16(*v)); }
}

/// DeepSeek V4 E4M3 mantissa table → linear value. Mirrors
/// `dsv4_e4m3fn_value_cpu`.
pub fn e4m3fn_value(i: i32) -> f32 {
    const EXP_SCALE: [f32; 16] = [
        0.0,         0.015_625,   0.031_25,    0.0625,
        0.125,       0.25,        0.5,         1.0,
        2.0,         4.0,         8.0,         16.0,
        32.0,        64.0,        128.0,       256.0,
    ];
    let exp = (i >> 3) & 0x0f;
    let mant = i & 0x07;
    if exp == 0 {
        mant as f32 * 0.001_953_125
    } else {
        (1.0 + mant as f32 * 0.125) * EXP_SCALE[exp as usize]
    }
}

pub fn e4m3fn_dequant(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(448.0);
    let (mut lo, mut hi) = (0i32, 126i32);
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if e4m3fn_value(mid) <= ax { lo = mid; } else { hi = mid - 1; }
    }
    sign * e4m3fn_value(lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_zero() {
        assert_eq!(f16_to_f32(f32_to_f16(0.0)), 0.0);
    }

    #[test]
    fn round_trip_one() {
        assert!((f16_to_f32(f32_to_f16(1.0)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn matches_half_crate() {
        for &x in &[0.5f32, 1.5, -2.25, 100.0, -0.125] {
            let ours = f16_to_f32(f32_to_f16(x));
            let theirs = half::f16::from_f32(x).to_f32();
            assert!((ours - theirs).abs() < 1e-3, "x={x} ours={ours} theirs={theirs}");
        }
    }

    #[test]
    fn e4m3_zero() {
        assert_eq!(e4m3fn_value(0), 0.0);
    }
}
