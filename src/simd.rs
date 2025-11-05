#![allow(clippy::manual_map)]  // For perf

use std::simd::{u8x16, u32x16, SimdUint, Std140};  // Rust 1.91+

const BASE: u32 = 58;

// Fixed-point mul for /58: 2^32 / 58 ≈ 73638637.655, but use exact reciprocal approx
const RECIP: u64 = 0x0DDF25201u64;  // Tuned for u32 div: (x * RECIP) >> 32 ≈ x / 58

pub fn simd_divmod_u32(vec: u32x16) -> (u32x16, u32x16) {  // (quot, rem)
    let wide = vec.cast::<u64>() * u64x16::splat(RECIP);
    let quot = (wide >> 32).cast::<u32>();
    let rem = vec - quot * u32x16::splat(BASE);
    // Correction loop (rare over/under)
    let mut q = quot;
    let mut r = rem;
    for lane in 0..16 {
        if r[lane] >= BASE {
            r[lane] -= BASE;
            q[lane] += 1;
        }
    }
    (q, r.cast())  // Rem to u8x16
}

// Decode helper: Batch *58 + add for 16 digits
pub fn simd_horner(digits: u8x16, powers: &[u32; 16]) -> u32x16 {  // Partial sum
    let vals = digits.cast::<u32>();
    let mut acc = u32x16::splat(0);
    for i in 0..16 {
        acc = acc * u32x16::splat(BASE) + vals * u32x16::splat(powers[i]);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divmod() {
        let input = u32x16::from_array([57, 58, 59, 0, 1, 57*58, 58*58, 0u32; 8]);
        let (q, r) = simd_divmod_u32(input);
        assert_eq!(r[0], 57);
        assert_eq!(q[1], 1);
    }
}
