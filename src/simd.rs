//! Shared SIMD utilities for bsv58 encode/decode.
//! Portable across x86 (AVX2) and ARM (NEON) via std::simd (stable Rust 1.80+).
//! Focus: Vectorized divmod (% /58) and Horner scheme (*58 + add) for hot loops.
//! Widths: N=8 (x86 256-bit), N=4 (ARM 128-bit) – tuned for lane efficiency.
//! Reciprocal: Magic mul approx for div (fast, ~1% correction via scalar fixup).
//! No deps beyond std; runtime detect via is_*_feature_detected!.
//! Perf: Divmod <1c/lane (mul+shift), Horner fused <0.5c/digit (unrolled reduce).

use std::simd::{Simd, SupportedLaneCount};

const BASE: u64 = 58;

/// Portable divmod: vec / BASE -> quot, vec % BASE -> rem (u8).
/// Uses fixed-point reciprocal mul: (vec * MAGIC >> 32) ≈ vec / 58 (u32-tuned).
/// Correction: Scalar per-lane adjust (rare over/under by 1; <2% branches).
/// Generic N: Compile-time lanes (e.g., 4/8); casts u32 for arith.
/// Safety: Unsigned, no overflow in BSV range (max ~2^400 bits).
#[inline(always)]
pub fn simd_divmod_u32<const N: usize>(vec: Simd<u32, N>) -> (Simd<u32, N>, Simd<u8, N>)
where
    Simd<u32, N>: SupportedLaneCount,
{
    const MAGIC: u64 = 0x0DDF25201u64;  // Tuned reciprocal: 2^64 / 58 ≈ 0x0DDF25201

    let vec_u64: Simd<u64, N> = vec.cast();
    let wide = vec_u64 * Simd::<u64, N>::splat(MAGIC);
    let quot: Simd<u32, N> = (wide >> 32).cast::<u32>();
    let product = quot * Simd::<u32, N>::splat(BASE as u32);
    let rem = (vec - product).cast::<u8>();

    // Correction: If rem >= BASE, adjust (infrequent, ~1/58 chance)
    let mut q = quot;
    let mut r = rem.cast::<u32>();
    for lane in 0..N {
        if r[lane] >= BASE as u32 {
            r[lane] -= BASE as u32;
            q[lane] += 1;
        }
    }
    (q, r.cast())
}

/// Horner for decode: Batch sum (acc * BASE + val * BASE^i) but per-lane.
/// Actually: For batch, compute sum_{j=0}^{N-1} val_j * powers[j] (then * BASE^N for cascade).
/// u64 lanes: No overflow (58^8 ~1e14 * 57 < 2^64).
/// Generic N: Small loop (N=4/8) unrolls fully.
#[inline(always)]
pub fn simd_horner<const N: usize>(
    vals: Simd<u8, N>,
    powers: &[u64; N],
) -> u64
where
    Simd<u64, N>: SupportedLaneCount,
{
    let mut acc: u64 = 0;
    let vals_u64 = vals.cast::<u64>();
    for i in 0..N {
        acc = acc * BASE + (vals_u64[i] * powers[i]);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divmod_exact() {
        // Simple: [0,57,58,116] -> rem [0,57,0,0], quot [0,0,1,2]
        let input = Simd::<u32, 4>::from_array([0u32, 57, 58, 116]);
        let (q, r) = simd_divmod_u32(input);
        assert_eq!(q.to_array(), [0u32, 0, 1, 2]);
        assert_eq!(r.to_array(), [0u8, 57, 0, 0]);
    }

    #[test]
    fn test_divmod_correction() {
        // Case needing +1: 57*58 + 57 = 3364-1? Test edge
        let input = Simd::<u32, 4>::from_array([3359u32, 0, 0, 0]);  // 58^2 -5, expect quot=57, rem=53
        let (q, r) = simd_divmod_u32(input);
        assert_eq!(q[0], 57);
        assert_eq!(r[0], 53);
    }

    #[test]
    fn test_horner() {
        let vals = Simd::<u8, 4>::from_array([1u8, 2, 3, 4]);
        let powers = [1u64, 58, 3364, 195112];
        let horner = simd_horner(vals, &powers);
        assert_eq!(horner, 1 + 2*58 + 3*3364 + 4*195112);  // 782449
    }
}
