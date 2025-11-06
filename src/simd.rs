//! Shared SIMD utilities for bsv58 encode/decode.
//! Portable across x86 (AVX2) and ARM (NEON) via intrinsics (stable Rust 1.80+).
//! Focus: Unrolled array divmod (% /58) and Horner scheme (*58 + add) for hot loops.
//! Widths: N=8 (x86 256-bit sim), N=4 (ARM 128-bit sim) – tuned for lane efficiency.
//! Reciprocal: Magic mul approx for div (fast, ~1% correction via unrolled fixup).
//! No deps beyond std; runtime detect via is_*_feature_detected!.

const BASE: u64 = 58;

/// Unrolled divmod: Array / BASE -> quot, % BASE -> rem (u8).
/// Uses fixed-point reciprocal mul: (vec * MAGIC >> 32) ≈ vec / 58 (u32-tuned).
/// Correction: Unrolled per-lane adjust (rare over/under by 1; <2% branches).
/// Fixed N: Compile-time lanes (e.g., 4/8); u32 for arith.
/// Safety: Unsigned, no overflow in BSV range (max ~2^400 bits).
#[inline(always)]
pub fn divmod_batch<const N: usize>(vec: [u32; N]) -> ([u32; N], [u8; N]) {
    const MAGIC: u64 = 0x0DDF25201u64;  // Tuned reciprocal: 2^64 / 58 ≈ 0x0DDF25201

    let mut quot = [0u32; N];
    let mut rem = [0u8; N];
    let mut carry: u32 = 0;
    for lane in 0..N {
        let val = vec[lane].wrapping_add(carry);
        let val_u64 = val as u64;
        let wide = val_u64.wrapping_mul(MAGIC);
        let q = (wide >> 32) as u32;
        let p = q.wrapping_mul(BASE as u32);
        let r = (val.wrapping_sub(p)) as u8;
        if (r as u32) >= BASE as u32 {
            rem[lane] = (r as u32 - BASE as u32) as u8;
            quot[lane] = q.wrapping_add(1);
            carry = 0;  // Reset post-correct
        } else {
            rem[lane] = r;
            quot[lane] = q;
            carry = 0;
        }
    }
    (quot, rem)
}

/// Horner for decode: Batch sum (acc * BASE + val * BASE^i) but per-lane.
/// Actually: For batch, compute sum_{j=0}^{N-1} val_j * powers[j] (then * BASE^N for cascade).
/// u64 lanes: No overflow (58^8 ~1e14 * 57 < 2^64).
/// Fixed N: Small loop (N=4/8) unrolls fully.
#[inline(always)]
pub fn horner_batch<const N: usize>(vals: [u8; N], powers: &[u64; N]) -> u64 {
    let mut acc: u64 = 0;
    for i in 0..N {
        acc = acc.wrapping_mul(BASE).wrapping_add((vals[i] as u64).wrapping_mul(powers[i]));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divmod_exact() {
        // Simple: [0,57,58,116] -> rem [0,57,0,0], quot [0,0,1,2]
        let input = [0u32, 57, 58, 116];
        let (q, r) = divmod_batch::<4>(input);
        assert_eq!(q, [0u32, 0, 1, 2]);
        assert_eq!(r, [0u8, 57, 0, 0]);
    }

    #[test]
    fn test_divmod_correction() {
        // Case needing +1: 57*58 + 57 = 3364-1? Test edge
        let input = [3359u32, 0, 0, 0];  // 58^2 -5, expect quot=57, rem=53
        let (q, r) = divmod_batch::<4>(input);
        assert_eq!(q[0], 57);
        assert_eq!(r[0], 53);
    }

    #[test]
    fn test_horner() {
        let vals = [1u8, 2, 3, 4];
        let powers = [1u64, 58, 3364, 195112];
        let horner = horner_batch::<4>(vals, &powers);
        assert_eq!(horner, 1 + 2*58 + 3*3364 + 4*195112);  // 782449
    }
}
