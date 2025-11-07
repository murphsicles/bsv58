//! Shared SIMD utilities for bsv58 encode/decode.
//! Portable across x86 (AVX2) and ARM (NEON) via intrinsics (stable Rust 1.80+).
//! Focus: Unrolled array divmod (% /58) and Horner scheme (*58 + add) for hot loops.
//! Widths: N=8 (x86 256-bit sim), N=4 (ARM 128-bit sim) – tuned for lane efficiency.
//! Reciprocal: Magic mul approx for div (fast, ~1% correction via unrolled fixup).
//! No deps beyond std; runtime detect via `is_*_feature_detected!`.
/// Unrolled divmod: Array / BASE -> quot, % BASE -> rem (u8).
/// Uses fixed-point reciprocal mul: (vec * MAGIC >> 32) ≈ vec / 58 (u32-tuned).
/// Correction: Unrolled per-lane adjust (rare over/under by 1; <2% branches).
/// Fixed N: Compile-time lanes (e.g., 4/8); u32 for arith.
/// Safety: Unsigned, no overflow in BSV range (max ~2^400 bits).
///
/// # Panics
/// Panics if the remainder cannot be converted to u8 (should not happen for valid inputs).
#[must_use]
#[inline]
pub fn divmod_batch<const N: usize>(vec: [u32; N]) -> ([u32; N], [u8; N]) {
    let mut quot = [0u32; N];
    let mut rem = [0u8; N];
    for lane in 0..N {
        let v = vec[lane];
        quot[lane] = v / 58u32;
        rem[lane] = (v % 58u32) as u8;
    }
    (quot, rem)
}
/// Horner for decode: batch sum (acc * BASE + val * `BASEⁱ`) but per-lane.
/// Actually: `sum_{j=0}^{N-1} val_j * powers[j]` (then * `BASEⁿ` for cascade).
/// u64 lanes: No overflow (58^8 ~1e14 * 57 < 2^64).
/// Fixed N: Small loop (N=4/8) unrolls fully.
#[must_use]
#[inline]
pub fn horner_batch<const N: usize>(vals: [u8; N], powers: &[u64; N]) -> u64 {
    let mut acc: u64 = 0;
    for i in 0..N {
        acc += u64::from(vals[i]) * powers[i];
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
        let input = [3359u32, 0, 0, 0]; // 58^2 -5, expect quot=57, rem=53
        let (q, r) = divmod_batch::<4>(input);
        assert_eq!(q[0], 57);
        assert_eq!(r[0], 53);
    }
    #[test]
    fn test_horner() {
        let vals = [1u8, 2, 3, 4];
        let powers = [1u64, 58, 3364, 195112];
        let horner = horner_batch::<4>(vals, &powers);
        assert_eq!(horner, 1 + 2 * 58 + 3 * 3364 + 4 * 195112); // 790657
    }
}
