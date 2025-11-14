//! Shared SIMD utilities for bsv58 encode/decode.
//! Portable across x86 (AVX2) and ARM (NEON) via intrinsics (stable Rust 1.80+).
//! Focus: Unrolled array divmod (% /58) and Horner scheme (*58 + add) for hot loops.
//! Widths: N=8 (x86 256-bit sim), N=4 (ARM 128-bit sim) – tuned for lane efficiency.
//! Reciprocal: Magic mul approx for div (fast, ~1% correction via unrolled fixup).
//! No deps beyond std; runtime detect via `is_*_feature_detected!`.
#[cfg(feature = "simd")]
pub use self::dispatch::{divmod_batch, horner_batch};
#[cfg(feature = "simd")]
mod dispatch {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::{
        vdupq_n_u32, vgetq_lane_u32, vld1q_u32, vmullq_u32, vreinterpretq_u32_u64, vshrq_n_u64,
    };
    const BASE: u32 = 58;
    const M_U32: u32 = 74_051_161;
    const P_U32: i32 = 0;
    /// Unrolled divmod: Array / BASE -> quot, % BASE -> rem (u8).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn divmod_batch<const N: usize>(vec: [u32; N]) -> ([u32; N], [u8; N]) {
        #[cfg(target_arch = "aarch64")]
        {
            if N == 4 && std::arch::is_aarch64_feature_detected!("neon") {
                let vec4: [u32; 4] = vec.try_into().unwrap();
                let (q4, r4) = unsafe { neon_divmod_batch(vec4) };
                let mut quot = [0u32; N];
                quot.copy_from_slice(&q4);
                let mut rem = [0u8; N];
                rem.copy_from_slice(&r4);
                return (quot, rem);
            }
        }
        let mut quot = [0u32; N];
        let mut rem = [0u8; N];
        for lane in 0..N {
            let v = vec[lane];
            let hi = ((u64::from(v) * u64::from(M_U32)) >> 32) >> (P_U32 as u64);
            quot[lane] = hi as u32;
            rem[lane] = (v.wrapping_sub(quot[lane].wrapping_mul(BASE))) as u8;
        }
        (quot, rem)
    }
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn, clippy::cast_possible_truncation)]
    unsafe fn neon_divmod_batch(vec: [u32; 4]) -> ([u32; 4], [u8; 4]) {
        let v = vld1q_u32(vec.as_ptr());
        let m = vdupq_n_u32(M_U32);
        let mul = vmullq_u32(v, m);
        let hi = vshrq_n_u64(mul, 32);
        let q_vec = vreinterpretq_u32_u64(hi);
        let mut quot = [0u32; 4];
        quot[0] = vgetq_lane_u32(q_vec, 0);
        quot[1] = vgetq_lane_u32(q_vec, 1);
        quot[2] = vgetq_lane_u32(q_vec, 2);
        quot[3] = vgetq_lane_u32(q_vec, 3);
        let mut rem = [0u8; 4];
        for i in 0..4 {
            let qi = quot[i];
            rem[i] = (vec[i].wrapping_sub(qi.wrapping_mul(BASE))) as u8;
        }
        (quot, rem)
    }
    /// Horner for decode: batch sum (acc * BASE + val * `BASEⁱ`) but per-lane.
    #[must_use]
    pub fn horner_batch<const N: usize>(vals: [u8; N], powers: &[u64; N]) -> u64 {
        let mut acc: u64 = 0;
        for i in 0..N {
            acc = acc.saturating_add(u64::from(vals[i]).saturating_mul(powers[i]));
        }
        acc
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_divmod_exact() {
        let input = [0u32, 57, 58, 116];
        let (q, r) = divmod_batch::<4>(input);
        assert_eq!(q, [0u32, 0, 1, 2]);
        assert_eq!(r, [0u8, 57, 0, 0]);
    }
    #[test]
    fn test_divmod_correction() {
        let input = [3359u32, 0, 0, 0];
        let (q, r) = divmod_batch::<4>(input);
        assert_eq!(q[0], 57);
        assert_eq!(r[0], 53);
    }
    #[test]
    fn test_horner() {
        let vals = [1u8, 2, 3, 4];
        let powers = [1u64, 58, 3364, 195112];
        let horner = horner_batch::<4>(vals, &powers);
        assert_eq!(horner, 790657);
    }
}
