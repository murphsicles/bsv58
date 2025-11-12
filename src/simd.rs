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
    use std::arch::{aarch64::*, x86_64::*};

    const BASE: u32 = 58;
    const M_U32: u32 = 4_739_274_257;
    const P_U32: i32 = 6;

    /// Unrolled divmod: Array / BASE -> quot, % BASE -> rem (u8).
    pub fn divmod_batch<const N: usize>(mut vec: [u32; N]) -> ([u32; N], [u8; N]) {
        let mut quot = [0u32; N];
        let mut rem = [0u8; N];
        if N == 8 && cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2") {
            unsafe {
                divmod_batch_avx2(&mut vec, &mut quot, &mut rem);
            }
        } else if N == 4 && cfg!(target_arch = "aarch64") && is_aarch64_feature_detected!("neon") {
            unsafe {
                divmod_batch_neon(&mut vec, &mut quot, &mut rem);
            }
        } else {
            for lane in 0..N {
                let v = vec[lane];
                let hi = ((v as u64 * M_U32 as u64) >> 32) >> P_U32 as u64;
                quot[lane] = hi as u32;
                rem[lane] = (v - quot[lane] * BASE) as u8;
            }
        }
        (quot, rem)
    }

    #[cfg(all(target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn divmod_batch_avx2(vec: &mut [u32; 8], quot: &mut [u32; 8], rem: &mut [u8; 8]) {
        // Process in pairs using _mm_mul_epu32 for low/high u32 pairs
        let v0 = _mm_loadu_si128(vec.as_ptr() as *const __m128i);
        let m0 = _mm_set1_epi32(M_U32 as i32);
        let mul0 = _mm_mul_epu32(v0, m0);
        let high0 = _mm_srli_epi64(mul0, 32);
        let q0 = _mm_srai_epi32(_mm_castsi128_si256(high0), P_U32);
        quot[0] = _mm_extract_epi32(q0, 0) as u32;
        quot[1] = _mm_extract_epi32(q0, 1) as u32;
        rem[0] = (vec[0] - quot[0] * BASE) as u8;
        rem[1] = (vec[1] - quot[1] * BASE) as u8;

        let v1 = _mm_loadu_si128(vec[2..].as_ptr() as *const __m128i);
        let m1 = _mm_set1_epi32(M_U32 as i32);
        let mul1 = _mm_mul_epu32(v1, m1);
        let high1 = _mm_srli_epi64(mul1, 32);
        let q1 = _mm_srai_epi32(_mm_castsi128_si256(high1), P_U32);
        quot[2] = _mm_extract_epi32(q1, 0) as u32;
        quot[3] = _mm_extract_epi32(q1, 1) as u32;
        rem[2] = (vec[2] - quot[2] * BASE) as u8;
        rem[3] = (vec[3] - quot[3] * BASE) as u8;

        let v2 = _mm_loadu_si128(vec[4..].as_ptr() as *const __m128i);
        let m2 = _mm_set1_epi32(M_U32 as i32);
        let mul2 = _mm_mul_epu32(v2, m2);
        let high2 = _mm_srli_epi64(mul2, 32);
        let q2 = _mm_srai_epi32(_mm_castsi128_si256(high2), P_U32);
        quot[4] = _mm_extract_epi32(q2, 0) as u32;
        quot[5] = _mm_extract_epi32(q2, 1) as u32;
        rem[4] = (vec[4] - quot[4] * BASE) as u8;
        rem[5] = (vec[5] - quot[5] * BASE) as u8;

        let v3 = _mm_loadu_si128(vec[6..].as_ptr() as *const __m128i);
        let m3 = _mm_set1_epi32(M_U32 as i32);
        let mul3 = _mm_mul_epu32(v3, m3);
        let high3 = _mm_srli_epi64(mul3, 32);
        let q3 = _mm_srai_epi32(_mm_castsi128_si256(high3), P_U32);
        quot[6] = _mm_extract_epi32(q3, 0) as u32;
        quot[7] = _mm_extract_epi32(q3, 1) as u32;
        rem[6] = (vec[6] - quot[6] * BASE) as u8;
        rem[7] = (vec[7] - quot[7] * BASE) as u8;
    }

    #[cfg(all(target_arch = "aarch64"))]
    #[target_feature(enable = "neon")]
    unsafe fn divmod_batch_neon(vec: &mut [u32; 4], quot: &mut [u32; 4], rem: &mut [u8; 4]) {
        let v = vld1q_u32(vec.as_ptr());
        let m = vdupq_n_u32(M_U32);
        let mul = vmull_u32(vget_low_u32(v), vget_low_u32(m)); // Low pair
        let high_low = vshrq_n_u64(mul, 32);
        let q_low = vrshrq_n_s32(vreinterpretq_s32_u64(high_low), P_U32);
        quot[0] = vgetq_lane_u32(vreinterpretq_u32_s32(q_low), 0);
        quot[1] = vgetq_lane_u32(vreinterpretq_u32_s32(q_low), 1);
        rem[0] = (vec[0] - quot[0] * BASE) as u8;
        rem[1] = (vec[1] - quot[1] * BASE) as u8;

        let mul_high = vmull_u32(vget_high_u32(v), vget_high_u32(m)); // High pair
        let high_high = vshrq_n_u64(mul_high, 32);
        let q_high = vrshrq_n_s32(vreinterpretq_s32_u64(high_high), P_U32);
        quot[2] = vgetq_lane_u32(vreinterpretq_u32_s32(q_high), 0);
        quot[3] = vgetq_lane_u32(vreinterpretq_u32_s32(q_high), 1);
        rem[2] = (vec[2] - quot[2] * BASE) as u8;
        rem[3] = (vec[3] - quot[3] * BASE) as u8;
    }

    /// Horner for decode: batch sum (acc * BASE + val * `BASEⁱ`) but per-lane.
    pub fn horner_batch<const N: usize>(vals: [u8; N], powers: &[u64; N]) -> u64 {
        let mut acc: u64 = 0;
        for i in 0..N {
            acc += u64::from(vals[i]) * powers[i];
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
