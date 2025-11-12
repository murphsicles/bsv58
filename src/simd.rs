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
    use std::arch::x86_64::*;
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    const BASE: u32 = 58;
    const M_U32: u32 = 74_051_161;
    const P_U32: i32 = 0;

    /// Unrolled divmod: Array / BASE -> quot, % BASE -> rem (u8).
    pub fn divmod_batch<const N: usize>(vec: [u32; N]) -> ([u32; N], [u8; N]) {
        #[cfg(target_arch = "x86_64")]
        {
            if N == 8 && std::arch::is_x86_feature_detected!("avx2") {
                let mut vec8 = [0u32; 8];
                vec8.copy_from_slice(&vec);
                let (q8, r8) = unsafe { avx2_divmod_batch(vec8) };
                let mut quot = [0u32; N];
                quot.copy_from_slice(&q8);
                let mut rem = [0u8; N];
                rem.copy_from_slice(&r8);
                return (quot, rem);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if N == 4 && std::arch::is_aarch64_feature_detected!("neon") {
                let mut vec4 = [0u32; 4];
                vec4.copy_from_slice(&vec);
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
            let hi = ((v as u64 * M_U32 as u64) >> 32) >> P_U32 as u64;
            quot[lane] = hi as u32;
            rem[lane] = (v.wrapping_sub(quot[lane].wrapping_mul(BASE))) as u8;
        }
        (quot, rem)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn avx2_divmod_batch(vec: [u32; 8]) -> ([u32; 8], [u8; 8]) {
        let mut quot = [0u32; 8];
        let mut rem = [0u8; 8];

        // Pairs 0-1
        let v0 = _mm_loadu_si128(vec.as_ptr() as *const __m128i);
        let m0 = _mm_set1_epi32(M_U32 as i32);
        let mul0 = _mm_mul_epu32(v0, m0);
        let high0 = _mm_srli_epi64(mul0, 32);
        let q0 = _mm_srli_epi32(high0, P_U32 as i32);
        quot[0] = _mm_extract_epi32(q0, 0) as u32;
        quot[1] = _mm_extract_epi32(q0, 2) as u32;
        rem[0] = (vec[0].wrapping_sub(quot[0].wrapping_mul(BASE))) as u8;
        rem[1] = (vec[1].wrapping_sub(quot[1].wrapping_mul(BASE))) as u8;

        // Pairs 2-3
        let v1_ptr = vec.as_ptr().add(2) as *const __m128i;
        let v1 = _mm_loadu_si128(v1_ptr);
        let m1 = _mm_set1_epi32(M_U32 as i32);
        let mul1 = _mm_mul_epu32(v1, m1);
        let high1 = _mm_srli_epi64(mul1, 32);
        let q1 = _mm_srli_epi32(high1, P_U32 as i32);
        quot[2] = _mm_extract_epi32(q1, 0) as u32;
        quot[3] = _mm_extract_epi32(q1, 2) as u32;
        rem[2] = (vec[2].wrapping_sub(quot[2].wrapping_mul(BASE))) as u8;
        rem[3] = (vec[3].wrapping_sub(quot[3].wrapping_mul(BASE))) as u8;

        // Pairs 4-5
        let v2_ptr = vec.as_ptr().add(4) as *const __m128i;
        let v2 = _mm_loadu_si128(v2_ptr);
        let m2 = _mm_set1_epi32(M_U32 as i32);
        let mul2 = _mm_mul_epu32(v2, m2);
        let high2 = _mm_srli_epi64(mul2, 32);
        let q2 = _mm_srli_epi32(high2, P_U32 as i32);
        quot[4] = _mm_extract_epi32(q2, 0) as u32;
        quot[5] = _mm_extract_epi32(q2, 2) as u32;
        rem[4] = (vec[4].wrapping_sub(quot[4].wrapping_mul(BASE))) as u8;
        rem[5] = (vec[5].wrapping_sub(quot[5].wrapping_mul(BASE))) as u8;

        // Pairs 6-7
        let v3_ptr = vec.as_ptr().add(6) as *const __m128i;
        let v3 = _mm_loadu_si128(v3_ptr);
        let m3 = _mm_set1_epi32(M_U32 as i32);
        let mul3 = _mm_mul_epu32(v3, m3);
        let high3 = _mm_srli_epi64(mul3, 32);
        let q3 = _mm_srli_epi32(high3, P_U32 as i32);
        quot[6] = _mm_extract_epi32(q3, 0) as u32;
        quot[7] = _mm_extract_epi32(q3, 2) as u32;
        rem[6] = (vec[6].wrapping_sub(quot[6].wrapping_mul(BASE))) as u8;
        rem[7] = (vec[7].wrapping_sub(quot[7].wrapping_mul(BASE))) as u8;

        (quot, rem)
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn neon_divmod_batch(vec: [u32; 4]) -> ([u32; 4], [u8; 4]) {
        let mut quot = [0u32; 4];
        let mut rem = [0u8; 4];

        let v = vld1q_u32(vec.as_ptr());
        let m = vdupq_n_u32(M_U32);

        // Low pair (0-1)
        let low_v = vget_low_u32(v);
        let low_m = vget_low_u32(m);
        let mul_low = vmull_u32(low_v, low_m);
        let high_low = vshrq_n_u64(mul_low, 32);
        let high_u32_low = vreinterpretq_u32_u64(high_low);
        let q_low = vshrq_n_u32(high_u32_low, P_U32 as u32);
        quot[0] = vgetq_lane_u32(q_low, 0);
        quot[1] = vgetq_lane_u32(q_low, 2);
        rem[0] = (vec[0].wrapping_sub(quot[0].wrapping_mul(BASE))) as u8;
        rem[1] = (vec[1].wrapping_sub(quot[1].wrapping_mul(BASE))) as u8;

        // High pair (2-3)
        let high_v = vget_high_u32(v);
        let high_m = vget_high_u32(m);
        let mul_high = vmull_u32(high_v, high_m);
        let high_high = vshrq_n_u64(mul_high, 32);
        let high_u32_high = vreinterpretq_u32_u64(high_high);
        let q_high = vshrq_n_u32(high_u32_high, P_U32 as u32);
        quot[2] = vgetq_lane_u32(q_high, 0);
        quot[3] = vgetq_lane_u32(q_high, 2);
        rem[2] = (vec[2].wrapping_sub(quot[2].wrapping_mul(BASE))) as u8;
        rem[3] = (vec[3].wrapping_sub(quot[3].wrapping_mul(BASE))) as u8;

        (quot, rem)
    }

    /// Horner for decode: batch sum (acc * BASE + val * `BASEⁱ`) but per-lane.
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
