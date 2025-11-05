use std::simd::{LaneIdentity, Simd, SimdPartialOrd, SupportedLaneType};
use std::arch::{self, *};  // For detection

const BASE: u32 = 58;
const RECIP: u64 = 0x469ee59;  // Magic for /58: (x * RECIP) >> 32 ≈ x / 58 (u32 unsigned)

#[cfg(target_arch = "x86_64")]
use std::simd::x86::avx2;  // For u32x8 if needed; here u32x4 for parity

#[cfg(target_arch = "aarch64")]
use std::simd::neon;  // Backing

// Portable divmod: Works on both, but width-tuned
pub fn simd_divmod_u32<const N: usize>(vec: Simd<u32, N>) -> (Simd<u32, N>, Simd<u8, N>) 
where
    LaneIdentity<N>: SupportedLaneType,
{
    let vec_u64: Simd<u64, N> = vec.cast();
    let wide = vec_u64 * Simd::<u64, N>::splat(RECIP);
    let quot: Simd<u32, N> = (wide >> 32).cast();
    let rem = (vec - quot * Simd::<u32, N>::splat(BASE)).cast();
    // Correction: Scalar loop (branchy but rare ~1/58)
    let mut q = quot;
    let mut r = rem.cast::<u32>();
    for lane in 0..N {
        if r[lane] >= BASE {
            r[lane] -= BASE as u32;
            q[lane] += 1;
        }
    }
    (q, r.cast())
}

// Horner for decode: Batch N digits
pub fn simd_horner<const N: usize>(digits: Simd<u8, N>, powers: &[u32; 16]) -> Simd<u32, N> 
where
    LaneIdentity<N>: SupportedLaneType,
{
    let vals = digits.cast::<u32>();
    let mut acc = Simd::<u32, N>::splat(0u32);
    for i in 0..N {
        acc = acc * Simd::<u32, N>::splat(BASE) + vals * Simd::<u32, N>::splat(powers[i]);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divmod() {
        let input = Simd::<u32, 4>::from_array([57, 58, 59, 116]);  // 116/58=2
        let (q, r) = simd_divmod_u32(input);
        assert_eq!(q[0], 0); assert_eq!(r[0], 57);
        assert_eq!(q[1], 1); assert_eq!(r[1], 0);
        assert_eq!(q[2], 1); assert_eq!(r[2], 1);
        assert_eq!(q[3], 2); assert_eq!(r[3], 0);
    }
}
