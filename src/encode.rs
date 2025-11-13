//! Base58 encoding module for bsv58.
//! Specialized for Bitcoin SV: Bitcoin alphabet, leading zero handling as '1's.
//! Optimizations: u64 limbs (LE) for fewer ops; repeated divmod with u128 temp for correctness.
//! Perf: O(n^2 / 8) scalar; SIMD-ready for batch divmod.
use crate::ALPHABET;
#[must_use]
#[inline]
pub fn encode(input: &[u8]) -> String {
    if input.is_empty() {
        return String::new();
    }
    let zeros = input.iter().take_while(|&&b| b == 0).count();
    let non_zero = &input[zeros..];
    if non_zero.is_empty() {
        return "1".repeat(zeros);
    }
    // Pack to u64 LE limbs (low limb first)
    let mut num: Vec<u64> = Vec::new();
    let mut i = non_zero.len();
    while i > 0 {
        let mut limb = 0u64;
        let mut shift = 0u32;
        for _ in 0..8 {
            if i > 0 {
                i -= 1;
                limb |= u64::from(non_zero[i]) << shift;
            }
            shift += 8;
            if shift >= 64 {
                break;
            }
        }
        num.push(limb);
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let mut output = Vec::with_capacity((non_zero.len() as f64 * 1.3652).ceil() as usize);
    loop {
        let mut remainder = 0u64;
        let mut all_zero = true;
        for limb in &mut num {
            #[allow(clippy::cast_possible_truncation)]
            let temp = u128::from(remainder) * (1u128 << 64) + u128::from(*limb);
            *limb = (temp / 58) as u64;
            remainder = (temp % 58) as u64;
            if *limb != 0 {
                all_zero = false;
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        output.push(ALPHABET[remainder as usize]);
        if all_zero {
            break;
        }
        // Trim leading zero limbs (high end)
        while num.last() == Some(&0) {
            num.pop();
        }
        if num.is_empty() {
            break;
        }
    }
    output.reverse();
    let mut result = String::with_capacity(zeros + output.len());
    result.extend(std::iter::repeat_n('1', zeros));
    result.extend(output.into_iter().map(|b| b as char));
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;
    #[test]
    fn encode_known_no_zeros() {
        assert_eq!(encode(b""), "");
        assert_eq!(encode(b"hello"), "Cn8eVZg");
        let txid = hex!("a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0");
        assert_eq!(
            encode(&txid),
            "BtCjvJYNhqehX2sbzvBNrbkCYp2qfc6AepXfK1JGnELw"
        );
    }
    #[test]
    fn encode_with_zeros() {
        assert_eq!(encode(&hex!("00")), "1");
        assert_eq!(
            encode(&hex!(
                "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
            )),
            "111114VYJtj3yEDffZem7N3PkK563wkLZZ8RjKzcfY"
        );
    }
    #[test]
    fn encode_large() {
        let large = vec![0u8; 50];
        let encoded = encode(&large);
        assert_eq!(encoded, "1".repeat(50));
    }
    #[test]
    fn correctness() {
        let long = vec![42u8; 64];
        let enc = encode(&long);
        let dec = crate::decode(&enc).unwrap();
        assert_eq!(dec, long);
    }
    #[test]
    fn wide_limbs_correctness() {
        // Smoke large
        let large = vec![42u8; 1024];
        let enc = encode(&large);
        let dec = crate::decode(&enc).unwrap();
        assert_eq!(dec, large);
    }
}
