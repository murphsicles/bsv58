//! Base58 encoding module for bsv58.
//! Specialized for Bitcoin SV: Bitcoin alphabet, leading zero handling as '1's.
//! Optimizations: Recursive divide-and-conquer for O(L log L) time (linear practical); u64 small encode.
//! Perf: Linear scalar; SIMD dispatch ready.
use crate::ALPHABET;
const CHUNK_SIZE: usize = 8;
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
    let digits = to_digits_le(non_zero);
    let mut result = String::with_capacity(digits.len() + zeros);
    for &dig in digits.iter().rev() {
        result.push(ALPHABET[dig as usize] as char);
    }
    "1".repeat(zeros) + &result
}
fn to_digits_le(bytes: &[u8]) -> Vec<u8> {
    if bytes.len() <= CHUNK_SIZE {
        to_digits_le_small(bytes)
    } else {
        let mid = bytes.len() / 2;
        let high_bytes = &bytes[..mid];
        let low_bytes = &bytes[mid..];
        let high_digits = to_digits_le(high_bytes);
        let low_digits = to_digits_le(low_bytes);
        let d = low_digits.len();
        let mut full = Vec::with_capacity(high_digits.len() + d);
        full.extend_from_slice(&high_digits);
        let low_start = full.len();
        full.extend(vec![0u8; d]);
        for (i, &dig) in low_digits.iter().enumerate() {
            full[low_start + i] = dig;
        }
        while full.last() == Some(&0) {
            full.pop();
        }
        if full.is_empty() {
            full.push(0);
        }
        full
    }
}
fn to_digits_le_small(bytes: &[u8]) -> Vec<u8> {
    let mut val: u64 = 0;
    for &b in bytes {
        val = val.wrapping_mul(256).wrapping_add(u64::from(b));
    }
    let mut digits = Vec::new();
    if val == 0 {
        digits.push(0);
    } else {
        while val > 0 {
            digits.push((val % 58) as u8);
            val /= 58;
        }
    }
    digits
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
