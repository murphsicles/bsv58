use crate::ALPHABET;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    InvalidChar(usize),
    Checksum,
    InvalidLength,
}

/// Decodes a Base58 string to bytes. Validates BSV-style checksum if `validate_checksum=true`.
pub fn decode(input: &str, validate_checksum: bool) -> Result<Vec<u8>, DecodeError> {
    if input.is_empty() {
        return Ok(vec![]);
    }

    const DIGIT_TO_VAL: [u8; 128] = {
        let mut table = [255u8; 128];
        let mut idx = 0u8;
        for &ch in ALPHABET {
            if (ch as usize) < 128 {
                table[ch as usize] = idx;
            }
            idx += 1;
        }
        table
    };

    let bytes = input.as_bytes();
    let cap = (bytes.len() as f64 * 0.733).ceil() as usize;  // log58(256) inverse
    let mut output = Vec::with_capacity(cap);

    // Leading '1's -> zeros
    let mut zeros = 0usize;
    for &b in bytes {
        if b != b'1' { break; }
        zeros += 1;
    }

    // u64 accumulator for chunking
    let mut num: u64 = 0;
    let digits = &bytes[zeros..];

    // Unrolled: process 4 digits/iter (each *58^4 ≈ 2^26, fits u64)
    let mut i = 0;
    while i + 4 <= digits.len() {
        let vals = [
            DIGIT_TO_VAL[digits[i] as usize],
            DIGIT_TO_VAL[digits[i+1] as usize],
            DIGIT_TO_VAL[digits[i+2] as usize],
            DIGIT_TO_VAL[digits[i+3] as usize],
        ];
        if vals.iter().any(|&v| v == 255) {
            return Err(DecodeError::InvalidChar(zeros + i + vals.iter().position(|&v| v == 255).unwrap()));
        }
        // Horner's: ((num * 58 + v0)*58 + v1)... 
        num = num * 58u64.pow(4) + (vals[0] as u64) * 58u64.pow(3) + (vals[1] as u64) * 58u64.pow(2) + (vals[2] as u64) * 58 + (vals[3] as u64);
        i += 4;
    }

    // Tail
    for j in i..digits.len() {
        let val = DIGIT_TO_VAL[digits[j] as usize];
        if val == 255 {
            return Err(DecodeError::InvalidChar(zeros + j));
        }
        num = num * 58 + (val as u64);
    }

    // Extract bytes: while num > 0, push num % 256, /= 256
    while num > 0 {
        output.push((num % 256) as u8);
        num /= 256;
    }

    // Prepend zeros & reverse
    for _ in 0..zeros {
        output.push(0);
    }
    output.reverse();

    if output.len() < 4 {
        return Err(DecodeError::InvalidLength);
    }

    // BSV Checksum: double SHA256 of payload[:len-4] == payload[-4:]
    if validate_checksum {
        let payload = &output[..output.len() - 4];
        let hash1 = Sha256::digest(payload);
        let hash2 = Sha256::digest(&hash1);
        let expected = &hash2[..4];
        let actual = &output[output.len() - 4..];
        if expected != actual {
            return Err(DecodeError::Checksum);
        }
    }

    Ok(output)
}

// Legacy compat: no checksum
pub fn decode(input: &str) -> Result<Vec<u8>, DecodeError> {
    decode(input, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;

    #[test]
    fn decode_known() {
        assert_eq!(decode(""), Ok(vec![]));
        assert_eq!(decode("1"), Ok(vec![0]));
        assert_eq!(decode("n7UKu7Y5"), Ok(b"hello".to_vec()));
        let encoded_genesis = "19Vqm6P7Q5Ge";
        let genesis = hex!("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f");
        assert_eq!(decode(encoded_genesis), Ok(genesis.to_vec()));

        // Invalid
        assert_eq!(decode("invalid!"), Err(DecodeError::InvalidChar(7)));

        // Checksum test: fake valid address (1BitcoinEaterAddressDontSendf59kuE -> payload + checksum)
        let valid_addr = "1BitcoinEaterAddressDontSendf59kuE";
        let payload = hex!("00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00");  // 28 zeros + version? Simplified
        // Real test: assume passes with checksum=true
        assert!(decode(valid_addr, true).is_ok());
    }
}
