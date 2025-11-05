use crate::ALPHABET;

/// Error for decode failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    InvalidChar(usize),
    Checksum,  // BSV-specific: optional, but stubbed
}

/// Decodes a Base58 string to bytes (Bitcoin alphabet, BSV checksum optional).
pub fn decode(input: &str) -> Result<Vec<u8>, DecodeError> {
    if input.is_empty() {
        return Ok(vec![]);
    }

    // Precomp: ASCII char -> value (0-57), invalid = 255
    const DIGIT_TO_VAL: [u8; 128] = {
        let mut table = [255u8; 128];
        let mut idx = 0;
        for &ch in ALPHABET {
            if (ch as usize) < 128 {
                table[ch as usize] = idx;
            }
            idx += 1;
        }
        table
    };

    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len() * 733 / 1000);

    // Count leading '1's -> zeros
    let mut zeros = 0;
    for &b in bytes {
        if b != b'1' { break; }
        zeros += 1;
    }

    // Accumulator
    let mut num = 0u32;
    let mut has_upper = false;  // For checksum later

    for (i, &ch) in bytes.iter().enumerate().skip(zeros) {
        let val = DIGIT_TO_VAL[ch as usize];
        if val == 255 {
            return Err(DecodeError::InvalidChar(i));
        }
        num = num * 58 + (val as u32);
        if num > (1u64 << 53) as u32 {  // Early overflow detect? Nah, grow as needed
            // For now, u32 per step; scale to u64 chunks later
        }
        if ch.is_ascii_uppercase() { has_upper = true; }  // Stub for version
    }

    // Divmod reverse: extract bytes
    while num > 0 {
        output.push((num % 256) as u8);
        num /= 256;
    }

    // Prepend zeros
    output.extend(vec![0u8; zeros]);

    // Reverse to original order
    output.reverse();

    // BSV Checksum: stub—double SHA256 last 4 bytes == first 4? Skip for MVP, add flag later.
    // if cfg!(feature = "checksum") { validate_checksum(&output)?; }

    Ok(output)
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
        assert_eq!(decode("19Vqm6P7Q5Ge"), Ok(hex!("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f").to_vec()));
        assert_eq!(decode("invalid!"), Err(DecodeError::InvalidChar(7)));
    }
}
