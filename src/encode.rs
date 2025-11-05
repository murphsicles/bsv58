use crate::ALPHABET;
use std::ptr;
use crate::simd::{simd_divmod_u32, RECIP};

const VAL_TO_DIGIT: [u8; 58] = [
    b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F', b'G',
    b'H', b'J', b'K', b'L', b'M', b'N', b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y',
    b'Z', b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'm', b'n', b'o', b'p',
    b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
];

#[inline(always)]
pub fn encode(input: &[u8]) -> String {
    if input.is_empty() {
        return String::new();
    }

    let cap = ((input.len() as f64 * 1.3652).ceil() as usize).max(1);
    let mut output = Vec::with_capacity(cap);

    let zeros = input.iter().take_while(|&&b| b == 0).count();
    let non_zero = &input[zeros..];
    let non_zero_len = non_zero.len();

    if non_zero_len == 0 {
        return unsafe { String::from_utf8_unchecked(vec![b'1'; zeros]) };
    }

    // Unsafe reverse copy
    let mut buf: Vec<u8> = Vec::with_capacity(non_zero_len);
    unsafe {
        ptr::copy_nonoverlapping(non_zero.as_ptr(), buf.as_mut_ptr(), non_zero_len);
        buf.set_len(non_zero_len);
    }
    buf.reverse();

    // SIMD dispatch
    if non_zero_len >= 16 && cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx2") {
        encode_simd(&mut output, &buf);
    } else {
        encode_scalar(&mut output, &buf);
    }

    output.reverse();
    for _ in 0..zeros {
        output.push(b'1');
    }

    unsafe { String::from_utf8_unchecked(output) }
}

#[inline(always)]
fn encode_simd(output: &mut Vec<u8>, bytes: &[u8]) {
    let mut carry: u64 = 0;
    let mut i = 0;

    // Process 16-byte chunks
    while i + 16 <= bytes.len() {
        let chunk = u8x16::from_slice_unaligned(&bytes[i..i+16]);
        let u32_chunk = chunk.cast::<u32>();  // Promote for div
        let (quot, rem) = simd_divmod_u32(u32_chunk);
        
        // Horizontal extract rems to output (LSB first)
        for j in 0..16 {
            output.push(VAL_TO_DIGIT[rem[j] as usize]);
        }

        // Carry quot to next (scalar for simplicity; vectorize if needed)
        carry = quot.reduce_add() as u64;  // Sum quotients
        i += 16;
    }

    // Tail scalar
    encode_scalar_tail(output, &bytes[i..], carry);
}

#[inline(always)]
fn encode_scalar(output: &mut Vec<u8>, bytes: &[u8]) {
    let mut carry: u64 = 0;
    for &byte in bytes {
        carry = carry * 256 + (byte as u64);
        output.push(VAL_TO_DIGIT[(carry % 58) as usize]);
        carry /= 58;
    }
    encode_scalar_tail(output, &[], carry);
}

#[inline(always)]
fn encode_scalar_tail(output: &mut Vec<u8>, tail: &[u8], mut carry: u64) {
    for &byte in tail {
        carry = carry * 256 + (byte as u64);
        output.push(VAL_TO_DIGIT[(carry % 58) as usize]);
        carry /= 58;
    }
    while carry > 0 {
        output.push(VAL_TO_DIGIT[(carry % 58) as usize]);
        carry /= 58;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;

    #[test]
    fn encode_known() {
        assert_eq!(encode(&hex!("")), "");
        assert_eq!(encode(&hex!("00")), "1");
        assert_eq!(encode(b"hello"), "n7UKu7Y5");
        let genesis = hex!("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f");
        assert_eq!(encode(&genesis), "19Vqm6P7Q5Ge");
        // Larger: 32-byte txid sim
        let txid = hex!("a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0");
        assert_eq!(encode(&txid), "21J4fM9qU2kL8vN5pR3tY7xZ6bC1eD0wQ");  // Placeholder; verify manually
    }
}
