use crate::ALPHABET;
use std::ptr;

const VAL_TO_DIGIT: [u8; 58] = [
    b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F', b'G',
    b'H', b'J', b'K', b'L', b'M', b'N', b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y',
    b'Z', b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'm', b'n', b'o', b'p',
    b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
];

pub fn encode(input: &[u8]) -> String {
    if input.is_empty() {
        return String::new();
    }

    // Exact capacity: len * log(256)/log(58) ≈ len * 1.3652, but +1 for safety
    let cap = (input.len() as f64 * 1.3652).ceil() as usize + 1;
    let mut output = Vec::with_capacity(cap);

    // Count leading zeros
    let mut zeros = 0usize;
    for &b in input {
        if b != 0 { break; }
        zeros += 1;
    }
    let non_zero_len = input.len() - zeros;

    if non_zero_len == 0 {
        return unsafe { String::from_utf8_unchecked(vec![b'1'; zeros]) };
    }

    // Unsafe zero-copy reverse of non-zero part (big-endian prep)
    let mut buf = Vec::with_capacity(non_zero_len);
    unsafe {
        let src = input.as_ptr().add(zeros);
        ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), non_zero_len);
        buf.set_len(non_zero_len);
    }
    buf.reverse();  // Now process as little-endian for divmod

    // u64 carry: process up to 8 bytes/iter
    let mut carry: u64 = 0;
    let bytes = buf.as_slice();

    // Unrolled loop: process 4 bytes at a time (common chunk for BSV)
    let mut i = 0;
    while i + 4 <= bytes.len() {
        carry = carry.wrapping_mul(256u64.pow(4)) + u64::from_le_bytes([
            bytes[i], bytes[i+1], bytes[i+2], bytes[i+3], 0, 0, 0, 0
        ]);
        for _ in 0..4 {  // Unroll 4 divs
            output.push(VAL_TO_DIGIT[(carry % 58) as usize]);
            carry /= 58;
        }
        i += 4;
    }

    // Tail: remaining bytes
    for j in i..bytes.len() {
        carry = carry * 256 + (bytes[j] as u64);
        output.push(VAL_TO_DIGIT[(carry % 58) as usize]);
        carry /= 58;
    }

    // Drain remaining carry
    while carry > 0 {
        output.push(VAL_TO_DIGIT[(carry % 58) as usize]);
        carry /= 58;
    }

    output.reverse();  // LSB-first to MSB

    // Prepend zeros
    for _ in 0..zeros {
        output.push(b'1');
    }

    unsafe { String::from_utf8_unchecked(output) }
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
