//! bsv58: Ultra-fast Base58 codec for Bitcoin SV.
//! BSV-only: Bitcoin alphabet, leading-zero '1's, optional double-SHA256 checksum decode.
//! No generics/multi-alphabets—hardcoded for perf. Targets 5x+ bs58-rs on BSV payloads.
//! Exports: `encode(&[u8]) -> String`, `decode(&str) -> Result<Vec<u8>, DecodeError>` (no checksum).
//! For checksum: `decode_full(&str, true)`. SIMD: AVX2 (x86) / NEON (ARM) dispatch; scalar fallback.
//! Rust 1.80+ stable. Usage: `cargo add bsv58`; benches via `cargo bench`.

pub const ALPHABET: [u8; 58] = *b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
mod decode;
mod encode;
mod simd;
#[cfg(feature = "simd")]
pub use simd::{divmod_batch, horner_batch};
/// Encodes bytes to Base58 string (Bitcoin alphabet, leading zeros as '1's).
pub use encode::encode;
/// Decodes Base58 string to bytes (Bitcoin alphabet, no checksum).
pub use decode::decode;
/// Decodes with optional BSV checksum validation (strips on success).
pub use decode::decode_full;
/// Decode errors.
pub use decode::DecodeError;
#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;
    /// BSV test corpus: Addresses (w/checksum), txids, hashes.
    const CORPUS: &[(&[u8], &str)] = &[
        // Empty
        (b"", ""),
        // Leading zeros
        (b"\x00", "1"),
        (b"\x00\x00", "11"),
        // Simple
        (b"hello", "Cn8eVZg"),
        // Genesis block hash (32B w/5 leading zeros)
        (
            &hex!("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"),
            "111114VYJtj3yEDffZem7N3PkK563wkLZZ8RjKzcfY",
        ),
        // P2PKH address payload (21B: version + hash) → raw Base58
        (
            &hex!("00759d6677091e973b9e9d99f19c68fbf43e3f05f9"),
            "12e3A9pcaDSMffCV3iBuhipLAGczU",
        ),
        // Txid sim (32B)
        (
            &hex!("a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0"),
            "BtCjvJYNhqehX2sbzvBNrbkCYp2qfc6AepXfK1JGnELw",
        ),
    ];
    #[test]
    fn roundtrip_no_checksum() {
        for (bytes, encoded) in CORPUS {
            let enc = encode(bytes);
            assert_eq!(enc, *encoded, "Encode fail: {:?}", bytes);
            let dec = decode(&enc).unwrap();
            assert_eq!(dec, *bytes, "Decode fail: {}", enc);
        }
    }
    #[test]
    fn roundtrip_with_checksum() {
        // Only test addrs with checksum (payload < full)
        let addr_cases: &[(&[u8], &str)] = &[(
            &hex!("00759d6677091e973b9e9d99f19c68fbf43e3f05f9"),
            "1BitcoinEaterAddressDontSendf59kuE",
        )];
        for (payload, addr) in addr_cases {
            // Encode payload → should not match addr (no checksum added)
            let enc_raw = encode(payload);
            assert_ne!(enc_raw, *addr);
            // Decode addr w/checksum → get payload
            let dec = decode_full(addr, true).unwrap();
            assert_eq!(dec, *payload, "Checksum decode fail: {}", addr);
        }
    }
    #[test]
    fn invalid_cases() {
        // Invalid char
        assert!(matches!(
            decode("invalid!"),
            Err(DecodeError::InvalidChar(4))
        ));
        // Checksum mismatch (flip last char)
        let invalid_addr = "1BitcoinEaterAddressDontSendf59kuF";
        assert!(matches!(
            decode_full(invalid_addr, true),
            Err(DecodeError::Checksum)
        ));
        // Too short for checksum
        assert!(matches!(
            decode_full("12", true),
            Err(DecodeError::InvalidLength)
        ));
    }
    #[test]
    fn simd_smoke() {
        // No panic on dispatch (SIMD if feat/cpu flags)
        let bytes = b"hello world bsv58 test";
        let enc = encode(bytes);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, bytes);
    }
    #[test]
    fn large_payload() {
        // 50B pubkey (BSV max): No overflow
        let pubkey = vec![0x42u8; 50]; // Dummy
        let enc = encode(&pubkey);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, pubkey);
        assert!(enc.len() >= 68); // ~1.36x
    }
}
