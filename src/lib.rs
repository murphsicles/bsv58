//! Ultra-fast Base58 codec for Bitcoin SV.

pub const ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

mod encode;
mod decode;

pub use encode::encode;
pub use decode::{decode, DecodeError};
