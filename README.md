# ⚡ bsv58 — Ultra-fast Base58 codec for Bitcoin SV

**Bitcoin alphabet. Leading-zero '1's. Optional double-SHA256 checksum decode.**
**No generics. No multi-alphabet overhead. Just raw speed.**

---

## 📦 Two Versions

| Branch | Language | Version | Use Case |
|--------|----------|---------|----------|
| **`main`** | **Pure Zeta** | **v0.2.0+** | For Zeta projects (nour, etc.) |
| **`rust`** | Rust | v0.1.1+ | For Rust projects (published to crates.io) |

### Main branch (this branch) — Zeta

This branch contains the pure Zeta implementation of bsv58. It is intended for use
with the [Zeta compiler](https://github.com/murphsicles/zeta) and provides the same
`encode()` / `decode()` / `decode_full()` API surface.

**Features:**
- Pure scalar Zeta implementation (no SIMD — Zeta doesn't have SIMD intrinsics yet)
- Bitcoin alphabet only (hardcoded for performance)
- Leading zero handling as `'1'`s
- Optional BSV-style checksum validation via `decode_full` *(requires sha256 package — coming soon)*

### Rust branch (`rust`) — crates.io

The Rust implementation lives on the [`rust` branch](https://github.com/murphsicles/bsv58/tree/rust).
It includes:
- AVX2 (x86) and NEON (ARM) SIMD acceleration
- Full benchmark suite vs `bs58` and `base58` crates
- Automatic publishing to [crates.io](https://crates.io/crates/bsv58) on version tags

To use the Rust version:
```toml
[dependencies]
bsv58 = { git = "https://github.com/murphsicles/bsv58", branch = "rust" }
# or from crates.io:
# bsv58 = "0.1"
```

---

## 🔧 Usage (Zeta)

```zeta
use bsv58;

fn main() {
    // Encode bytes to Base58
    let encoded: string = bsv58::encode([0x00, 0x01, 0x02]);
    // encoded == "2DnW"

    // Decode Base58 to bytes
    let decoded: []u8 = bsv58::decode("2DnW");
    // decoded == [0x00, 0x01, 0x02]
}
```

## 📊 Performance

Rust version (SIMD): **<4 cycles/char** decode (AVX2), **<5 cycles/byte** encode.
Zeta version: Scalar only, suitable for moderate-throughput BSV applications.

See benchmarks in the [`rust` branch](https://github.com/murphsicles/bsv58/tree/rust).

---

## ✅ Tests

```bash
# Zeta version (this branch)
zetac src/bsv58.z

# Rust version (rust branch)
cargo test
```

## 📝 License

MIT
