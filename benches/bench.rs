//! Benchmarks for bsv58: Microbenchmarks via Criterion.
//! Targets: Encode/decode on BSV payloads (5B strings, 21B hashes, 32B txids).
//! Baselines: base58 (slow ref) + bs58 (fast ref).
//! Run: cargo bench --bench bench [-- --save-baselines --open].
//! Outputs: GB/s (higher better); HTML in target/criterion.
//! Projections (i9-13900K/M3 Max): bsv58 encode/decode ~6 GB/s (5x bs58 ~1.2 GB/s).

use base58::{FromBase58, ToBase58};
use bs58;
use bsv58::{decode, decode_full, encode};
use criterion::{
    BenchmarkId, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use hex_literal::hex;
use std::hint::black_box;
use std::time::Duration;

/// Raw BSV samples: (bytes for encode, raw_encoded for decode/roundtrip).
/// Sizes: 5B (hello), 21B (addr payload), 32B (txid), 32B (genesis w/zeros).
fn bsv_samples() -> Vec<(&'static [u8], &'static str)> {
    vec![
        // Hello (5B)
        (b"hello", "Cn8eVZg"),
        // Addr payload raw (21B, no checksum)
        (
            &hex!("00759d6677091e973b9e9d99f19c68fbf43e3f05f9"),
            "12e3A9pcaDSMffCV3iBuhipLAGczU",
        ),
        // Txid (32B)
        (
            &hex!("a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0"),
            "BtCjvJYNhqehX2sbzvBNrbkCYp2qfc6AepXfK1JGnELw",
        ),
        // Genesis (32B w/5 leading zeros)
        (
            &hex!("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"),
            "111114VYJtj3yEDffZem7N3PkK563wkLZZ8RjKzcfY",
        ),
    ]
}

/// Full addr samples for checksum decode (str -> payload bytes).
fn checksum_samples() -> Vec<&'static str> {
    vec![
        // Burn addr (34 chars -> 21B payload)
        "1BitcoinEaterAddressDontSendf59kuE",
    ]
}

/// Bench encode: Per-size groups for GB/s.
fn bench_encode(c: &mut Criterion) {
    for (bytes, _) in bsv_samples() {
        let size = bytes.len();
        let mut group = c.benchmark_group(format!("encode/{}B", size));
        group
            .sample_size(200)
            .measurement_time(Duration::from_secs(1))
            .throughput(Throughput::Bytes(size as u64));

        let input = black_box(bytes);
        group.bench_function(BenchmarkId::new("bsv58", size), |b| {
            b.iter(|| encode(input))
        });
        group.bench_function(BenchmarkId::new("base58", size), |b| {
            b.iter(|| input.to_base58())
        });
        group.bench_function(BenchmarkId::new("bs58", size), |b| {
            b.iter(|| bs58::encode(input).into_string())
        });

        group.finish();
    }
}

/// Bench decode raw (no checksum): Per-size groups, input chars.
fn bench_decode(c: &mut Criterion) {
    for (_, encoded) in bsv_samples() {
        let size = encoded.len(); // ASCII chars ≈ bytes
        let mut group = c.benchmark_group(format!("decode/{}chars", size));
        group
            .sample_size(200)
            .measurement_time(Duration::from_secs(1))
            .throughput(Throughput::Bytes(size as u64));

        let s = black_box(encoded);
        group.bench_function(BenchmarkId::new("bsv58", size), |b| {
            b.iter(|| decode(s).unwrap())
        });
        group.bench_function(BenchmarkId::new("base58", size), |b| {
            b.iter(|| s.from_base58().unwrap())
        });
        group.bench_function(BenchmarkId::new("bs58", size), |b| {
            b.iter(|| bs58::decode(s).into_vec().unwrap())
        });

        group.finish();
    }
}

/// Bench decode w/ BSV checksum validation (sha2 overhead).
fn bench_decode_checksum(c: &mut Criterion) {
    for addr in checksum_samples() {
        let size = addr.len();
        let mut group = c.benchmark_group(format!("decode_checksum/{}chars", size));
        group
            .sample_size(100) // Fewer: sha2 noisy
            .measurement_time(Duration::from_secs(1))
            .throughput(Throughput::Bytes(size as u64));

        let s = black_box(addr);
        group.bench_function(BenchmarkId::new("bsv58", size), |b| {
            b.iter(|| decode_full(s, true).unwrap())
        });
        // Baselines: base58/bs58 lack built-in checksum; skip for now

        group.finish();
    }
}

/// Roundtrip: encode -> decode raw (bytes in/out).
fn bench_roundtrip(c: &mut Criterion) {
    for (bytes, _) in bsv_samples() {
        let size = bytes.len();
        let mut group = c.benchmark_group(format!("roundtrip/{}B", size));
        group
            .sample_size(200)
            .measurement_time(Duration::from_secs(1))
            .throughput(Throughput::Bytes(size as u64));

        let input = black_box(bytes);
        group.bench_function(BenchmarkId::new("bsv58", size), |b| {
            b.iter(|| {
                let enc = encode(input);
                let _ = decode(&enc).unwrap(); // Drop for perf
            })
        });
        group.bench_function(BenchmarkId::new("base58", size), |b| {
            b.iter(|| {
                let enc = input.to_base58();
                let _ = enc.as_str().from_base58().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("bs58", size), |b| {
            b.iter(|| {
                let enc = bs58::encode(input).into_string();
                let _ = bs58::decode(&enc).into_vec().unwrap();
            })
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_decode_checksum,
    bench_roundtrip
);
criterion_main!(benches);
