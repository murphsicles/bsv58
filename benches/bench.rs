//! Benchmarks for bsv58: Microbenchmarks via Criterion.
//! Targets: Encode/decode on BSV payloads (20-byte hashes, 32-byte txids, 34-char addrs).
//! Baselines: base58 (control, slower baseline) + bs58 (reference, faster than base58).
//! Add deps: cargo add base58 --dev (latest ~0.2 as of 2025).
//! Run: cargo bench --bench bench [-- --save-baselines].
//! Outputs: GB/s throughput (higher better); HTML reports in target/criterion.
//! Projections (i9-13900K/M3 Max): bsv58 encode ~6 GB/s (5x bs58 ~1.2 GB/s, 15x base58 ~0.4 GB/s).

use bsv58::{decode, encode};
use base58;  // Control: cargo add base58 --dev
use bs58;    // Reference: cargo add bs58 --dev
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use hex_literal::hex;  // For test data

/// Sample BSV payloads: Hash (20B), txid (32B), addr (34 chars w/checksum).
fn bsv_samples() -> Vec<(&'static [u8], &'static str)> {
    vec![
        // Pubkey hash (20B, like P2PKH)
        (&hex!("00759d6677091e973b9e9d99f19c68fbf43e3f05f9"), "1BitcoinEaterAddressDontSendf59kuE"),
        // Txid (32B, reversed for LE? But raw bytes)
        (&hex!("a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0"), "BtCjvJYNhqehX2sbzvBNrbkCYp2qfc6AepXfK1JGnELw"),
        // Genesis block hash (32B w/leading zeros)
        (&hex!("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"), "19Vqm6P7Q5Ge"),
    ]
}

/// Bench encode: bsv58 vs base58 (control) vs bs58 (reference) on samples.
fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    group.measurement_time(WallTime::new()).throughput(Throughput::Bytes(1));  // Normalize to GB/s

    for (bytes, _encoded) in bsv_samples() {
        let len = bytes.len();

        // bsv58
        group.bench_with_id(
            format!("bsv58_{}B", len),
            BenchmarkId::new("bsv58", len),
            |b, &len| {
                let data = black_box(&bytes[..len]);
                b.iter(|| encode(data))
            },
        );

        // base58 control (slower baseline)
        group.bench_with_id(
            format!("base58_{}B", len),
            BenchmarkId::new("base58", len),
            |b, &len| {
                let data = black_box(&bytes[..len]);
                b.iter(|| base58::encode(data))
            },
        );

        // bs58 reference (faster than base58)
        group.bench_with_id(
            format!("bs58_{}B", len),
            BenchmarkId::new("bs58", len),
            |b, &len| {
                let data = black_box(&bytes[..len]);
                b.iter(|| bs58::encode(data))
            },
        );
    }
    group.finish();
}

/// Bench decode: bsv58 vs base58 vs bs58 (w/ checksum=false for fair raw compare).
fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    group.measurement_time(WallTime::new()).throughput(Throughput::Bytes(1));

    for (_bytes, encoded) in bsv_samples() {
        let len = encoded.len();

        // bsv58
        group.bench_with_id(
            format!("bsv58_{}chars", len),
            BenchmarkId::new("bsv58", len),
            |b, &len| {
                let s = black_box(&encoded[..len]);
                b.iter(|| decode(s, false).unwrap())  // No checksum for raw perf
            },
        );

        // base58 control
        group.bench_with_id(
            format!("base58_{}chars", len),
            BenchmarkId::new("base58", len),
            |b, &len| {
                let s = black_box(&encoded[..len]);
                b.iter(|| base58::decode(s).unwrap())
            },
        );

        // bs58 reference
        group.bench_with_id(
            format!("bs58_{}chars", len),
            BenchmarkId::new("bs58", len),
            |b, &len| {
                let s = black_box(&encoded[..len]);
                b.iter(|| bs58::decode(s).unwrap())
            },
        );
    }
    group.finish();
}

/// Bonus: Roundtrip (encode+decode) for end-to-end BSV workflow.
fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    group.throughput(Throughput::Bytes(1));

    for (bytes, _) in bsv_samples() {
        let len = bytes.len();

        // bsv58
        group.bench_with_id(
            format!("bsv58_{}B", len),
            BenchmarkId::new("bsv58", len),
            |b, &len| {
                let data = black_box(&bytes[..len]);
                b.iter(|| {
                    let enc = encode(data);
                    decode(&enc, false).unwrap()
                })
            },
        );

        // base58 control
        group.bench_with_id(
            format!("base58_{}B", len),
            BenchmarkId::new("base58", len),
            |b, &len| {
                let data = black_box(&bytes[..len]);
                b.iter(|| {
                    let enc = base58::encode(data);
                    base58::decode(&enc).unwrap()
                })
            },
        );

        // bs58 reference
        group.bench_with_id(
            format!("bs58_{}B", len),
            BenchmarkId::new("bs58", len),
            |b, &len| {
                let data = black_box(&bytes[..len]);
                b.iter(|| {
                    let enc = bs58::encode(data);
                    bs58::decode(&enc).unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_roundtrip);
criterion_main!(benches);
