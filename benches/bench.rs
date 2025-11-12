//! Benchmarks for bsv58: Micro + macro on BSV payloads (5B-1MB).
//! Baselines: base58 (slow) + bs58 (fast).
//! Run: cargo bench --bench bench [-- --save-baselines --open].
//! Outputs: MB/s (higher better); HTML in target/criterion.
//! Projections (i9/M3): bsv58 ~500 MB/s encode/decode (4x+ bs58 ~120 MB/s on 1KB+).
use base58::{FromBase58, ToBase58};
use bs58::{decode as bs58_decode, encode as bs58_encode};
use bsv58::{decode, decode_full, encode};
use criterion::{black_box, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hex_literal::hex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Duration;

/// Samples: Small BSV (bytes, encoded); large random.
fn samples() -> Vec<(Vec<u8>, String)> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let small = vec![
        // Hello (5B)
        (b"hello".to_vec(), "Cn8eVZg".to_string()),
        // Addr (21B)
        (
            hex!("00759d6677091e973b9e9d99f19c68fbf43e3f05f9").to_vec(),
            "12e3A9pcaDSMffCV3iBuhipLAGczU".to_string(),
        ),
        // Txid (32B)
        (
            hex!("a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0").to_vec(),
            "BtCjvJYNhqehX2sbzvBNrbkCYp2qfc6AepXfK1JGnELw".to_string(),
        ),
        // Genesis (32B zeros)
        (
            hex!("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f").to_vec(),
            "111114VYJtj3yEDffZem7N3PkK563wkLZZ8RjKzcfY".to_string(),
        ),
    ];
    let large = vec![
        (1024, || (0..1024).map(|_| rng.gen()).collect()),
        (1_048_576, || (0..1_048_576).map(|_| rng.gen()).collect()), // 1MB
    ].into_iter().map(|(sz, gen)| {
        let bytes = gen();
        let encoded = encode(&bytes);
        (bytes, encoded)
    }).collect::<Vec<_>>();
    small.into_iter().chain(large).collect()
}

/// Checksum addrs (34 chars -> 21B).
fn checksum_addrs() -> Vec<&'static str> {
    vec!["1BitcoinEaterAddressDontSendf59kuE"]
}

/// Encode: Groups by input bytes.
fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    group.sample_size(200).measurement_time(Duration::from_secs(2));
    for (bytes, _) in samples() {
        let size = bytes.len();
        group.throughput(Throughput::Bytes(size as u64));
        let input = black_box(&bytes);
        group.bench_function(BenchmarkId::new("bsv58", format!("{}B", size)), |b| {
            b.iter(|| encode(black_box(input)));
        });
        group.bench_function(BenchmarkId::new("base58", format!("{}B", size)), |b| {
            b.iter(|| black_box(input.to_base58()));
        });
        group.bench_function(BenchmarkId::new("bs58", format!("{}B", size)), |b| {
            b.iter(|| black_box(bs58_encode(black_box(input)).into_string()));
        });
    }
    group.finish();
}

/// Decode raw: Groups by input chars; throughput approx bytes out (~0.73 * chars).
fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    group.sample_size(200).measurement_time(Duration::from_secs(2));
    for (_, encoded) in samples() {
        let size = encoded.len();
        let out_bytes = (size as f64 * 0.733).ceil() as u64;
        group.throughput(Throughput::Bytes(out_bytes));
        let s = black_box(&encoded);
        group.bench_function(BenchmarkId::new("bsv58", format!("{}chars", size)), |b| {
            b.iter(|| black_box(decode(black_box(s))).unwrap());
        });
        group.bench_function(BenchmarkId::new("base58", format!("{}chars", size)), |b| {
            b.iter(|| black_box(s.from_base58().unwrap()));
        });
        group.bench_function(BenchmarkId::new("bs58", format!("{}chars", size)), |b| {
            b.iter(|| black_box(bs58_decode(black_box(s)).into_vec().unwrap()));
        });
    }
    group.finish();
}

/// Decode checksum: bsv58 only (sha2); small only (noisy).
fn bench_decode_checksum(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_checksum");
    group.sample_size(100).measurement_time(Duration::from_secs(1));
    for addr in checksum_addrs() {
        let size = addr.len();
        let out_bytes = 21u64; // Fixed payload
        group.throughput(Throughput::Bytes(out_bytes));
        let s = black_box(addr);
        group.bench_function(BenchmarkId::new("bsv58", format!("{}chars", size)), |b| {
            b.iter(|| black_box(decode_full(black_box(s), true)).unwrap());
        });
    }
    group.finish();
}

/// Roundtrip: encode+decode raw; throughput bytes in.
fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    group.sample_size(200).measurement_time(Duration::from_secs(2));
    for (bytes, encoded) in samples() {
        let size = bytes.len();
        group.throughput(Throughput::Bytes(size as u64));
        let input = black_box(&bytes);
        let s = black_box(&encoded);
        group.bench_function(BenchmarkId::new("bsv58", format!("{}B", size)), |b| {
            b.iter(|| {
                let enc = encode(black_box(input));
                let _ = decode(&enc).unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("base58", format!("{}B", size)), |b| {
            b.iter(|| {
                let enc = black_box(input.to_base58());
                let _ = enc.as_str().from_base58().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("bs58", format!("{}B", size)), |b| {
            b.iter(|| {
                let enc = bs58_encode(black_box(input)).into_string();
                let _ = bs58_decode(&enc).into_vec().unwrap();
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_decode_checksum, bench_roundtrip);
criterion_main!(benches);
