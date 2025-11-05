use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bsv58::{encode, decode};

fn bench_encode(c: &mut Criterion) {
    let data = b"hello world this is a test payload for BSV base58 encoding, 32 bytes typical for hashes";
    c.bench_function("bsv58_encode", |b| b.iter(|| encode(black_box(data))));
}

fn bench_decode(c: &mut Criterion) {
    let encoded = encode("hello world this is a test payload for BSV base58 encoding, 32 bytes typical for hashes".as_bytes());
    c.bench_function("bsv58_decode", |b| b.iter(|| decode(black_box(&encoded)).unwrap()));
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
