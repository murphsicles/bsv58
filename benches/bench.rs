use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bsv58::{encode, decode};

fn bench_encode(c: &mut Criterion) {
    let data = include_bytes!("../tests/data/txid.bin");  // Add a 32-byte test file
    c.bench_function("bsv58_encode_simd", |b| b.iter(|| encode(black_box(data))));
}

fn bench_decode(c: &mut Criterion) {
    let encoded = encode(data);
    c.bench_function("bsv58_decode_simd", |b| b.iter(|| decode(black_box(&encoded), true).unwrap()));
}

// Add scalar vs SIMD if curious
criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
