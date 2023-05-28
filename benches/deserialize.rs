use bosque::abacussummit::{self, decompress_position};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    const DATA: usize = 512 * 512 * 512;
    let data: Vec<f32> = (0..DATA).map(|_| rand::random::<f32>() - 0.5).collect();
    let bytes: Vec<u8> = data
        .clone()
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let compressed_dwords: Vec<u32> = data
        .into_iter()
        .map(|p| abacussummit::compress(p, 0.0))
        .collect();

    c.bench_function("deser lossless", |b| b.iter(|| deserialize_slice(&bytes)));
    c.bench_function("decompress lossy", |b| {
        b.iter(|| {
            black_box(compressed_dwords.iter())
                .map(decompress_position)
                .collect::<Vec<_>>()
        })
    });
}

fn deserialize_slice(slice: &[u8]) -> Vec<f32> {
    slice
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
