use bosque::{
    abacussummit::uncompressed::CP32,
    tree::{self, Index},
};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

fn criterion_benchmark(c: &mut Criterion) {
    // const DATA: usize = 512 * 512 * 512;
    const DATA: usize = 100_000;
    const QUERIES: usize = 1_000_000;

    let mut data: Vec<[CP32; 3]> = (0..DATA)
        .map(|_| [CP32::compress(rand::random::<f32>() - 0.5, 0.0); 3])
        .collect();
    let mut data_f32: Vec<[f32; 3]> = (0..DATA)
        .map(|_| [rand::random::<f32>() - 0.5; 3])
        .collect();
    let mut idxs: Vec<Index> = (0..DATA as Index).collect();
    let mut idxs_f32: Vec<Index> = (0..DATA as Index).collect();

    let mut build_group = c.benchmark_group("build");
    let g = build_group.sample_size(10);
    g.bench_function("build", |b| {
        b.iter_batched_ref(
            || (data.clone(), idxs.clone()),
            |(ref mut d, ref mut i)| tree::build_tree_with_indices(black_box(d), black_box(i)),
            BatchSize::LargeInput,
        )
    });
    g.bench_function("build_f32", |b| {
        b.iter_batched_ref(
            || (data_f32.clone(), idxs_f32.clone()),
            |(ref mut d, ref mut i)| tree::build_tree_with_indices(black_box(d), black_box(i)),
            BatchSize::LargeInput,
        )
    });
    build_group.finish();

    tree::build_tree_with_indices(&mut data, &mut idxs);
    tree::build_tree_with_indices(&mut data_f32, &mut idxs_f32);

    let query = [-0.1; 3];
    let mut query_nearest_group = c.benchmark_group("query_nearest");
    let g = query_nearest_group.sample_size(10);
    g.bench_function("query_const", |b| {
        b.iter(|| {
            for _ in 0..QUERIES {
                black_box(tree::nearest_one(black_box(&data), black_box(&query)));
            }
        });
    });

    g.bench_function("query_constf32", |b| {
        b.iter(|| {
            for _ in 0..QUERIES {
                black_box(tree::nearest_one(black_box(&data_f32), black_box(&query)));
            }
        });
    });

    let queries: Vec<[f32; 3]> = (0..QUERIES)
        .map(|_| [rand::random::<f32>() - 0.5; 3])
        .collect();
    g.bench_function("query", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(tree::nearest_one(black_box(&data), black_box(q)));
            }
        });
    });
    g.bench_function("query_par", |b| {
        b.iter(|| {
            queries.par_iter().for_each(|q| {
                black_box(tree::nearest_one(black_box(&data), black_box(q)));
            })
        });
    });

    g.bench_function("queryf32", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(tree::nearest_one(black_box(&data_f32), black_box(q)));
            }
        });
    });
    g.bench_function("queryf32_par", |b| {
        b.iter(|| {
            queries.par_iter().for_each(|q| {
                black_box(tree::nearest_one(black_box(&data_f32), black_box(q)));
            })
        });
    });

    g.bench_function("query_periodic", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(tree::nearest_one_periodic(
                    black_box(&data),
                    black_box(q),
                    -0.5,
                    0.5,
                ));
            }
        });
    });
    g.bench_function("query_par_periodic", |b| {
        b.iter(|| {
            queries.par_iter().for_each(|q| {
                black_box(tree::nearest_one_periodic(
                    black_box(&data),
                    black_box(q),
                    -0.5,
                    0.5,
                ));
            })
        });
    });

    g.bench_function("queryf32_periodic", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(tree::nearest_one_periodic(
                    black_box(&data_f32),
                    black_box(q),
                    -0.5,
                    0.5,
                ));
            }
        });
    });

    g.bench_function("queryf32_par_periodic", |b| {
        b.iter(|| {
            queries.par_iter().for_each(|q| {
                black_box(tree::nearest_one_periodic(
                    black_box(&data_f32),
                    black_box(q),
                    -0.5,
                    0.5,
                ));
            })
        });
    });
    query_nearest_group.finish();

    let mut query_k_group = c.benchmark_group("query_k");
    let g = query_k_group.sample_size(10);

    g.bench_function("query_k", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(tree::nearest_k(black_box(&data), black_box(q), 128));
            }
        });
    });

    g.bench_function("query_k_par", |b| {
        b.iter(|| {
            queries.par_iter().for_each(|q| {
                black_box(tree::nearest_k(black_box(&data), black_box(q), 128));
            })
        });
    });

    g.bench_function("query_k_f32", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(tree::nearest_k(black_box(&data_f32), black_box(q), 128));
            }
        });
    });

    g.bench_function("query_k_f32_par", |b| {
        b.iter(|| {
            queries.par_iter().for_each(|q| {
                black_box(tree::nearest_k(black_box(&data_f32), black_box(q), 128));
            })
        });
    });
    query_k_group.finish();
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
