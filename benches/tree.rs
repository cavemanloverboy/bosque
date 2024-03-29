use bosque::{
    abacussummit::uncompressed::CP32,
    tree::{self, Index},
    treef32,
};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

fn criterion_benchmark(c: &mut Criterion) {
    const DATA: usize = 512 * 512 * 512;
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
            |(ref mut d, ref mut i)| tree::into_tree(black_box(d), black_box(i), 0),
            BatchSize::LargeInput,
        )
    });
    g.bench_function("build_f32", |b| {
        b.iter_batched_ref(
            || (data_f32.clone(), idxs_f32.clone()),
            |(ref mut d, ref mut i)| treef32::into_tree(black_box(d), black_box(i), 0),
            BatchSize::LargeInput,
        )
    });
    build_group.finish();

    tree::into_tree(&mut data, &mut idxs, 0);
    treef32::into_tree(&mut data_f32, &mut idxs_f32, 0);

    let query = [-0.1; 3];
    let mut query_nearest_group = c.benchmark_group("query_nearest");
    let g = query_nearest_group.sample_size(10);
    g.bench_function("query_const", |b| {
        b.iter(|| unsafe {
            for _ in 0..QUERIES {
                black_box(tree::nearest_one(
                    black_box(&data),
                    data.as_ptr(),
                    black_box(&query),
                    0,
                    0,
                    f32::INFINITY,
                ));
            }
        });
    });

    g.bench_function("query_constf32", |b| {
        b.iter(|| unsafe {
            for _ in 0..QUERIES {
                black_box(treef32::nearest_one(
                    black_box(&data_f32),
                    data_f32.as_ptr(),
                    black_box(&query),
                    0,
                    0,
                    f32::INFINITY,
                ));
            }
        });
    });

    let queries: Vec<[f32; 3]> = (0..QUERIES)
        .map(|_| [rand::random::<f32>() - 0.5; 3])
        .collect();
    g.bench_function("query", |b| {
        b.iter(|| unsafe {
            for q in &queries {
                black_box(tree::nearest_one(
                    black_box(&data),
                    data.as_ptr(),
                    black_box(q),
                    0,
                    0,
                    f32::INFINITY,
                ));
            }
        });
    });
    g.bench_function("query_par", |b| {
        b.iter(|| unsafe {
            queries.par_iter().for_each(|q| {
                black_box(tree::nearest_one(
                    black_box(&data),
                    data.as_ptr(),
                    black_box(q),
                    0,
                    0,
                    f32::INFINITY,
                ));
            })
        });
    });

    g.bench_function("queryf32", |b| {
        b.iter(|| unsafe {
            for q in &queries {
                black_box(treef32::nearest_one(
                    black_box(&data_f32),
                    data_f32.as_ptr(),
                    black_box(q),
                    0,
                    0,
                    f32::INFINITY,
                ));
            }
        });
    });
    g.bench_function("queryf32_par", |b| {
        b.iter(|| unsafe {
            queries.par_iter().for_each(|q| {
                black_box(treef32::nearest_one(
                    black_box(&data_f32),
                    data_f32.as_ptr(),
                    black_box(q),
                    0,
                    0,
                    f32::INFINITY,
                ));
            })
        });
    });

    g.bench_function("query_periodic", |b| {
        b.iter(|| unsafe {
            for q in &queries {
                black_box(tree::nearest_one_periodic(
                    black_box(&data),
                    data.as_ptr(),
                    black_box(q),
                    0,
                ));
            }
        });
    });
    g.bench_function("query_par_periodic", |b| {
        b.iter(|| unsafe {
            queries.par_iter().for_each(|q| {
                black_box(tree::nearest_one_periodic(
                    black_box(&data),
                    data.as_ptr(),
                    black_box(q),
                    0,
                ));
            })
        });
    });

    g.bench_function("queryf32_periodic", |b| {
        b.iter(|| unsafe {
            for q in &queries {
                black_box(treef32::nearest_one_periodic(
                    black_box(&data_f32),
                    data_f32.as_ptr(),
                    black_box(q),
                    0,
                ));
            }
        });
    });

    g.bench_function("queryf32_par_periodic", |b| {
        b.iter(|| unsafe {
            queries.par_iter().for_each(|q| {
                black_box(treef32::nearest_one_periodic(
                    black_box(&data_f32),
                    data_f32.as_ptr(),
                    black_box(q),
                    0,
                ));
            })
        });
    });
    query_nearest_group.finish();

    let mut query_k_group = c.benchmark_group("query_k");
    let g = query_k_group.sample_size(10);

    g.bench_function("query_k", |b| {
        b.iter(|| unsafe {
            for q in &queries {
                black_box(
                    tree::nearest_k(
                        black_box(&data),
                        data.as_ptr(),
                        black_box(q),
                        0,
                        128,
                        std::collections::BinaryHeap::with_capacity(128),
                    )
                    .into_sorted_vec(),
                );
            }
        });
    });

    g.bench_function("query_k_par", |b| {
        b.iter(|| unsafe {
            queries.par_iter().for_each(|q| {
                black_box(
                    tree::nearest_k(
                        black_box(&data),
                        data.as_ptr(),
                        black_box(q),
                        0,
                        128,
                        std::collections::BinaryHeap::with_capacity(128),
                    )
                    .into_sorted_vec(),
                );
            })
        });
    });

    g.bench_function("query_k_f32", |b| {
        b.iter(|| unsafe {
            for q in &queries {
                black_box(
                    treef32::nearest_k(
                        black_box(&data_f32),
                        data_f32.as_ptr(),
                        black_box(q),
                        0,
                        128,
                        std::collections::BinaryHeap::with_capacity(128),
                    )
                    .into_sorted_vec(),
                );
            }
        });
    });

    g.bench_function("query_k_f32_par", |b| {
        b.iter(|| unsafe {
            queries.par_iter().for_each(|q| {
                black_box(
                    treef32::nearest_k(
                        black_box(&data_f32),
                        black_box(data_f32.as_ptr()),
                        black_box(q),
                        0,
                        black_box(128),
                        black_box(std::collections::BinaryHeap::with_capacity(128)),
                    )
                    .into_sorted_vec(),
                );
            })
        });
    });
    query_k_group.finish();
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
