//! This example is similar to abacus_mock, except it uses rkyv and CPF32.
//!
//! Particles are put into the order expected by a kDTree
use bosque::abacussummit::uncompressed::CP32;
use kiddo::float::distance::squared_euclidean;

fn main() {
    const D: usize = 3;

    // Generate some sample points in [-0.5, 0.5]
    // This is done with some verbosity for clarity
    let root = [0.0; D];
    let root_vel = [2000.0; D];
    let left = [-0.25; D];
    let left_vel = [850.0; D];
    let right = [0.10; D];
    let right_vel = [-20.0; D];

    // Compress points
    let root = std::array::from_fn(|i| CP32::compress(root[i], root_vel[i]));
    let left = std::array::from_fn(|i| CP32::compress(left[i], left_vel[i]));
    let right = std::array::from_fn(|i| CP32::compress(right[i], right_vel[i]));

    // Build a kiddo tree
    let mut tree = kiddo::float::kdtree::KdTree::<CP32, usize, 3, 2, u32>::with_capacity(3);
    tree.add(&root, 0);
    tree.add(&left, 1);
    tree.add(&right, 2);

    // Query
    let query_compressed = [CP32::compress(0.02, 0.0); D];
    let result = tree.nearest_one(&query_compressed, &squared_euclidean);
    println!("query near root: {query_compressed:?} -> {result:?}");

    let query_compressed = [CP32::compress(-0.24, 0.0); D];
    let result = tree.nearest_one(&query_compressed, &squared_euclidean);
    println!("query near left: {query_compressed:?} -> {result:?}");

    let query_compressed = [CP32::compress(0.4, 0.0); D];
    let result = tree.nearest_one(&query_compressed, &squared_euclidean);
    println!("query near right: {query_compressed:?} -> {result:?}");

    println!(
        "query decompressed = {:?}",
        query_compressed.map(|f| CP32::decompress(&f))
    );
}
