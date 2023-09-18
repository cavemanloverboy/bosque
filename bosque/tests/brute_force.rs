use std::error::Error;

use bosque::{
    abacussummit::uncompressed::CP32,
    tree::{build_tree_with_indices, nearest_one, nearest_one_periodic, Index},
};
use rand::{rngs::ThreadRng, Rng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

const NDATA: usize = 1_000;
const NQUERY: usize = 10_000;

#[test]
fn test_brute_force() -> Result<(), Box<dyn Error>> {
    // Random number generator
    let mut rng = rand::thread_rng();

    // Generate random data, query
    let mut data = Vec::with_capacity(NDATA);
    let mut idxs = Vec::with_capacity(NDATA);
    let mut query = Vec::with_capacity(NQUERY);
    for i in 0..NDATA {
        data.push(random_cp32(&mut rng));
        idxs.push(i as Index);
    }
    for _ in 0..NQUERY {
        query.push(random_point(&mut rng));
    }

    // Construct tree
    build_tree_with_indices(&mut data, &mut idxs);

    // Query tree
    let results: Vec<_> = query.par_iter().map(|q| nearest_one(&data, q)).collect();

    // Brute force check results
    query
        .par_iter()
        .enumerate()
        .for_each(|(i, q)| assert_eq!(results[i], brute_force(q, &data)));

    // Query tree periodic
    let lo = -0.5;
    let hi = 0.5;
    let results: Vec<_> = query
        .par_iter()
        .take(NQUERY / 10)
        .map(|q| nearest_one_periodic(&data, q, lo, hi))
        .collect();

    // Brute force check periodic results
    query
        .iter()
        .take(NQUERY / 10)
        .enumerate()
        .for_each(|(i, q)| assert_eq!(results[i], brute_force_periodic(q, &data), "failed on {i}"));

    Ok(())
}

fn random_point<const D: usize>(rng: &mut ThreadRng) -> [f32; D] {
    [(); D].map(|_| rng.gen::<f32>() - 0.5)
}

fn random_cp32<const D: usize>(rng: &mut ThreadRng) -> [CP32; D] {
    [(); D].map(|_| CP32::compress(rng.gen::<f32>() - 0.5, 0.0))
}

fn brute_force(q: &[f32; 3], data: &[[CP32; 3]]) -> (f32, usize) {
    let mut best_dist_sq = f32::MAX;
    let mut best = usize::MAX;
    for (d, i) in data.iter().zip(0..) {
        let dist_sq = squared_euclidean(d, q);

        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best = i;
        }
    }

    if cfg!(feature = "sqrt-dist") {
        best_dist_sq = best_dist_sq.sqrt();
    }

    (best_dist_sq, best)
}

fn brute_force_periodic(q: &[f32; 3], data: &[[CP32; 3]]) -> (f32, usize) {
    let mut best_dist_sq = f32::MAX;
    let mut best = usize::MAX;
    for (d, i) in data.iter().zip(0..) {
        for img in 0..3_i32.pow(3) {
            let image = [0_i32, 1, 2]
                .map(|idx| q[idx as usize] + (((img / 3_i32.pow(idx as u32)) % 3) - 1) as f32);

            let dist_sq = squared_euclidean(d, &image);

            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best = i;
            }
        }
    }

    if cfg!(feature = "sqrt-dist") {
        best_dist_sq = best_dist_sq.sqrt();
    }
    (best_dist_sq, best)
}

fn squared_euclidean(d: &[CP32; 3], q: &[f32; 3]) -> f32 {
    let dx = d[0].decompress() - q[0];
    let dy = d[1].decompress() - q[1];
    let dz = d[2].decompress() - q[2];

    dx * dx + dy * dy + dz * dz
}
