use std::error::Error;

use bosque::{
    abacussummit::uncompressed::CP32,
    tree::{into_tree, nearest_one, nearest_one_periodic, squared_euclidean, Index},
};
use rand::{rngs::ThreadRng, Rng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

const NDATA: usize = 10_000;
const NQUERY: usize = 100_000;

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
    into_tree(&mut data, &mut idxs, 0);

    // Query tree
    let results: Vec<_> = query
        .par_iter()
        .map(|q| unsafe { nearest_one(&data, data.as_ptr(), q, 0, 0, f32::MAX) })
        .collect();

    // Brute force check results
    query
        .par_iter()
        .enumerate()
        .for_each(|(i, q)| assert_eq!(results[i], brute_force(q, &data)));

    // Query tree periodic
    let results: Vec<_> = query
        .par_iter()
        .take(NQUERY / 100)
        .map(|q| unsafe { nearest_one_periodic(&data, data.as_ptr(), q, 0) })
        .collect();

    // Brute force check periodic results
    query
        .iter()
        .take(NQUERY / 100)
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

    (best_dist_sq, best)
}
