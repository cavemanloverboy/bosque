use std::error::Error;

use bosque::{
    abacussummit::uncompressed::CP32,
    tree::{into_tree, nearest_one, squared_euclidean, Index},
};
use rand::{rngs::ThreadRng, Rng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

const NDATA: usize = 10_000;
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
    into_tree(&mut data, &mut idxs, 0);

    // Query tree
    let results: Vec<_> = query
        .par_iter()
        .map(|q| nearest_one(&data, data.as_ptr(), q, 0, 0, f32::MAX))
        .collect();

    // Brute force check results
    query
        .par_iter()
        .enumerate()
        .for_each(|(i, q)| assert_eq!(results[i], brute_force(q, &data)));

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
        let dist = squared_euclidean(d, q);

        if dist < best_dist_sq {
            best_dist_sq = dist;
            best = i;
        }
    }

    (best_dist_sq, best)
}
