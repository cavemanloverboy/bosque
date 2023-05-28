use std::{collections::BinaryHeap, error::Error};

use bosque::{
    abacussummit::uncompressed::CP32,
    tree::{into_tree, nearest_k, squared_euclidean, Index, F32},
};
use rand::{rngs::ThreadRng, Rng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

const NDATA: usize = 10_000;
const NQUERY: usize = 10_000;

const K: usize = 32;
#[test]
fn test_brute_force_k() -> Result<(), Box<dyn Error>> {
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
    let results: Vec<[(f32, usize); K]> = query
        .par_iter()
        .map(|q| unsafe {
            // let bh = BinaryHeap::from([(F32(f32::MAX), 0)]);
            let bh = BinaryHeap::default();
            core::mem::transmute::<[(F32, usize); K], [(f32, usize); K]>(
                nearest_k(&data, data.as_ptr(), q, 0, K, bh)
                    .into_sorted_vec()
                    .try_into()
                    .unwrap(),
            )
        })
        .collect();

    // Brute force check results
    query
        .iter()
        .enumerate()
        .for_each(|(i, q)| assert_eq!(results[i], brute_force(q, &data), "failed on {i}"));

    println!("done with {NQUERY} queries");

    Ok(())
}

fn random_point<const D: usize>(rng: &mut ThreadRng) -> [f32; D] {
    [(); D].map(|_| rng.gen::<f32>() - 0.5)
}

fn random_cp32<const D: usize>(rng: &mut ThreadRng) -> [CP32; D] {
    [(); D].map(|_| CP32::compress(rng.gen::<f32>() - 0.5, 0.0))
}

fn brute_force(q: &[f32; 3], data: &[[CP32; 3]]) -> [(f32, usize); K] {
    let mut best = [(f32::MAX, usize::MAX); K];
    for (d, i) in data.iter().zip(0..) {
        let dist = squared_euclidean(d, q);

        if dist < best[K - 1].0 {
            best[K - 1] = (dist, i);

            best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
    }

    best
}
