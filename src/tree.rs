use crate::{abacussummit::uncompressed::CP32, mirror_select::mirror_select_nth_unstable_by};

pub const BUCKET_SIZE: usize = 32;
pub type Index = u32;

pub fn into_tree(data: &mut [[CP32; 3]], idxs: &mut [Index], level: usize) {
    if data.len() <= BUCKET_SIZE {
        return;
    }

    // Do current level
    let median = data.len() / 2;
    let level_dim = level % 3;
    mirror_select_nth_unstable_by(data, idxs, median, |a, b| unsafe {
        a.get_unchecked(level_dim)
            .partial_cmp(b.get_unchecked(level_dim))
            .unwrap() // this better be a notnan...
    });

    // Get left and right data and indices, sans median
    let (left_data, median_and_right_data) = data.split_at_mut(median);
    let (left_idxs, median_and_right_idxs) = idxs.split_at_mut(median);
    let right_data = &mut median_and_right_data[1..];
    let right_idxs = &mut median_and_right_idxs[1..];

    // Do left and right, recursively
    into_tree(left_data, left_idxs, level + 1);
    into_tree(right_data, right_idxs, level + 1);
}

pub fn nearest_one(
    data: &[[CP32; 3]],
    data_start: *const [CP32; 3],
    query: &[f32; 3],
    level: usize,
    mut best: usize,
    mut best_dist_sq: f32,
) -> (f32, usize) {
    if data.len() <= BUCKET_SIZE {
        for d in data {
            let dist_sq = squared_euclidean(d, query);
            if dist_sq <= best_dist_sq {
                best_dist_sq = dist_sq;
                let dptr = d as *const [CP32; 3];
                best = unsafe { dptr.offset_from(data_start) as usize };
            }
        }
        return (best_dist_sq, best);
    }

    // Do current level
    let median = data.len() / 2;
    let level_dim = level % 3;
    let stem = unsafe { data.get_unchecked(median) };

    // Check stem if necessary
    let dx = unsafe { stem.get_unchecked(level_dim).decompress() - query.get_unchecked(level_dim) };

    // Determine which direction
    let go_left = dx > 0.0;

    // Get left and query left if necessary
    let (left_data, median_and_right_data) = data.split_at(median);
    if go_left {
        let (left_best_dist_sq, left_best) =
            nearest_one(left_data, data_start, query, level + 1, best, best_dist_sq);
        if left_best_dist_sq < best_dist_sq {
            best = left_best;
            best_dist_sq = left_best_dist_sq;
        }
    } else {
        // Get right and query right if necessary
        let right_data = &median_and_right_data[1..];
        let (right_best_dist_sq, right_best) =
            nearest_one(right_data, data_start, query, level + 1, best, best_dist_sq);
        if right_best_dist_sq < best_dist_sq {
            best = right_best;
            best_dist_sq = right_best_dist_sq;
        }
    }

    // Check whether we have to check stem or other dim
    let check_stem_and_other_dim = best_dist_sq >= dx * dx;

    if check_stem_and_other_dim {
        // Check stem
        let dist_sq = squared_euclidean(stem, query);
        if dist_sq <= best_dist_sq {
            best_dist_sq = dist_sq;
            let dptr = stem as *const [CP32; 3];
            best = unsafe { dptr.offset_from(data_start) as usize };
        }

        // Check other dim
        // Invert logic
        if !go_left {
            let (left_best_dist_sq, left_best) =
                nearest_one(left_data, data_start, query, level + 1, best, best_dist_sq);
            if left_best_dist_sq < best_dist_sq {
                best = left_best;
                best_dist_sq = left_best_dist_sq;
            }
        } else {
            // Get right and query right if necessary
            let right_data = &median_and_right_data[1..];
            let (right_best_dist_sq, right_best) =
                nearest_one(right_data, data_start, query, level + 1, best, best_dist_sq);
            if right_best_dist_sq < best_dist_sq {
                best = right_best;
                best_dist_sq = right_best_dist_sq;
            }
        }
    }

    (best_dist_sq, best)
}

use std::collections::BinaryHeap;

#[derive(Debug, PartialOrd, PartialEq, Clone, Copy)]
pub struct F32(pub f32);

impl Eq for F32 {}

impl Ord for F32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).expect("you likely had a nan")
    }
}

pub fn nearest_k(
    data: &[[CP32; 3]],
    data_start: *const [CP32; 3],
    query: &[f32; 3],
    level: usize,
    k: usize,
    mut bests: BinaryHeap<(F32, usize)>,
) -> BinaryHeap<(F32, usize)> {
    if data.len() <= BUCKET_SIZE {
        for d in data {
            let dist_sq = F32(squared_euclidean(d, query));
            if bests.len() < k || dist_sq < bests.peek().unwrap().0 {
                if bests.len() == k {
                    bests.pop();
                }
                bests.push((dist_sq, unsafe {
                    (d as *const [CP32; 3]).offset_from(data_start) as usize
                }));
            }
        }
        return bests;
    }

    let median = data.len() / 2;
    let level_dim = level % 3;
    let stem = unsafe { data.get_unchecked(median) };

    let dx = unsafe { stem.get_unchecked(level_dim).decompress() - query.get_unchecked(level_dim) };
    let go_left = dx > 0.0;

    let (left_data, median_and_right_data) = data.split_at(median);
    if go_left {
        bests = nearest_k(left_data, data_start, query, level + 1, k, bests);
    } else {
        let right_data = &median_and_right_data[1..];
        bests = nearest_k(right_data, data_start, query, level + 1, k, bests);
    }

    // Check whether we have to check stem or other dim
    // 1) if bests is not full (regardless of current bests)
    // 2) if plane is closer than kth best
    let check_stem_and_other_dim = if bests.len() < k {
        true
    } else {
        bests
            .peek()
            .map_or(true, |&(dist_sq, _)| dist_sq >= F32(dx * dx))
    };

    if check_stem_and_other_dim {
        let dist_sq = F32(squared_euclidean(stem, query));
        if bests.len() < k || dist_sq < bests.peek().unwrap().0 {
            if bests.len() == k {
                bests.pop();
            }
            bests.push((dist_sq, unsafe {
                (stem as *const [CP32; 3]).offset_from(data_start) as usize
            }));
        }

        if !go_left {
            bests = nearest_k(left_data, data_start, query, level + 1, k, bests);
        } else {
            let right_data = &median_and_right_data[1..];
            bests = nearest_k(right_data, data_start, query, level + 1, k, bests);
        }
    }

    bests
}

pub fn squared_euclidean(a: &[CP32; 3], q: &[f32; 3]) -> f32 {
    unsafe {
        let dx = a.get_unchecked(0).decompress() - q.get_unchecked(0);
        let dy = a.get_unchecked(1).decompress() - q.get_unchecked(1);
        let dz = a.get_unchecked(2).decompress() - q.get_unchecked(2);

        dx * dx + dy * dy + dz * dz
    }
}

pub mod ffi {
    use super::{into_tree, Index, CP32};

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct QueryNearest {
        pub dist_sq: f32,
        pub idx_within: u64,
    }

    #[no_mangle]
    pub unsafe extern "C" fn construct_compressed_tree(
        flat_data_ptr: *mut CP32,
        num_points: u64,
        idxs_ptr: *mut Index,
    ) {
        let flat_data: &mut [CP32] =
            unsafe { std::slice::from_raw_parts_mut(flat_data_ptr, 3 * num_points as usize) };
        let data: &mut [[CP32; 3]] = bytemuck::cast_slice_mut(flat_data);
        let idxs: &mut [Index] =
            unsafe { std::slice::from_raw_parts_mut(idxs_ptr, num_points as usize) };
        into_tree(data, idxs, 0);
    }

    #[no_mangle]
    pub unsafe extern "C" fn query_compressed_nearest(
        flat_data_ptr: *const CP32,
        num_points: u64,
        flat_query_ptr: *const f32,
        num_queries: u64,
    ) -> *const QueryNearest {
        let flat_data: &[CP32] = std::slice::from_raw_parts(flat_data_ptr, 3 * num_points as usize);
        let data: &[[CP32; 3]] = bytemuck::cast_slice(flat_data);

        let flat_queries: &[f32] =
            std::slice::from_raw_parts(flat_query_ptr, 3 * num_queries as usize);
        let queries: &[[f32; 3]] = bytemuck::cast_slice(flat_queries);

        let results: Vec<QueryNearest> = queries
            .iter()
            .map(|q| {
                let (dist_sq, idx_within) =
                    super::nearest_one(data, data.as_ptr(), q, 0, 0, f32::MAX);
                let idx_within = idx_within as u64;
                QueryNearest {
                    dist_sq,
                    idx_within,
                }
            })
            .collect();
        Box::leak(results.into_boxed_slice()).as_ptr()
    }

    #[no_mangle]
    #[cfg(feature = "parallel")]
    pub unsafe extern "C" fn query_compressed_nearest_parallel(
        flat_data_ptr: *const CP32,
        num_points: u64,
        flat_query_ptr: *const f32,
        num_queries: u64,
    ) -> *const QueryNearest {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::IntoParallelRefIterator;

        let flat_data: &[CP32] = std::slice::from_raw_parts(flat_data_ptr, 3 * num_points as usize);
        let data: &[[CP32; 3]] = bytemuck::cast_slice(flat_data);

        let flat_queries: &[f32] =
            std::slice::from_raw_parts(flat_query_ptr, 3 * num_queries as usize);
        let queries: &[[f32; 3]] = bytemuck::cast_slice(flat_queries);

        let results: Vec<QueryNearest> = queries
            .par_iter()
            .map(|q| {
                let (dist_sq, idx_within) =
                    super::nearest_one(data, data.as_ptr(), q, 0, 0, f32::MAX);
                let idx_within = idx_within as u64;
                QueryNearest {
                    dist_sq,
                    idx_within,
                }
            })
            .collect();
        Box::leak(results.into_boxed_slice()).as_ptr()
    }
}

#[test]
fn test_into_tree() {
    const DATA: usize = BUCKET_SIZE * 3;
    let ref mut data: Vec<[CP32; 3]> = (0..DATA)
        .map(|_| [CP32::compress(rand::random::<f32>() - 0.5, 0.0); 3])
        .collect();
    let ref mut idxs: Vec<Index> = (0..DATA as Index).collect();

    into_tree(data, idxs, 0);
}

#[test]
fn test_into_tree_query() {
    const DATA: usize = BUCKET_SIZE * 3;
    let ref mut data: Vec<[CP32; 3]> = (0..DATA)
        .map(|_| [CP32::compress(rand::random::<f32>() - 0.5, 0.0); 3])
        .collect();
    let ref mut idxs: Vec<Index> = (0..DATA as Index).collect();

    into_tree(data, idxs, 0);
    println!("{data:#?}");
    println!("{idxs:#?}");

    let query = [-0.1; 3];
    let (best_dist_sq, best) = nearest_one(data, data.as_ptr(), &query, 0, 0, f32::INFINITY);
    println!("query: {query:?}");
    println!("best: {:?} -> {best_dist_sq}", data[best]);
}
