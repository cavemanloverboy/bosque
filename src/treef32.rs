use crate::mirror_select::mirror_select_nth_unstable_by;
pub const BUCKET_SIZE: usize = 32;
pub type Index = u32;

pub fn into_tree(data: &mut [[f32; 3]], idxs: &mut [Index], level: usize) {
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

/// Queries a tree made up of the points in `flat_data_ptr` for the nearest neighbor.
///
/// # Safety
/// This pointer must be valid
pub unsafe fn nearest_one(
    data: &[[f32; 3]],
    data_start: *const [f32; 3],
    query: &[f32; 3],
    level: usize,
    mut best: usize,
    mut best_dist_sq: f32,
) -> (f32, usize) {
    // Deal with bucket
    if data.len() <= BUCKET_SIZE {
        for d in data {
            let dist_sq = squared_euclidean(d, query);
            if dist_sq <= best_dist_sq {
                best_dist_sq = dist_sq;
                let dptr = d as *const [f32; 3];
                best = unsafe { dptr.offset_from(data_start) as usize };
            }
        }
        return (best_dist_sq, best);
    }

    // Get level stem
    let median = data.len() / 2;
    let level_dim = level % 3;
    let stem = unsafe { data.get_unchecked(median) };

    // Check stem if necessary
    let dx = unsafe { stem.get_unchecked(level_dim) - query.get_unchecked(level_dim) };

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
            let dptr = stem as *const [f32; 3];
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

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct F32(pub f32);

impl Eq for F32 {}

impl Ord for F32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).expect("you likely had a nan")
    }
}
impl PartialOrd for F32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

/// Queries a tree made up of the points in `flat_data_ptr` for the k nearest neighbors.
///
/// # Safety
/// This pointer must be valid
pub unsafe fn nearest_k(
    data: &[[f32; 3]],
    data_start: *const [f32; 3],
    query: &[f32; 3],
    level: usize,
    k: usize,
    mut bests: BinaryHeap<(F32, usize)>,
) -> BinaryHeap<(F32, usize)> {
    // Deal with bucket
    if data.len() <= BUCKET_SIZE {
        for d in data {
            let dist_sq = F32(squared_euclidean(d, query));
            if bests.len() < k || dist_sq < bests.peek().unwrap().0 {
                if bests.len() == k {
                    bests.pop();
                }
                bests.push((dist_sq, unsafe {
                    (d as *const [f32; 3]).offset_from(data_start) as usize
                }));
            }
        }
        return bests;
    }

    // Get level stem
    let median = data.len() / 2;
    let level_dim = level % 3;
    let stem = unsafe { data.get_unchecked(median) };

    let dx = unsafe { stem.get_unchecked(level_dim) - query.get_unchecked(level_dim) };
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
        // Check stem
        let dist_sq = F32(squared_euclidean(stem, query));
        if bests.len() < k || dist_sq < bests.peek().unwrap().0 {
            if bests.len() == k {
                bests.pop();
            }
            bests.push((dist_sq, unsafe {
                (stem as *const [f32; 3]).offset_from(data_start) as usize
            }));
        }

        // Check other dim
        // Invert logic
        if !go_left {
            bests = nearest_k(left_data, data_start, query, level + 1, k, bests);
        } else {
            let right_data = &median_and_right_data[1..];
            bests = nearest_k(right_data, data_start, query, level + 1, k, bests);
        }
    }

    bests
}

pub fn squared_euclidean(a: &[f32; 3], q: &[f32; 3]) -> f32 {
    unsafe {
        let dx = a.get_unchecked(0) - q.get_unchecked(0);
        let dy = a.get_unchecked(1) - q.get_unchecked(1);
        let dz = a.get_unchecked(2) - q.get_unchecked(2);

        dx * dx + dy * dy + dz * dz
    }
}

#[test]
fn test_into_tree() {
    const DATA: usize = BUCKET_SIZE * 3;
    let data: &mut Vec<[f32; 3]> = &mut (0..DATA)
        .map(|_| [rand::random::<f32>() - 0.5; 3])
        .collect();
    let idxs: &mut Vec<Index> = &mut (0..DATA as Index).collect();

    into_tree(data, idxs, 0);
}

#[test]
fn test_into_tree_query() {
    const DATA: usize = BUCKET_SIZE * 3;
    let data: &mut Vec<[f32; 3]> = &mut (0..DATA)
        .map(|_| [rand::random::<f32>() - 0.5; 3])
        .collect();
    let idxs: &mut Vec<Index> = &mut (0..DATA as Index).collect();

    into_tree(data, idxs, 0);
    println!("{data:#?}");
    println!("{idxs:#?}");

    let query = [-0.1; 3];
    let (best_dist_sq, best) =
        unsafe { nearest_one(data, data.as_ptr(), &query, 0, 0, f32::INFINITY) };
    println!("query: {query:?}");
    println!("best: {:?} -> {best_dist_sq}", data[best]);
}
