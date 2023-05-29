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
    // on level 0 we get 2^1 = 2 and spawn a thread so 2 total
    // on level 1 we get 2^2 = 4 and spawn a thread on each of 2 threads so 4 total
    // on level 2 we get 2^3 = 8 and spawn a thread on each of 4 threads so 8 total
    // on level 3 we get 2^4 = 16 > 8 -> so sequential.
    let lte_8_threads = 2_usize.pow(1 + level as u32) > 8;
    let small_data = left_data.len() < 25_000;
    let sequential = small_data | lte_8_threads;
    if sequential {
        into_tree(left_data, left_idxs, level + 1);
        into_tree(right_data, right_idxs, level + 1);
    } else {
        std::thread::scope(|s| {
            s.spawn(|| into_tree(left_data, left_idxs, level + 1));
            into_tree(right_data, right_idxs, level + 1);
        });
    }
}

/// Queries a compressed tree made up of the points in `flat_data_ptr` for the nearest neighbor.
///
/// # Safety
/// This pointer must be valid
pub unsafe fn nearest_one(
    data: &[[CP32; 3]],
    data_start: *const [CP32; 3],
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
                let dptr = d as *const [CP32; 3];
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

/// Queries a compressed tree made up of the points in `flat_data_ptr` for the nearest neighbor.
/// Applies periodic boundary conditions on [-0.5, 0.5].
///
/// # Safety
/// This pointer must be valid
pub unsafe fn nearest_one_periodic(
    data: &[[CP32; 3]],
    data_start: *const [CP32; 3],
    query: &[f32; 3],
    level: usize,
) -> (f32, usize) {
    // First do real image
    let (mut best_dist_sq, mut best) = nearest_one(data, data_start, query, level, 0, f32::MAX);

    // Going to actually specify dimensions here in case we generalize to D != 3 and other boxsizes
    const D: usize = 3;
    const BOXSIZE: [f32; D] = [1.0; D];

    // Find which images we need to check (skip real and start at 1)
    for image in 1..2_usize.pow(D as u32) {
        // Closest image in the form of bool array
        let closest_image = (0..D as u32).map(|idx| ((image / 2_usize.pow(idx)) % 2) == 1);

        // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
        let dist_sq_to_side_edge_or_other = closest_image
            .clone()
            .enumerate()
            .flat_map(|(side, flag)| {
                if flag {
                    // Get minimum of dist2 to lower and upper side
                    // safety: made safe with const generic
                    Some(unsafe {
                        (BOXSIZE.get_unchecked(side) / 2.0 - query.get_unchecked(side).abs())
                            .powi(2)
                    })
                } else {
                    None
                }
            })
            .fold(0.0, |acc, x| acc + x);

        // Query with image if necessary
        if dist_sq_to_side_edge_or_other <= best_dist_sq {
            let mut image_to_check = query.clone();

            for (idx, flag) in closest_image.enumerate() {
                // If moving image along this dimension
                if flag {
                    // Do a single index here.
                    // safety: made safe with const generic
                    let query_component = unsafe { query.get_unchecked(idx) };

                    // Single index here as well
                    // safety: made safe with const generic
                    let boxsize_component = unsafe { BOXSIZE.get_unchecked(idx) };

                    // safety: made safe with const generic
                    unsafe {
                        if *query_component < 0.0 {
                            // Add if in lower half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                query_component + boxsize_component
                        } else {
                            // Subtract if in upper half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                query_component - boxsize_component
                        }
                    }
                }
            }

            // Check image
            (best_dist_sq, best) =
                nearest_one(data, data_start, &image_to_check, level, best, best_dist_sq);
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

/// Queries a compressed tree made up of the points in `flat_data_ptr` for the k nearest neighbors.
///
/// # Safety
/// This pointer must be valid
pub unsafe fn nearest_k(
    data: &[[CP32; 3]],
    data_start: *const [CP32; 3],
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
                    (d as *const [CP32; 3]).offset_from(data_start) as usize
                }));
            }
        }
        return bests;
    }

    // Get level stem
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
        // Check stem
        let dist_sq = F32(squared_euclidean(stem, query));
        if bests.len() < k || dist_sq < bests.peek().unwrap().0 {
            if bests.len() == k {
                bests.pop();
            }
            bests.push((dist_sq, unsafe {
                (stem as *const [CP32; 3]).offset_from(data_start) as usize
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

/// Queries a compressed tree made up of the points in `flat_data_ptr` for the nearest neighbor.
/// Applies periodic boundary conditions on [-0.5, 0.5].
///
/// # Safety
/// This pointer must be valid
pub unsafe fn nearest_k_periodic(
    data: &[[CP32; 3]],
    data_start: *const [CP32; 3],
    query: &[f32; 3],
    level: usize,
    k: usize,
) -> BinaryHeap<(F32, usize)> {
    // First do real image
    let mut bests = nearest_k(
        data,
        data_start,
        query,
        level,
        k,
        BinaryHeap::with_capacity(k),
    );

    // Going to actually specify dimensions here in case we generalize to D != 3 and other boxsizes
    const D: usize = 3;
    const BOXSIZE: [f32; D] = [1.0; D];

    // Find which images we need to check (skip real and start at 1)
    for image in 1..2_usize.pow(D as u32) {
        // Closest image in the form of bool array
        let closest_image = (0..D as u32).map(|idx| ((image / 2_usize.pow(idx)) % 2) == 1);

        // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
        let dist_sq_to_side_edge_or_other = closest_image
            .clone()
            .enumerate()
            .flat_map(|(side, flag)| {
                if flag {
                    // Get minimum of dist2 to lower and upper side
                    // safety: made safe with const generic
                    Some(unsafe {
                        (BOXSIZE.get_unchecked(side) / 2.0 - query.get_unchecked(side).abs())
                            .powi(2)
                    })
                } else {
                    None
                }
            })
            .fold(0.0, |acc, x| acc + x);

        // Query with image if necessary
        if dist_sq_to_side_edge_or_other <= bests.peek().unwrap().0 .0 {
            let mut image_to_check = query.clone();

            for (idx, flag) in closest_image.enumerate() {
                // If moving image along this dimension
                if flag {
                    // Do a single index here.
                    // safety: made safe with const generic
                    let query_component = unsafe { query.get_unchecked(idx) };

                    // Single index here as well
                    // safety: made safe with const generic
                    let boxsize_component = unsafe { BOXSIZE.get_unchecked(idx) };

                    // safety: made safe with const generic
                    unsafe {
                        if *query_component < 0.0 {
                            // Add if in lower half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                query_component + boxsize_component
                        } else {
                            // Subtract if in upper half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                query_component - boxsize_component
                        }
                    }
                }
            }

            // Check image
            bests = nearest_k(data, data_start, &image_to_check, level, k, bests);
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
    use crate::cast::{cast_slice, cast_slice_mut};

    use super::{into_tree, Index, CP32};

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct QueryNearest {
        pub dist_sq: f32,
        pub idx_within: u64,
    }

    /// Builds a compressed tree made up of the `num_points` points in `flat_data_ptr` inplace.
    ///
    /// # Safety
    /// Slices to the data are made from these raw parts. This pointer and length must be
    /// correct and valid.
    #[no_mangle]
    pub unsafe extern "C" fn construct_compressed_tree(
        flat_data_ptr: *mut CP32,
        num_points: u64,
        idxs_ptr: *mut Index,
    ) {
        let flat_data: &mut [CP32] =
            unsafe { std::slice::from_raw_parts_mut(flat_data_ptr, 3 * num_points as usize) };
        let data: &mut [[CP32; 3]] = cast_slice_mut(flat_data);
        let idxs: &mut [Index] =
            unsafe { std::slice::from_raw_parts_mut(idxs_ptr, num_points as usize) };
        into_tree(data, idxs, 0);
    }

    /// Queries a compressed tree made up of the `num_points` points in `flat_data_ptr` for the nearest neighbor.
    ///
    /// # Safety
    /// Slices to the data and queries are made from these raw parts. These pointers and lengths must be
    /// correct and valid.
    #[no_mangle]
    pub unsafe extern "C" fn query_compressed_nearest(
        flat_data_ptr: *const CP32,
        num_points: u64,
        flat_query_ptr: *const f32,
        num_queries: u64,
    ) -> *const QueryNearest {
        let flat_data: &[CP32] = std::slice::from_raw_parts(flat_data_ptr, 3 * num_points as usize);
        let data: &[[CP32; 3]] = cast_slice(flat_data);

        let flat_queries: &[f32] =
            std::slice::from_raw_parts(flat_query_ptr, 3 * num_queries as usize);
        let queries: &[[f32; 3]] = cast_slice(flat_queries);

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

    /// Queries a compressed tree made up of the `num_points` points in `flat_data_ptr` for the nearest neighbor.
    /// This query is parallelized via rayon
    ///
    /// # Safety
    /// Slices to the data and queries are made from these raw parts. These pointers and lengths must be
    /// correct and valid.
    #[cfg(feature = "parallel")]
    #[no_mangle]
    pub unsafe extern "C" fn query_compressed_nearest_parallel(
        flat_data_ptr: *const CP32,
        num_points: u64,
        flat_query_ptr: *const f32,
        num_queries: u64,
    ) -> *const QueryNearest {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::IntoParallelRefIterator;

        let flat_data: &[CP32] = std::slice::from_raw_parts(flat_data_ptr, 3 * num_points as usize);
        let data: &[[CP32; 3]] = cast_slice(flat_data);

        let flat_queries: &[f32] =
            std::slice::from_raw_parts(flat_query_ptr, 3 * num_queries as usize);
        let queries: &[[f32; 3]] = cast_slice(flat_queries);

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
    let data: &mut Vec<[CP32; 3]> = &mut (0..DATA)
        .map(|_| [CP32::compress(rand::random::<f32>() - 0.5, 0.0); 3])
        .collect();
    let idxs: &mut Vec<Index> = &mut (0..DATA as Index).collect();

    into_tree(data, idxs, 0);
}

#[test]
fn test_into_tree_query() {
    const DATA: usize = BUCKET_SIZE * 3;
    let data: &mut Vec<[CP32; 3]> = &mut (0..DATA)
        .map(|_| [CP32::compress(rand::random::<f32>() - 0.5, 0.0); 3])
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

#[test]
fn test_query_periodic() {
    let data: &mut Vec<[CP32; 3]> = &mut vec![
        [CP32::compress(0.0, 0.0); 3],
        [CP32::compress(0.1, 0.0); 3],
        [CP32::compress(0.49, 0.0); 3],
    ];
    let idxs: &mut Vec<Index> = &mut (0..data.len() as Index).collect();

    into_tree(data, idxs, 0);
    println!("{data:#?}");
    println!("{idxs:#?}");

    let query = [-0.49; 3];
    let (_best_dist_sq, best) = unsafe { nearest_one_periodic(data, data.as_ptr(), &query, 0) };
    assert_eq!(best, 2);
}
