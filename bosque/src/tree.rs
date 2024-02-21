use crate::{
    abacussummit::uncompressed::CP32, float::TreeFloat,
    mirror_select::mirror_select_nth_unstable_by,
};

pub const BUCKET_SIZE: usize = 32;
pub type Index = u32;

pub fn build_tree_with_indices<T: TreeFloat>(data: &mut [[T; 3]], idxs: &mut [Index]) {
    into_tree(data, idxs, 0)
}

pub fn build_tree<T: TreeFloat>(data: &mut [[T; 3]]) {
    into_tree_no_idxs(data, 0)
}

#[inline(always)]
fn into_tree<T: TreeFloat>(data: &mut [[T; 3]], idxs: &mut [Index], level: usize) {
    let mut trampoline_state = (data, idxs, level);
    'trampoline_loop: loop {
        let (data, idxs, level) = trampoline_state;
        return {
            if data.len() <= BUCKET_SIZE {
                return;
            }

            // Do current level
            let median = data.len() / 2;
            let level_dim = level % 3;
            mirror_select_nth_unstable_by(data, idxs, median, |a, b| unsafe {
                a.get_unchecked(level_dim)
                    .partial_cmp(b.get_unchecked(level_dim))
                    .unwrap()
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
                // Need to call into_tree twice -- only one is unrolled
                into_tree(left_data, left_idxs, level + 1);
                trampoline_state = (right_data, right_idxs, level + 1);
                continue 'trampoline_loop;
            } else {
                std::thread::scope(|s| {
                    s.spawn(|| into_tree(left_data, left_idxs, level + 1));
                    into_tree(right_data, right_idxs, level + 1);
                });
            }
        };
    }
}

#[inline(always)]
fn into_tree_no_idxs<T: TreeFloat>(data: &mut [[T; 3]], level: usize) {
    let mut trampoline_state = (data, level);
    'trampoline_loop: loop {
        let (data, level) = trampoline_state;
        return {
            if data.len() <= BUCKET_SIZE {
                return;
            }

            // Do current level
            let median = data.len() / 2;
            let level_dim = level % 3;
            data.select_nth_unstable_by(median, |a, b| unsafe {
                a.get_unchecked(level_dim)
                    .partial_cmp(b.get_unchecked(level_dim))
                    .unwrap()
            });

            // Get left and right data and indices, sans median
            let (left_data, median_and_right_data) = data.split_at_mut(median);
            let right_data = &mut median_and_right_data[1..];

            // Do left and right, recursively
            // on level 0 we get 2^1 = 2 and spawn a thread so 2 total
            // on level 1 we get 2^2 = 4 and spawn a thread on each of 2 threads so 4 total
            // on level 2 we get 2^3 = 8 and spawn a thread on each of 4 threads so 8 total
            // on level 3 we get 2^4 = 16 > 8 -> so sequential.
            let lte_8_threads = 2_usize.pow(1 + level as u32) > 8;
            let small_data = left_data.len() < 25_000;
            let sequential = small_data | lte_8_threads;
            if sequential {
                // Need to call into_tree twice -- only one is unrolled
                into_tree_no_idxs(left_data, level + 1);
                trampoline_state = (right_data, level + 1);
                continue 'trampoline_loop;
            } else {
                std::thread::scope(|s| {
                    s.spawn(|| into_tree_no_idxs(left_data, level + 1));
                    into_tree_no_idxs(right_data, level + 1);
                });
            }
        };
    }
}

/// Queries a compressed tree made up of the points in `tree` for the nearest neighbor.
pub fn nearest_one<T: TreeFloat>(tree: &[[T; 3]], query: &[T::Query; 3]) -> (T::Query, usize) {
    // SAFETY: we are given the whole tree as a valid reference so pointer must be valid
    T::output(unsafe { _nearest_one(tree, tree.as_ptr(), T::input(query), 0, 0, T::max()) })
}

/// Queries a compressed tree made up of the points in `tree` for the nearest neighbor.
/// Applied periodic boundary conditions on [-0.5, 0.5]
pub fn nearest_one_periodic<T: TreeFloat>(
    tree: &[[T; 3]],
    query: &[T::Query; 3],
    lo: T::Query,
    hi: T::Query,
) -> (T::Query, usize) {
    // SAFETY: we are given the whole tree as a valid reference so pointer must be valid
    T::output(unsafe {
        _nearest_one_periodic(
            tree,
            tree.as_ptr(),
            T::input(query),
            0,
            T::new_accumulator_from_query(lo),
            T::new_accumulator_from_query(hi),
        )
    })
}

/// Queries a compressed tree made up of the points in `tree` for the nearest k neighbors.
pub fn nearest_k<T: TreeFloat>(
    tree: &[[T; 3]],
    query: &[T::Query; 3],
    k: usize,
) -> Vec<(T::Query, usize)> {
    // SAFETY: we are given the whole tree as a valid reference so pointer must be valid
    T::output_bh(unsafe {
        _nearest_k(
            tree,
            tree.as_ptr(),
            T::input(query),
            0,
            k,
            BinaryHeap::with_capacity(k),
        )
    })
}

/// Queries a compressed tree made up of the points in `tree` for the nearest k neighbors.
pub fn nearest_k_sep<T: TreeFloat>(
    tree: &[[T; 3]],
    query: &[T::Query; 3],
    k: usize,
) -> (Vec<T::Query>, Vec<usize>) {
    // SAFETY: we are given the whole tree as a valid reference so pointer must be valid
    T::output_bh_sep(unsafe {
        _nearest_k(
            tree,
            tree.as_ptr(),
            T::input(query),
            0,
            k,
            BinaryHeap::with_capacity(k),
        )
    })
}

/// Queries a compressed tree made up of the points in `tree` for the nearest k neighbors.
///
/// To be used by `pybosque` crate to remove an allocation
pub fn nearest_k_bh<T: TreeFloat>(
    tree: &[[T; 3]],
    query: &[T::Query; 3],
    k: usize,
) -> BinaryHeap<(T::Accumulator, usize)> {
    // SAFETY: we are given the whole tree as a valid reference so pointer must be valid
    unsafe {
        _nearest_k(
            tree,
            tree.as_ptr(),
            T::input(query),
            0,
            k,
            BinaryHeap::with_capacity(k),
        )
    }
}

/// Queries a compressed tree made up of the points in `tree` for the nearest k neighbors.
pub fn nearest_k_periodic<T: TreeFloat>(
    tree: &[[T; 3]],
    query: &[T::Query; 3],
    k: usize,
    lo: T::Query,
    hi: T::Query,
) -> Vec<(T::Query, usize)> {
    // SAFETY: we are given the whole tree as a valid reference so pointer must be valid
    T::output_bh(unsafe {
        _nearest_k_periodic(
            tree,
            tree.as_ptr(),
            T::input(query),
            k,
            T::new_accumulator_from_query(lo),
            T::new_accumulator_from_query(hi),
        )
    })
}

/// Queries a compressed tree made up of the points in `tree` for the nearest k neighbors.
/// //
/// To be used by `pybosque` crate to remove an allocation
pub fn nearest_k_periodic_bh<T: TreeFloat>(
    tree: &[[T; 3]],
    query: &[T::Query; 3],
    k: usize,
    lo: T::Query,
    hi: T::Query,
) -> BinaryHeap<(T::Accumulator, usize)> {
    // SAFETY: we are given the whole tree as a valid reference so pointer must be valid
    unsafe {
        _nearest_k_periodic(
            tree,
            tree.as_ptr(),
            T::input(query),
            k,
            T::new_accumulator_from_query(lo),
            T::new_accumulator_from_query(hi),
        )
    }
}

/// Queries a compressed tree made up of the points in `tree` for the nearest k neighbors.
pub fn nearest_k_periodic_sep<T: TreeFloat>(
    tree: &[[T; 3]],
    query: &[T::Query; 3],
    k: usize,
    lo: T::Query,
    hi: T::Query,
) -> (Vec<T::Query>, Vec<usize>) {
    // SAFETY: we are given the whole tree as a valid reference so pointer must be valid
    T::output_bh_sep(unsafe {
        _nearest_k_periodic(
            tree,
            tree.as_ptr(),
            T::input(query),
            k,
            T::new_accumulator_from_query(lo),
            T::new_accumulator_from_query(hi),
        )
    })
}

/// Inner recursive function.
///
/// Queries a compressed tree made up of the points in `flat_data_ptr` for the nearest neighbor.
///
/// # Safety
/// This pointer must be valid
#[inline(always)]
unsafe fn _nearest_one<T: TreeFloat>(
    data: &[[T; 3]],
    data_start: *const [T; 3],
    query: &[T::Accumulator; 3],
    level: usize,
    mut best: usize,
    mut best_dist_sq: T::Accumulator,
) -> (T::Accumulator, usize) {
    // Deal with bucket
    if data.len() <= BUCKET_SIZE {
        for d in data {
            let dist_sq = squared_euclidean(d, query);
            if dist_sq <= best_dist_sq {
                best_dist_sq = dist_sq;
                let dptr = d as *const [T; 3];
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
    let dx = unsafe {
        stem.get_unchecked(level_dim)
            .sub_accumulator(query.get_unchecked(level_dim))
    };

    // Determine which direction
    let go_left = dx > T::zero();

    // Get left and query left if necessary
    let (left_data, median_and_right_data) = data.split_at(median);
    if go_left {
        let (left_best_dist_sq, left_best) =
            _nearest_one(left_data, data_start, query, level + 1, best, best_dist_sq);
        if left_best_dist_sq < best_dist_sq {
            best = left_best;
            best_dist_sq = left_best_dist_sq;
        }
    } else {
        // Get right and query right if necessary
        let right_data = &median_and_right_data[1..];
        let (right_best_dist_sq, right_best) =
            _nearest_one(right_data, data_start, query, level + 1, best, best_dist_sq);
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
            let dptr = stem as *const [T; 3];
            best = unsafe { dptr.offset_from(data_start) as usize };
        }

        // Check other dim
        // Invert logic
        if !go_left {
            let (left_best_dist_sq, left_best) =
                _nearest_one(left_data, data_start, query, level + 1, best, best_dist_sq);
            if left_best_dist_sq < best_dist_sq {
                best = left_best;
                best_dist_sq = left_best_dist_sq;
            }
        } else {
            // Get right and query right if necessary
            let right_data = &median_and_right_data[1..];
            let (right_best_dist_sq, right_best) =
                _nearest_one(right_data, data_start, query, level + 1, best, best_dist_sq);
            if right_best_dist_sq < best_dist_sq {
                best = right_best;
                best_dist_sq = right_best_dist_sq;
            }
        }
    }

    (best_dist_sq, best)
}

/// Inner recursive function.
///
/// Queries a compressed tree made up of the points in `flat_data_ptr` for the nearest neighbor.
/// Applies periodic boundary conditions on [-0.5, 0.5].
///
/// # Safety
/// This pointer must be valid
#[inline(always)]
unsafe fn _nearest_one_periodic<T: TreeFloat>(
    data: &[[T; 3]],
    data_start: *const [T; 3],
    query: &[T::Accumulator; 3],
    level: usize,
    lo: T::Accumulator,
    hi: T::Accumulator,
) -> (T::Accumulator, usize) {
    // First do real image
    let (mut best_dist_sq, mut best) = _nearest_one(data, data_start, query, level, 0, T::max());

    // Going to actually specify dimensions here in case we generalize to D != 3 and other boxsizes
    const D: usize = 3;

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
                        let dx = T::abs(&(lo - *query.get_unchecked(side)))
                            .min(T::abs(&(hi - *query.get_unchecked(side))));
                        // T::half(boxsize.get_unchecked(side))
                        // - T::abs(query.get_unchecked(side));
                        dx * dx
                    })
                } else {
                    None
                }
            })
            .fold(T::zero(), |acc, x| acc + x);

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
                    let boxsize_component = hi - lo;

                    // safety: made safe with const generic
                    unsafe {
                        let midpoint = T::half(&(lo + hi));
                        if *query_component < midpoint {
                            // Add if in lower half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                *query_component + boxsize_component
                        } else {
                            // Subtract if in upper half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                *query_component - boxsize_component
                        }
                    }
                }
            }

            // Check image
            (best_dist_sq, best) =
                _nearest_one(data, data_start, &image_to_check, level, best, best_dist_sq);
        }
    }

    (best_dist_sq, best)
}

use std::collections::BinaryHeap;

/// Inner recursive function.
///
/// Queries a compressed tree made up of the points in `flat_data_ptr` for the k nearest neighbors.
///
/// # Safety
/// This pointer must be valid
#[inline(always)]
unsafe fn _nearest_k<T: TreeFloat>(
    data: &[[T; 3]],
    data_start: *const [T; 3],
    query: &[T::Accumulator; 3],
    level: usize,
    k: usize,
    mut bests: BinaryHeap<(T::Accumulator, usize)>,
) -> BinaryHeap<(T::Accumulator, usize)> {
    // Deal with bucket
    if data.len() <= BUCKET_SIZE {
        for d in data {
            let dist_sq = squared_euclidean(d, query);
            if bests.len() < k || dist_sq < bests.peek().unwrap().0 {
                if bests.len() == k {
                    bests.pop();
                }
                bests.push((dist_sq, unsafe {
                    (d as *const [T; 3]).offset_from(data_start) as usize
                }));
            }
        }
        return bests;
    }

    // Get level stem
    let median = data.len() / 2;
    let level_dim = level % 3;
    let stem = unsafe { data.get_unchecked(median) };

    let dx = unsafe {
        stem.get_unchecked(level_dim)
            .sub_accumulator(query.get_unchecked(level_dim))
    };
    let go_left = dx > T::zero();

    let (left_data, median_and_right_data) = data.split_at(median);
    if go_left {
        bests = _nearest_k(left_data, data_start, query, level + 1, k, bests);
    } else {
        let right_data = &median_and_right_data[1..];
        bests = _nearest_k(right_data, data_start, query, level + 1, k, bests);
    }

    // Check whether we have to check stem or other dim
    // 1) if bests is not full (regardless of current bests)
    // 2) if plane is closer than kth best
    let check_stem_and_other_dim = if bests.len() < k {
        true
    } else {
        bests
            .peek()
            .map_or(true, |&(dist_sq, _)| dist_sq >= dx * dx)
    };

    if check_stem_and_other_dim {
        // Check stem
        let dist_sq = squared_euclidean(stem, query);
        if bests.len() < k || dist_sq < bests.peek().unwrap().0 {
            if bests.len() == k {
                bests.pop();
            }
            bests.push((dist_sq, unsafe {
                (stem as *const [T; 3]).offset_from(data_start) as usize
            }));
        }

        // Check other dim
        // Invert logic
        if !go_left {
            bests = _nearest_k(left_data, data_start, query, level + 1, k, bests);
        } else {
            let right_data = &median_and_right_data[1..];
            bests = _nearest_k(right_data, data_start, query, level + 1, k, bests);
        }
    }

    bests
}

/// Inner recursive function.
///
/// Queries a compressed tree made up of the points in `flat_data_ptr` for the nearest neighbor.
/// Applies periodic boundary conditions on [-0.5, 0.5].
///
/// # Safety
/// This pointer must be valid
#[inline(always)]
unsafe fn _nearest_k_periodic<T: TreeFloat>(
    data: &[[T; 3]],
    data_start: *const [T; 3],
    query: &[T::Accumulator; 3],
    k: usize,
    lo: T::Accumulator,
    hi: T::Accumulator,
) -> BinaryHeap<(T::Accumulator, usize)> {
    // First do real image
    let mut bests = _nearest_k(data, data_start, query, 0, k, BinaryHeap::with_capacity(k));

    // Going to actually specify dimensions here in case we generalize to D != 3 and other boxsizes
    const D: usize = 3;

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
                        let dx = T::abs(&(lo - *query.get_unchecked(side)))
                            .min(T::abs(&(hi - *query.get_unchecked(side))));
                        dx * dx
                    })
                } else {
                    None
                }
            })
            .fold(T::zero(), |acc, x| acc + x);

        // Query with image if necessary
        if dist_sq_to_side_edge_or_other <= bests.peek().unwrap().0 {
            let mut image_to_check = query.clone();

            for (idx, flag) in closest_image.enumerate() {
                // If moving image along this dimension
                if flag {
                    // Do a single index here.
                    // safety: made safe with const generic
                    let query_component = unsafe { query.get_unchecked(idx) };

                    // Single index here as well
                    // safety: made safe with const generic
                    let boxsize_component = hi - lo;

                    // safety: made safe with const generic
                    unsafe {
                        let midpoint = T::half(&(lo + hi));
                        if *query_component < midpoint {
                            // Add if in lower half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                *query_component + boxsize_component
                        } else {
                            // Subtract if in upper half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                *query_component - boxsize_component
                        }
                    }
                }
            }

            // Check image
            bests = _nearest_k(data, data_start, &image_to_check, 0, k, bests);
        }
    }

    bests
}

#[inline(always)]
pub fn squared_euclidean<T: TreeFloat>(a: &[T; 3], q: &[T::Accumulator; 3]) -> T::Accumulator {
    unsafe {
        let dx = a.get_unchecked(0).sub_accumulator(q.get_unchecked(0));
        let dy = a.get_unchecked(1).sub_accumulator(q.get_unchecked(1));
        let dz = a.get_unchecked(2).sub_accumulator(q.get_unchecked(2));

        dx * dx + dy * dy + dz * dz
    }
}

pub fn construct_cp32(data: &mut [[CP32; 3]], idxs: &mut [Index]) {
    into_tree(data, idxs, 0)
}

pub fn construct_f32(data: &mut [[f32; 3]], idxs: &mut [Index]) {
    into_tree(data, idxs, 0)
}

pub fn construct_f64(data: &mut [[f64; 3]], idxs: &mut [Index]) {
    into_tree(data, idxs, 0)
}

pub mod ffi {
    use crate::{
        cast::{cast_slice, cast_slice_mut},
        float::F32,
    };

    use super::{into_tree, Index, CP32};

    /// An auxiliary ffi type for multi-value returns.
    ///
    /// # SAFETY
    /// If the fields of this type ever change review the extern "C" functions in this module for safety.
    #[derive(Default, Clone, Copy)]
    #[repr(C)]
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

    /// Builds a compressed tree made up of the `num_points` points in `flat_data_ptr` inplace.
    ///
    /// # Safety
    /// Slices to the data are made from these raw parts. This pointer and length must be
    /// correct and valid.
    #[no_mangle]
    pub unsafe extern "C" fn construct_tree_f32(
        flat_data_ptr: *mut f32,
        num_points: u64,
        idxs_ptr: *mut Index,
    ) {
        let flat_data: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(flat_data_ptr, 3 * num_points as usize) };
        let data: &mut [[f32; 3]] = cast_slice_mut(flat_data);
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
        let queries: &[[F32; 3]] = cast_slice(flat_queries);

        let results: Vec<(F32, u64)> = queries
            .iter()
            .map(|q| {
                let (dist_sq, idx_within) =
                    super::_nearest_one(data, data.as_ptr(), q, 0, 0, F32(f32::MAX));
                let idx_within = idx_within as u64;
                (dist_sq, idx_within)
            })
            .collect();
        std::mem::transmute(Box::leak(results.into_boxed_slice()).as_ptr())
    }

    /// Queries a f32 tree made up of the `num_points` points in `flat_data_ptr` for the nearest neighbor.
    ///
    /// # Safety
    /// Slices to the data and queries are made from these raw parts. These pointers and lengths must be
    /// correct and valid.
    #[no_mangle]
    pub unsafe extern "C" fn query_f32_nearest(
        flat_data_ptr: *const f32,
        num_points: u64,
        flat_query_ptr: *const f32,
        num_queries: u64,
    ) -> *const QueryNearest {
        let flat_data: &[f32] = std::slice::from_raw_parts(flat_data_ptr, 3 * num_points as usize);
        let data: &[[f32; 3]] = cast_slice(flat_data);

        let flat_queries: &[f32] =
            std::slice::from_raw_parts(flat_query_ptr, 3 * num_queries as usize);
        let queries: &[[F32; 3]] = cast_slice(flat_queries);

        let results: Vec<(F32, u64)> = queries
            .iter()
            .map(|q| {
                let (dist_sq, idx_within) =
                    super::_nearest_one(data, data.as_ptr(), q, 0, 0, F32(f32::MAX));
                let idx_within = idx_within as u64;
                (dist_sq, idx_within)
            })
            .collect();
        std::mem::transmute(Box::leak(results.into_boxed_slice()).as_ptr())
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
        let queries: &[[F32; 3]] = cast_slice(flat_queries);

        let results: Vec<(F32, u64)> = queries
            .par_iter()
            .map(|q| {
                let (dist_sq, idx_within) =
                    super::_nearest_one(data, data.as_ptr(), q, 0, 0, F32(f32::MAX));
                let idx_within = idx_within as u64;
                (dist_sq, idx_within)
            })
            .collect();

        std::mem::transmute(Box::leak(results.into_boxed_slice()).as_ptr())
    }

    /// Queries a f32 tree made up of the `num_points` points in `flat_data_ptr` for the nearest neighbor.
    /// This query is parallelized via rayon
    ///
    /// # Safety
    /// Slices to the data and queries are made from these raw parts. These pointers and lengths must be
    /// correct and valid.
    #[cfg(feature = "parallel")]
    #[no_mangle]
    pub unsafe extern "C" fn query_f32_nearest_parallel(
        flat_data_ptr: *const f32,
        num_points: u64,
        flat_query_ptr: *const f32,
        num_queries: u64,
    ) -> *const QueryNearest {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::IntoParallelRefIterator;

        let flat_data: &[f32] = std::slice::from_raw_parts(flat_data_ptr, 3 * num_points as usize);
        let data: &[[f32; 3]] = cast_slice(flat_data);

        let flat_queries: &[f32] =
            std::slice::from_raw_parts(flat_query_ptr, 3 * num_queries as usize);
        let queries: &[[F32; 3]] = cast_slice(flat_queries);

        // TODO remove
        let timer = std::time::Instant::now();
        let results: Vec<(F32, u64)> = queries
            .par_iter()
            .map(|q| {
                let (dist_sq, idx_within) =
                    super::_nearest_one(data, data.as_ptr(), q, 0, 0, F32(f32::MAX));
                let idx_within = idx_within as u64;
                (dist_sq, idx_within)
            })
            .collect();
        // TODO remove
        println!("\ndone in {} ms", timer.elapsed().as_millis());

        std::mem::transmute(Box::leak(results.into_boxed_slice()).as_ptr())
    }
}
