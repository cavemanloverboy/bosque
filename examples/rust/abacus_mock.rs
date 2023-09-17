//! A full AbacusSummit kdtree mock run.
//!
//! 1) Generate mock position and velocity data (i.e. snapshot to be output)
//! 2) Compress mock position and velocity data `f32/f32` -> `u32`
//! 3) Use bosque for inplace kdtree, which rearranges the compressed `u32`s and particle ids.
//!   3.5) Not done here, but at this stage we can now save/load this data.
//! 4) Query tree
use bosque::{
    abacussummit::uncompressed::CP32,
    cast::{cast_slice, cast_slice_mut},
    tree::ffi::{
        construct_compressed_tree, query_compressed_nearest, query_compressed_nearest_parallel,
    },
};
use std::time::Instant;

const NUM_POINTS: usize = 512 * 512 * 512;
const NUM_QUERIES: usize = 1_000_000;
fn main() {
    // Allocate arrays for mock AbacusSummit sim data
    let init_data_timer = Instant::now();
    let mut pos = Vec::with_capacity(NUM_POINTS);
    let mut vel = Vec::with_capacity(NUM_POINTS);
    let mut idx = Vec::with_capacity(NUM_POINTS);
    for i in 0..NUM_POINTS {
        // Position in [-0.5, 0.5]
        pos.push([(); 3].map(|_| rand::random::<f32>() - 0.5));

        // Velocity in [-6000.0, 6000.0]
        vel.push([(); 3].map(|_| 6000.0 * (rand::random::<f32>() - 0.5)));

        idx.push(i);
    }
    println!(
        "Initialized mock data in {} seconds",
        init_data_timer.elapsed().as_secs_f32()
    );

    // Compressed values
    let compress_timer = Instant::now();
    let mut cpos = Vec::with_capacity(NUM_POINTS);
    for (p, v) in pos.into_iter().zip(vel) {
        let compressed: [CP32; 3] = std::array::from_fn(|i| CP32::compress(p[i], v[i]));
        cpos.push(compressed);
    }
    println!(
        "Compressed mock data in {} seconds",
        compress_timer.elapsed().as_secs_f32()
    );

    // Create tree on compressed data (using extern "C")
    let construction_timer = Instant::now();
    let flat_data_ptr = cast_slice_mut(&mut cpos);
    let idxs_ptr = cast_slice_mut(&mut idx);
    unsafe {
        construct_compressed_tree(
            flat_data_ptr.as_mut_ptr(),
            NUM_POINTS as u64,
            idxs_ptr.as_mut_ptr(),
        );
    }
    println!(
        "Constructed compressed tree in {} seconds",
        construction_timer.elapsed().as_secs_f32()
    );

    // Query near origin (using extern "C")
    let flat_data_ptr = cast_slice(&cpos);
    let result = unsafe {
        *query_compressed_nearest(
            flat_data_ptr.as_ptr(),
            NUM_POINTS as u64,
            [0.0; 3].as_ptr(),
            1,
        )
    };
    println!(
        "Queried at origin: ({:.5}, {:.5}, {:.5}) -> euclidean distance {:.5}",
        cpos[result.idx_within as usize][0].decompress(),
        cpos[result.idx_within as usize][1].decompress(),
        cpos[result.idx_within as usize][2].decompress(),
        result.dist_sq.sqrt()
    );

    // Initialized queries
    let mut queries = Vec::with_capacity(NUM_POINTS);
    for _ in 0..NUM_QUERIES * 3 {
        // Position in [-0.5, 0.5]
        queries.push(rand::random::<f32>() - 0.5);
    }
    println!("Initialized queries");

    // Query tree many times
    let query_timer = Instant::now();
    unsafe {
        std::hint::black_box(query_compressed_nearest(
            flat_data_ptr.as_ptr(),
            NUM_POINTS as u64,
            queries.as_ptr(),
            NUM_QUERIES as u64,
        ))
    };
    println!(
        "Carried out {NUM_QUERIES} queries in {} millis",
        query_timer.elapsed().as_millis(),
    );
    #[cfg(feature = "parallel")]
    {
        let query_timer = Instant::now();
        unsafe {
            std::hint::black_box(query_compressed_nearest_parallel(
                flat_data_ptr.as_ptr(),
                NUM_POINTS as u64,
                queries.as_ptr(),
                NUM_QUERIES as u64,
            ))
        };
        println!(
            "Carried out {NUM_QUERIES} queries in parallel in {} millis",
            query_timer.elapsed().as_millis(),
        );
    }
}
