//! This example is similar to mock_tree, except it uses rkyv and CPF32.
//!
//! A minimal proof of concept for a zero mem-overhead kdtree. This tree supports
//! zero-copy deserialization.
use bosque::{
    abacussummit::{self, uncompressed::CP32},
    MockTree,
};

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

    // A kDTree algorithm will produce some permutation of the particles
    // that uniquely define a queryable kDtree
    let particle_positions = [root, left, right];
    let particle_velocities = [root_vel, left_vel, right_vel];

    // Compression of particle position + saving to disk and loading bytes
    let compressed_particle_data: Vec<u8> = particle_positions
        .into_iter()
        .flatten()
        .zip(particle_velocities.into_iter().flatten())
        .map(|(position, velocity)| abacussummit::compress(position, velocity).to_le_bytes())
        .flatten()
        .collect();

    // Create an tree using uncompressed values to compare to decompressed tree
    let uncompressed_tree = MockTree::new(
        particle_positions[0],
        particle_positions[1],
        particle_positions[2],
    );

    // Now we can zero-copy deserialize
    let cf_tree = unsafe {
        rkyv::archived_root::<bosque::CompressedTree<CP32, D>>(&compressed_particle_data)
    };
    println!("uncompressed tree = {uncompressed_tree:#?}");
    println!("zero copy tree = {cf_tree:#?}");
}
