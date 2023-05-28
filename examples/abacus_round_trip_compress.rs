//! A compression/decompression round trip of a `f32/f32` pair as in AbacusSummit.
//!
//! `f32/f32` -> `u32` -> `f32/f32`
use bosque::abacussummit;

fn main() {
    // Initialize some test values in [-0.5, 0.5] x [-6000, 6000]
    let x: f32 = -0.33;
    let v: f32 = 129.98765432101;
    println!("full {x} and {v}");

    // Compress into u32
    let compressed: u32 = abacussummit::compress(x, v);
    println!("masked {:?}", compressed.to_le_bytes());

    // Decompress
    let [uncompressed_x, uncompressed_v] = abacussummit::decompress(compressed);
    println!("uncompressed {uncompressed_x} and {uncompressed_v}");

    // Check fractional error
    let xerr = 100.0 * (uncompressed_x - x) / x;
    let verr = 100.0 * (uncompressed_v - v) / v;
    println!("error is {xerr}% and {verr}%");
}
