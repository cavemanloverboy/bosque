//! This module implements the compression scheme used in AbacussSummit.
//!
//! Position:
//! Given a position in [-0.5, 0.5], map to integers in [-500,000, 500,000].
//! This can be stored in 20 bits, as 2^20 = 1 048 576
//!
//! Velocity:
//! Given a velocity in [-6000 km/s, 6000 km/s], map to integers in [0, 4096).
//! This can be stored in 12 bits, as 2^12 = 4096

/// Compress a position/velocity `f32/f32` pair in [-0.5, 0.5] x [-6000, 6000] into a `u32`.
/// For AbacusSummit, this corresponds to simulation units for position and km/s for velocity.
#[no_mangle]
pub extern "C" fn compress(position: f32, velocity: f32) -> u32 {
    let position_mapped = ((position + 0.5) * 1_000_000.0) as u32;
    let velocity_mapped = ((velocity + 6000.0) * 4096.0 / 12000.0) as u32;

    // Ensure the mapped values fit into their respective bit sizes
    let position_masked = position_mapped & ((1 << 20) - 1);
    let velocity_masked = velocity_mapped & ((1 << 12) - 1);

    // Shift position by 12 bits to the left to make room for velocity
    let position_shifted = position_masked << 12;

    // Combine position_shifted and velocity_masked using bitwise OR
    position_shifted | velocity_masked
}

/// Decompresses a `u32` dword into  a position/velocity `f32/f32` pair in [-0.5, 0.5] x [-6000, 6000].
/// For AbacusSummit, this corresponds to simulation units for position and km/s for velocity.
pub fn decompress(compressed: u32) -> [f32; 2] {
    // Extract the position by shifting 12 bits to the right
    let position = (compressed >> 12) as f32;

    // Extract the velocity by masking off the higher bits
    let velocity = (compressed & ((1 << 12) - 1)) as f32;

    // Un-map the position and velocity to their original ranges
    let position_unmapped = (position / 1_000_000.0) - 0.5;
    let velocity_unmapped = (velocity * 12000.0 / 4096.0) - 6000.0;

    [position_unmapped, velocity_unmapped]
}

/// Decompresses only the position from the `u32` dword containing the position/velocity pair.
/// For AbacusSummit, this corresponds to simulation units for position and km/s for velocity.
#[inline(always)]
pub fn decompress_position(compressed: &u32) -> f32 {
    // Extract the position by shifting 12 bits to the right
    let position = (compressed >> 12) as f32;

    // Un-map the position to its original range
    (position / 1_000_000.0) - 0.5
}

pub mod ffi {
    /// Auxiliary struct used for ffi
    #[repr(C)]
    pub struct DecompressedPair {
        pub pos: f32,
        pub vel: f32,
    }

    /// Decompresses a `u32` dword into  a position/velocity `f32/f32` pair in [-0.5, 0.5] x [-6000, 6000].
    /// For AbacusSummit, this corresponds to simulation units for position and km/s for velocity.
    #[no_mangle]
    pub extern "C" fn decompress(compressed: u32) -> DecompressedPair {
        let [pos, vel] = super::decompress(compressed);
        DecompressedPair { pos, vel }
    }
}

#[allow(unused_variables)]
pub mod uncompressed {
    use std::{
        fmt::Debug,
        ops::{Add, Div, Mul, Neg, Rem, Sub},
    };

    // use num_traits::*;

    #[derive(Clone, Copy)]
    #[repr(C)]
    /// A wrapper for the `u32` d which holds the compressed position and velocity.
    /// It provides utilities for using the positional information.
    ///
    /// Compressed Position 32-bit -> `CP32`.
    pub struct CP32(u32);

    impl Default for CP32 {
        fn default() -> Self {
            Self(super::compress(0.0, 0.0))
        }
    }

    impl CP32 {
        #[inline(always)]
        pub fn compress(pos: f32, vel: f32) -> CP32 {
            Self(super::compress(pos, vel))
        }
        #[inline(always)]
        pub fn decompress(&self) -> f32 {
            super::decompress_position(&self.0)
        }
    }

    impl Debug for CP32 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.decompress())
        }
    }

    impl PartialEq for CP32 {
        #[inline(always)]
        fn eq(&self, other: &Self) -> bool {
            self.decompress().eq(&other.decompress())
        }
    }

    impl PartialOrd for CP32 {
        #[inline(always)]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.decompress().partial_cmp(&other.decompress())
        }
    }

    impl Neg for CP32 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self::Output {
            let [x, v] = super::decompress(self.0);
            CP32(super::compress(-x, v))
        }
    }

    impl Rem for CP32 {
        type Output = Self;
        #[inline(always)]
        fn rem(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.rem(xr);
            Self(super::compress(output, 0.0))
        }
    }
    impl Div for CP32 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.div(xr);
            Self(super::compress(output, 0.0))
        }
    }
    impl Sub for CP32 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.sub(xr);
            Self(super::compress(output, 0.0))
        }
    }
    impl Add for CP32 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.add(xr);
            Self(super::compress(output, 0.0))
        }
    }
    impl Mul for CP32 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.mul(xr);
            Self(super::compress(output, 0.0))
        }
    }
}
