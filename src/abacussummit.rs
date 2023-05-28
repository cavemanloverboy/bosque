//! This module implements the compression scheme used in AbacussSummit.
//!
//! Position:
//! Given a position in [-0.5, 0.5], map to integers in [-500,000, 500,000].
//! This can be stored in 20 bits, as 2^20 = 1 048 576
//!
//! Velocity:
//! Given a velocity in [-6000 km/s, 6000 km/s], map to integers in [0, 4096).
//! This can be stored in 12 bits, as 2^12 = 4096

use crate::MockTree;

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
pub fn decompress_position(compressed: &u32) -> f32 {
    // Extract the position by shifting 12 bits to the right
    let position = (compressed >> 12) as f32;

    // Un-map the position to its original range
    let position_unmapped = (position / 1_000_000.0) - 0.5;

    position_unmapped
}

impl<const D: usize> MockTree<D> {
    pub fn new_abacus(root: [f32; D], left: [f32; D], right: [f32; D]) -> MockTree<D> {
        MockTree { root, left, right }
    }

    pub fn from_abacussummit_compressed(compressed_bytes: &[u8]) -> MockTree<D> {
        // Iterator over u32 values
        let compressed_dwords: &[u32] = bytemuck::cast_slice(compressed_bytes);
        let mut compressed_dwords_iter = compressed_dwords.into_iter();

        MockTree {
            root: [(); D].map(|_| decompress(*compressed_dwords_iter.next().unwrap())[0]),
            left: [(); D].map(|_| decompress(*compressed_dwords_iter.next().unwrap())[0]),
            right: [(); D].map(|_| decompress(*compressed_dwords_iter.next().unwrap())[0]),
        }
    }
}

pub mod ffi {
    use crate::MockTree;

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

    #[no_mangle]
    pub extern "C" fn from_abacussummit_compressed(
        compressed_bytes: *const u8,
        bytes_len: usize,
    ) -> MockTree<3> {
        assert_eq!(bytes_len, 3 * 3 * core::mem::size_of::<u32>());

        let slice: &[u8] = unsafe { core::slice::from_raw_parts(compressed_bytes, bytes_len) };
        MockTree::from_abacussummit_compressed(slice)
    }

    #[no_mangle]
    pub extern "C" fn pretty_print_tree(
        prefix: *const core::ffi::c_char,
        tree: *const MockTree<3>,
    ) {
        unsafe {
            let c_str = core::ffi::CStr::from_ptr(prefix);
            let str_slice = c_str.to_str().expect("Invalid UTF-8 sequence in C string");
            println!("{str_slice}{:#?}", *tree);
        }
    }

    #[no_mangle]
    pub extern "C" fn new_abacus(
        root: *const [f32; 3],
        left: *const [f32; 3],
        right: *const [f32; 3],
    ) -> MockTree<3> {
        unsafe {
            let root = *root;
            let left = *left;
            let right = *right;
            MockTree { root, left, right }
        }
    }
}

#[cfg(feature = "uncompressed")]
#[clippy::allow(unused_variables)]
#[allow(unused_variables)]
pub mod uncompressed {
    use std::{
        fmt::Debug,
        ops::{Add, Div, Mul, Neg, Rem, Sub},
    };

    use bytemuck::{Pod, Zeroable};
    use num_traits::*;

    #[derive(Clone, Copy)]
    #[repr(C)]
    /// A wrapper for the `u32` d which holds the compressed position and velocity.
    /// It provides utilities for using the positional information.
    ///
    /// Compressed Position 32-bit -> `CP32`.
    pub struct CP32(u32);

    impl rkyv::Archive for CP32 {
        type Archived = Self;
        type Resolver = ();

        #[inline]
        unsafe fn resolve(&self, _: usize, _: Self::Resolver, out: *mut Self::Archived) {
            out.write(*self);
        }
    }
    impl<S: rkyv::Fallible + ?Sized> rkyv::Serialize<S> for CP32 {
        #[inline]
        fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
            Ok(())
        }
    }

    unsafe impl Pod for CP32 {}
    unsafe impl Zeroable for CP32 {}

    impl Default for CP32 {
        fn default() -> Self {
            Self(super::compress(0.0, 0.0))
        }
    }

    impl CP32 {
        pub fn compress(pos: f32, vel: f32) -> CP32 {
            Self(super::compress(pos, vel))
        }
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
        fn eq(&self, other: &Self) -> bool {
            self.decompress().eq(&other.decompress())
        }
    }

    impl PartialOrd for CP32 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.decompress().partial_cmp(&other.decompress())
        }
    }
    impl ToPrimitive for CP32 {
        fn to_i64(&self) -> Option<i64> {
            panic!("unnecessary boilerplate")
        }
        fn to_u64(&self) -> Option<u64> {
            panic!("unnecessary boilerplate")
        }
    }

    impl NumCast for CP32 {
        fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
            None
        }
    }
    impl Neg for CP32 {
        type Output = Self;
        fn neg(self) -> Self::Output {
            let [x, v] = super::decompress(self.0);
            CP32(super::compress(-x, v))
        }
    }

    impl Rem for CP32 {
        type Output = Self;
        fn rem(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.rem(xr);
            Self(super::compress(output, 0.0))
        }
    }
    impl Div for CP32 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.div(xr);
            Self(super::compress(output, 0.0))
        }
    }
    impl Sub for CP32 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.sub(xr);
            Self(super::compress(output, 0.0))
        }
    }
    impl Add for CP32 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.add(xr);
            Self(super::compress(output, 0.0))
        }
    }
    impl Mul for CP32 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self::Output {
            let [xl, _] = super::decompress(self.0);
            let [xr, _] = super::decompress(rhs.0);
            let output = xl.mul(xr);
            Self(super::compress(output, 0.0))
        }
    }
    impl One for CP32 {
        fn one() -> Self {
            panic!("unnecessary boilerplate")
        }
    }

    impl Zero for CP32 {
        fn zero() -> Self {
            Self(super::compress(0.0, 0.0))
        }
        fn is_zero(&self) -> bool {
            panic!("unnecessary boilerplate")
        }
    }

    impl Num for CP32 {
        type FromStrRadixErr = &'static str;
        fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
            Err("UncompressedF32s are bespoke and should not be parsed")
        }
    }

    impl Float for CP32 {
        fn nan() -> Self {
            panic!("unnecessary boilerplate")
        }
        fn infinity() -> Self {
            panic!("unnecessary boilerplate")
        }

        fn neg_infinity() -> Self {
            panic!("unnecessary boilerplate")
        }

        fn neg_zero() -> Self {
            panic!("unnecessary boilerplate")
        }

        fn min_value() -> Self {
            Self(super::compress(-0.5, 0.0))
        }

        fn min_positive_value() -> Self {
            panic!("unnecessary boilerplate")
        }

        fn max_value() -> Self {
            Self(super::compress(0.5, 0.0))
        }

        fn is_nan(self) -> bool {
            false
        }

        fn is_infinite(self) -> bool {
            false
        }

        fn is_finite(self) -> bool {
            true
        }

        fn is_normal(self) -> bool {
            self.decompress() == 0.0
        }

        fn classify(self) -> std::num::FpCategory {
            panic!("unnecessary boilerplate")
        }

        fn floor(self) -> Self {
            Self(super::compress(self.decompress().floor(), 0.0))
        }

        fn ceil(self) -> Self {
            Self(super::compress(self.decompress().ceil(), 0.0))
        }

        fn round(self) -> Self {
            Self(super::compress(self.decompress().round(), 0.0))
        }

        fn trunc(self) -> Self {
            Self(super::compress(self.decompress().trunc(), 0.0))
        }

        fn fract(self) -> Self {
            Self(super::compress(self.decompress().fract(), 0.0))
        }

        fn abs(self) -> Self {
            Self(super::compress(self.decompress().abs(), 0.0))
        }

        fn signum(self) -> Self {
            Self(super::compress(self.decompress().signum(), 0.0))
        }
        fn is_sign_positive(self) -> bool {
            self.decompress().is_sign_positive()
        }

        fn is_sign_negative(self) -> bool {
            self.decompress().is_sign_negative()
        }

        fn mul_add(self, a: Self, b: Self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn recip(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn powi(self, n: i32) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn powf(self, n: Self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn sqrt(self) -> Self {
            Self(super::compress(self.decompress().sqrt(), 0.0))
        }

        fn exp(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn exp2(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn ln(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn log(self, base: Self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn log2(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn log10(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn max(self, other: Self) -> Self {
            Self(super::compress(
                self.decompress().max(other.decompress()),
                0.0,
            ))
        }

        fn min(self, other: Self) -> Self {
            Self(super::compress(
                self.decompress().min(other.decompress()),
                0.0,
            ))
        }

        fn abs_sub(self, other: Self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn cbrt(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn hypot(self, other: Self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn sin(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn cos(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn tan(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn asin(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn acos(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn atan(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn atan2(self, other: Self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn sin_cos(self) -> (Self, Self) {
            panic!("unnecessary boilerplate")
        }

        fn exp_m1(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn ln_1p(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn sinh(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn cosh(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn tanh(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn asinh(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn acosh(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn atanh(self) -> Self {
            panic!("unnecessary boilerplate")
        }

        fn integer_decode(self) -> (u64, i16, i8) {
            panic!("unnecessary boilerplate")
        }
    }
}
