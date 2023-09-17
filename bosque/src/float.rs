use std::{
    collections::BinaryHeap,
    ops::{Add, Mul, Sub},
};

use crate::abacussummit::uncompressed::CP32;

/// The primary purpose of this trait is to be able to make functions and types generic
/// over CP32 and future compression formats, in addition to the std floats `f32` and `f64`.
///
/// For these compressed formats, the accumulator type is a std float.
///
/// The std floats are their own accumulator type.
pub trait TreeFloat: PartialOrd + Send {
    type Accumulator: Mul<Output = Self::Accumulator>
        + Add<Output = Self::Accumulator>
        + Sub<Output = Self::Accumulator>
        + Sub<Output = Self::Accumulator>
        + PartialOrd
        + Ord
        + Copy
        + Sqrt;

    type Query: Mul + Add + Sub + PartialOrd;

    const NAME: &'static str;

    fn new_accumulator_from_query(query: Self::Query) -> Self::Accumulator;

    /// Subtracts self with an accumulator value
    fn sub_accumulator(&self, rhs: &Self::Accumulator) -> Self::Accumulator;

    /// Accumulator's zero
    fn zero() -> Self::Accumulator;

    /// Accumulator's one
    fn one() -> Self::Accumulator;

    /// Accumulator's max
    fn max() -> Self::Accumulator;

    /// Converts the input to an inner accumulator (at beginning of query)
    fn input<const K: usize>(acc: &[Self::Query; K]) -> &[Self::Accumulator; K];

    /// Converts the inner accumulator into the original input type (at end of query)
    fn output(acc: (Self::Accumulator, usize)) -> (Self::Query, usize);

    /// Converts the inner accumulator into the original input type (at end of query k)
    #[inline(always)]
    fn output_bh(mut acc: BinaryHeap<(Self::Accumulator, usize)>) -> Vec<(Self::Query, usize)> {
        // SAFETY: transparent wrapper
        let mut v: Vec<(Self::Accumulator, usize)> = Vec::with_capacity(acc.len());
        let mut write_index = acc.len();
        let orig_len = acc.len();
        while let Some((dist_sq, idx)) = acc.pop() {
            write_index -= 1;
            unsafe {
                if cfg!(not(feature = "sqrt-dist")) {
                    *v.as_mut_ptr().add(write_index) = (dist_sq, idx);
                } else {
                    *v.as_mut_ptr().add(write_index) = (dist_sq.sqrt(), idx);
                }
            }
        }
        unsafe {
            v.set_len(orig_len);
            std::mem::transmute(v)
        }
    }

    /// Converts the inner accumulator into the original input type (at end of query k)
    #[inline(always)]
    fn output_bh_sep(
        mut acc: BinaryHeap<(Self::Accumulator, usize)>,
    ) -> (Vec<Self::Query>, Vec<usize>) {
        // SAFETY: transparent wrapper
        let mut distances: Vec<Self::Accumulator> = Vec::with_capacity(acc.len());
        let mut indices: Vec<usize> = Vec::with_capacity(acc.len());
        let mut write_index = acc.len();
        let orig_len = acc.len();
        while let Some((dist_sq, idx)) = acc.pop() {
            write_index -= 1;
            unsafe {
                if cfg!(not(feature = "sqrt-dist")) {
                    *distances.as_mut_ptr().add(write_index) = dist_sq;
                } else {
                    *distances.as_mut_ptr().add(write_index) = dist_sq.sqrt();
                }
                *indices.as_mut_ptr().add(write_index) = idx;
            }
        }
        unsafe {
            distances.set_len(orig_len);
            indices.set_len(orig_len);
            (std::mem::transmute(distances), std::mem::transmute(indices))
        }
    }

    fn abs(acc: &Self::Accumulator) -> Self::Accumulator;
    fn half(acc: &Self::Accumulator) -> Self::Accumulator;
}

pub trait Sqrt {
    fn sqrt(&self) -> Self;
}

impl TreeFloat for CP32 {
    type Accumulator = F32;
    type Query = f32;

    const NAME: &'static str = "CP32";

    #[inline(always)]
    fn new_accumulator_from_query(query: Self::Query) -> Self::Accumulator {
        F32(query)
    }

    fn sub_accumulator(&self, rhs: &Self::Accumulator) -> Self::Accumulator {
        F32(self.decompress()) - rhs
    }
    #[inline(always)]
    fn zero() -> Self::Accumulator {
        F32(0.0)
    }
    #[inline(always)]
    fn one() -> Self::Accumulator {
        F32(1.0)
    }
    #[inline(always)]
    fn max() -> Self::Accumulator {
        F32(f32::MAX)
    }
    #[inline(always)]
    fn input<const K: usize>(acc: &[Self::Query; K]) -> &[Self::Accumulator; K] {
        // SAFETY: transparent wrapper w/ same lifetime
        unsafe { std::mem::transmute(acc) }
    }
    #[inline(always)]
    fn output((dist_sq, idx): (Self::Accumulator, usize)) -> (Self::Query, usize) {
        // SAFETY: transparent wrapper
        unsafe {
            #[cfg(not(feature = "sqrt-dist"))]
            return std::mem::transmute((dist_sq, idx));
            #[cfg(feature = "sqrt-dist")]
            return std::mem::transmute((dist_sq.sqrt(), idx));
        }
    }

    #[inline(always)]
    fn abs(acc: &Self::Accumulator) -> Self::Accumulator {
        F32(acc.0.abs())
    }
    #[inline(always)]
    fn half(acc: &Self::Accumulator) -> Self::Accumulator {
        F32(acc.0 / 2.0)
    }
}

impl TreeFloat for f32 {
    type Accumulator = F32;
    type Query = f32;

    const NAME: &'static str = "F32";
    #[inline(always)]
    fn new_accumulator_from_query(query: Self::Query) -> Self::Accumulator {
        F32(query)
    }
    #[inline(always)]
    fn sub_accumulator(&self, rhs: &Self::Accumulator) -> Self::Accumulator {
        F32(self - rhs.0)
    }
    #[inline(always)]
    fn zero() -> Self::Accumulator {
        F32(0.0)
    }
    #[inline(always)]
    fn one() -> Self::Accumulator {
        F32(1.0)
    }
    #[inline(always)]
    fn max() -> Self::Accumulator {
        F32(f32::MAX)
    }
    #[inline(always)]
    fn input<const K: usize>(acc: &[Self::Query; K]) -> &[Self::Accumulator; K] {
        // SAFETY: transparent wrapper w/ same lifetime
        unsafe { std::mem::transmute(acc) }
    }
    #[inline(always)]
    fn output((dist_sq, idx): (Self::Accumulator, usize)) -> (Self::Query, usize) {
        // SAFETY: transparent wrapper
        unsafe {
            #[cfg(not(feature = "sqrt-dist"))]
            return std::mem::transmute((dist_sq, idx));
            #[cfg(feature = "sqrt-dist")]
            return std::mem::transmute((dist_sq.sqrt(), idx));
        }
    }

    #[inline(always)]
    fn abs(acc: &Self::Accumulator) -> Self::Accumulator {
        F32(acc.0.abs())
    }
    #[inline(always)]
    fn half(acc: &Self::Accumulator) -> Self::Accumulator {
        F32(acc.0 / 2.0)
    }
}

impl TreeFloat for f64 {
    type Accumulator = F64;
    type Query = f64;

    const NAME: &'static str = "F64";
    #[inline(always)]
    fn new_accumulator_from_query(query: Self) -> Self::Accumulator {
        F64(query)
    }
    #[inline(always)]
    fn sub_accumulator(&self, rhs: &Self::Accumulator) -> Self::Accumulator {
        F64(self - rhs.0)
    }
    #[inline(always)]
    fn zero() -> Self::Accumulator {
        F64(0.0)
    }
    #[inline(always)]
    fn one() -> Self::Accumulator {
        F64(1.0)
    }
    #[inline(always)]
    fn max() -> Self::Accumulator {
        F64(f64::MAX)
    }
    #[inline(always)]
    fn input<const K: usize>(acc: &[Self::Query; K]) -> &[Self::Accumulator; K] {
        // SAFETY: transparent wrapper w/ same lifetime
        unsafe { std::mem::transmute(acc) }
    }
    #[inline(always)]
    fn output((dist_sq, idx): (Self::Accumulator, usize)) -> (Self::Query, usize) {
        // SAFETY: transparent wrapper
        unsafe {
            #[cfg(not(feature = "sqrt-dist"))]
            return std::mem::transmute((dist_sq, idx));
            #[cfg(feature = "sqrt-dist")]
            return std::mem::transmute((dist_sq.sqrt(), idx));
        }
    }

    #[inline(always)]
    fn abs(acc: &Self::Accumulator) -> Self::Accumulator {
        F64(acc.0.abs())
    }
    #[inline(always)]
    fn half(acc: &Self::Accumulator) -> Self::Accumulator {
        F64(acc.0 / 2.0)
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct F32(pub f32);

impl Eq for F32 {}

impl Sqrt for F32 {
    #[inline(always)]
    fn sqrt(&self) -> Self {
        F32(self.0.sqrt())
    }
}

impl Ord for F32 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).expect("you likely had a nan")
    }
}

impl PartialOrd for F32 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Sub for F32 {
    type Output = F32;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        F32(self.0 - rhs.0)
    }
}

impl Mul for F32 {
    type Output = F32;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        F32(self.0 * rhs.0)
    }
}

impl Add for F32 {
    type Output = F32;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        F32(self.0 + rhs.0)
    }
}

impl Sub<&F32> for F32 {
    type Output = F32;
    #[inline(always)]
    fn sub(self, rhs: &Self) -> Self::Output {
        F32(self.0 - rhs.0)
    }
}

impl Mul<&F32> for F32 {
    type Output = F32;
    #[inline(always)]
    fn mul(self, rhs: &Self) -> Self::Output {
        F32(self.0 * rhs.0)
    }
}

impl Add<&F32> for F32 {
    type Output = F32;
    #[inline(always)]
    fn add(self, rhs: &Self) -> Self::Output {
        F32(self.0 + rhs.0)
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct F64(pub f64);

impl Sqrt for F64 {
    #[inline(always)]
    fn sqrt(&self) -> Self {
        F64(self.0.sqrt())
    }
}

impl Eq for F64 {}

impl Ord for F64 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).expect("you likely had a nan")
    }
}

impl PartialOrd for F64 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Sub for F64 {
    type Output = F64;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        F64(self.0 - rhs.0)
    }
}

impl Mul for F64 {
    type Output = F64;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        F64(self.0 * rhs.0)
    }
}

impl Add for F64 {
    type Output = F64;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        F64(self.0 + rhs.0)
    }
}

impl Sub<&F64> for F64 {
    type Output = F64;
    #[inline(always)]
    fn sub(self, rhs: &Self) -> Self::Output {
        F64(self.0 - rhs.0)
    }
}

impl Mul<&F64> for F64 {
    type Output = F64;
    #[inline(always)]
    fn mul(self, rhs: &Self) -> Self::Output {
        F64(self.0 * rhs.0)
    }
}

impl Add<&F64> for F64 {
    type Output = F64;
    #[inline(always)]
    fn add(self, rhs: &Self) -> Self::Output {
        F64(self.0 + rhs.0)
    }
}
