//! This module copies some code from the `bytemuck` crate to remove dependencies.
use std::mem::{align_of, size_of};

/// Cast `&mut [A]` into `&mut [B]`.
///
/// ## Panics
///
/// This is [`try_cast_slice_mut`] but will panic on error.
#[inline]
pub fn cast_slice_mut<A, B>(a: &mut [A]) -> &mut [B] {
    // Note(Lokathor): everything with `align_of` and `size_of` will optimize away
    // after monomorphization.
    if align_of::<B>() > align_of::<A>() && (a.as_mut_ptr() as usize) % align_of::<B>() != 0 {
        panic!(
            "You tried to cast a slice to an element type with a higher alignment \
            requirement but the slice wasn't aligned."
        )
    } else if size_of::<B>() == size_of::<A>() {
        unsafe { core::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut B, a.len()) }
    } else if size_of::<A>() == 0 || size_of::<B>() == 0 {
        panic!(
            "For this type of cast the alignments must be exactly the same and they \
             were not so now you're sad. \
             This error is generated **only** by operations that cast allocated types \
             (such as `Box` and `Vec`), because in that case the alignment must stay \
             exact."
        )
    } else if core::mem::size_of_val(a) % size_of::<B>() == 0 {
        let new_len = core::mem::size_of_val(a) / size_of::<B>();
        unsafe { core::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut B, new_len) }
    } else {
        panic!(
            "If the element size changes then the output slice changes length \
        accordingly. If the output slice wouldn't be a whole number of elements \
        then the conversion fails."
        )
    }
}

/// Cast `&[A]` into `&[B]`.
///
/// ## Panics
///
/// This is [`try_cast_slice`] but will panic on error.
#[inline]
pub fn cast_slice<A, B>(a: &[A]) -> &[B] {
    // Note(Lokathor): everything with `align_of` and `size_of` will optimize away
    // after monomorphization.
    if align_of::<B>() > align_of::<A>() && (a.as_ptr() as usize) % align_of::<B>() != 0 {
        panic!(
            "You tried to cast a slice to an element type with a higher alignment \
            requirement but the slice wasn't aligned."
        )
    } else if size_of::<B>() == size_of::<A>() {
        unsafe { core::slice::from_raw_parts(a.as_ptr() as *mut B, a.len()) }
    } else if size_of::<A>() == 0 || size_of::<B>() == 0 {
        panic!(
            "For this type of cast the alignments must be exactly the same and they \
             were not so now you're sad. \
             This error is generated **only** by operations that cast allocated types \
             (such as `Box` and `Vec`), because in that case the alignment must stay \
             exact."
        )
    } else if core::mem::size_of_val(a) % size_of::<B>() == 0 {
        let new_len = core::mem::size_of_val(a) / size_of::<B>();
        unsafe { core::slice::from_raw_parts(a.as_ptr() as *mut B, new_len) }
    } else {
        panic!(
            "If the element size changes then the output slice changes length \
        accordingly. If the output slice wouldn't be a whole number of elements \
        then the conversion fails."
        )
    }
}

// This is here just for copying error messages
// /// The things that can go wrong when casting between [`Pod`] data forms.
// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// pub enum PodCastError {
//     /// You tried to cast a slice to an element type with a higher alignment
//     /// requirement but the slice wasn't aligned.
//     TargetAlignmentGreaterAndInputNotAligned,
//     /// If the element size changes then the output slice changes length
//     /// accordingly. If the output slice wouldn't be a whole number of elements
//     /// then the conversion fails.
//     OutputSliceWouldHaveSlop,
//     /// When casting a slice you can't convert between ZST elements and non-ZST
//     /// elements. When casting an individual `T`, `&T`, or `&mut T` value the
//     /// source size and destination size must be an exact match.
//     SizeMismatch,
//     /// For this type of cast the alignments must be exactly the same and they
//     /// were not so now you're sad.
//     ///
//     /// This error is generated **only** by operations that cast allocated types
//     /// (such as `Box` and `Vec`), because in that case the alignment must stay
//     /// exact.
//     AlignmentMismatch,
// }
