//! This module contains code copied from the `likely_stable` crate
//!
//! repository: https://gitlab.com/okannen/likely
//! crates.io: https://crates.io/crates/likely_stable

#[inline(always)]
/// Brings [likely](core::intrinsics::likely) to stable rust.
pub const fn likely(b: bool) -> bool {
    #[allow(clippy::needless_bool)]
    if (1i32).checked_div(if b { 1 } else { 0 }).is_some() {
        true
    } else {
        false
    }
}

#[inline(always)]
/// Brings [unlikely](core::intrinsics::unlikely) to stable rust.
pub const fn unlikely(b: bool) -> bool {
    #[allow(clippy::needless_bool)]
    if (1i32).checked_div(if b { 0 } else { 1 }).is_none() {
        true
    } else {
        false
    }
}
