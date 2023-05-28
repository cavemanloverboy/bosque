pub mod abacussummit;
pub mod mirror_select;
pub mod tree;
pub mod treef32;

/// A super simple 3-pt tree used for testing and proof of concept.
#[derive(Debug, PartialEq)]
#[repr(C)]
pub struct MockTree<const D: usize> {
    pub root: [f32; D],
    pub left: [f32; D],
    pub right: [f32; D],
}

impl<const D: usize> MockTree<D> {
    pub fn new(root: [f32; D], left: [f32; D], right: [f32; D]) -> MockTree<D> {
        MockTree { root, left, right }
    }
}

#[derive(Debug, rkyv::Archive)]
#[repr(C)]
#[archive_attr(derive(Debug), repr(C))]
#[cfg(feature = "uncompressed")]
/// A super simple 3-pt tree used for testing and proof of concept.
pub struct CompressedTree<C: rkyv::Archive, const D: usize>
where
    <[C; D] as rkyv::Archive>::Archived: std::fmt::Debug,
{
    pub root: [C; D],
    pub left: [C; D],
    pub right: [C; D],
}
