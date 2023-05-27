pub mod abacussummit;
pub mod mirror_select;
pub mod tree;
pub mod treef32;

#[derive(Debug)]
#[repr(C)]
pub struct MockTree<const D: usize> {
    pub root: [f32; D],
    pub left: [f32; D],
    pub right: [f32; D],
}

#[derive(Debug, rkyv::Archive)]
#[repr(C)]
#[archive_attr(derive(Debug), repr(C))]
#[cfg(feature = "uncompressed")]
pub struct CompressedTree<C: rkyv::Archive, const D: usize>
where
    <[C; D] as rkyv::Archive>::Archived: std::fmt::Debug,
{
    pub root: [C; D],
    pub left: [C; D],
    pub right: [C; D],
}
