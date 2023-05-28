//! This example is similar to cp32_poc, except it uses kiddo instead of a dummy tree struct.
//! This demonstrates tree generation, how to save to disk, zc deserialization, and querying.
//!
//! In the current version of kiddo, the archived tree does not implement query methods.
//! Thils will be available in the next version (see #36 (https://github.com/sdd/kiddo/issues/36)).

use bosque::abacussummit::uncompressed::CP32;
use kiddo::float::distance::squared_euclidean;
use num_format::ToFormattedString;
use rkyv::AlignedVec;

const B: usize = 32;
type Tree = kiddo::float::kdtree::KdTree<CP32, u64, 3, B, u32>;
// type ArchivedTree = kiddo::float::kdtree::ArchivedKdTree<CPF32, usize, 3, B, u32>;

fn main() {
    const D: usize = 3;
    const DATA: usize = 128 * 128 * 128;

    // Build a kiddo tree
    let mut tree = Tree::with_capacity(DATA);
    let mut pts = Vec::with_capacity(DATA);
    for i in 0..DATA {
        let vi = -6000.0 + 12000.0 * i as f32 / (DATA - 1) as f32;
        let pt = [(); D].map(|_| CP32::compress(rand::random::<f32>() - 0.5, vi));
        tree.add(&pt, i as u64);
        pts.push(pt);
    }

    // Query
    let query_compressed = [CP32::compress(0.2, 0.0); D];
    let result = tree.nearest_one(&query_compressed, &squared_euclidean);
    println!("some query: {query_compressed:?} -> {result:?}");

    // Serialize
    // NOTE: At this stage we could save and load the tree. See commented out code in fn.
    let buffer = serialize_to_rkyv(tree);

    let raw_value_bytes = DATA * (D * core::mem::size_of::<CP32>() + core::mem::size_of::<u32>());
    println!(
        "buffer with tree of size {} has length {} vs {} f32/u32 bytes",
        DATA.to_formatted_string(&num_format::Locale::en),
        buffer.len().to_formatted_string(&num_format::Locale::en),
        raw_value_bytes.to_formatted_string(&num_format::Locale::en)
    );

    // ZC Deserialize
    let zc_tree = unsafe { rkyv::archived_root::<Tree>(&buffer) };

    // Query
    let result = zc_tree.nearest_one(&[CP32::compress(0.0, 0.0); D], &squared_euclidean);
    let nn = pts[result.1 as usize];
    println!(
        "kiddo query at origin. NN = {nn:?}, dist2 = {}",
        result.0.decompress()
    );
}

fn serialize_to_rkyv(tree: Tree) -> AlignedVec {
    use rkyv::ser::{
        serializers::{AlignedSerializer, BufferScratch, CompositeSerializer},
        Serializer,
    };
    use rkyv::Infallible;
    let mut serialize_buffer = AlignedVec::with_capacity(tree.size() as usize);
    let mut serialize_scratch = AlignedVec::with_capacity(tree.size() as usize);
    unsafe {
        serialize_buffer.set_len(tree.size() as usize);
        serialize_scratch.set_len(tree.size() as usize);
    }
    serialize_buffer.clear();
    let mut serializer = CompositeSerializer::new(
        AlignedSerializer::new(&mut serialize_buffer),
        BufferScratch::new(&mut serialize_scratch),
        Infallible,
    );
    serializer
        .serialize_value(&tree)
        .expect("Could not serialize with rkyv");
    // use std::io::Write;
    // std::fs::File::create("ser_tree")
    //     .unwrap()
    //     .write_all(&serialize_buffer)
    //     .unwrap();
    serialize_buffer
}
