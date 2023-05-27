#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

constexpr static const uintptr_t BUCKET_SIZE = 32;

/// Auxiliary struct used for ffi
struct DecompressedPair {
  float pos;
  float vel;
};

template<uintptr_t D>
struct MockTree {
  float root[D];
  float left[D];
  float right[D];
};

/// A wrapper for the `u32` d which holds the compressed position and velocity.
/// It provides utilities for using the positional information.
///
/// Compressed Position 32-bit -> `CP32`.
struct CP32 {
  uint32_t _0;
};

using Index = uint32_t;

struct QueryNearest {
  float dist_sq;
  uint64_t idx_within;
};

extern "C" {

/// Compress a position/velocity `f32/f32` pair in [-0.5, 0.5] x [-6000, 6000] into a `u32`.
/// For AbacusSummit, this corresponds to simulation units for position and km/s for velocity.
uint32_t compress(float position, float velocity);

/// Decompresses a `u32` dword into  a position/velocity `f32/f32` pair in [-0.5, 0.5] x [-6000, 6000].
/// For AbacusSummit, this corresponds to simulation units for position and km/s for velocity.
DecompressedPair decompress(uint32_t compressed);

MockTree<3> from_abacussummit_compressed(const uint8_t *compressed_bytes, uintptr_t bytes_len);

void pretty_print_tree(const char *prefix, const MockTree<3> *tree);

MockTree<3> new_abacus(const float (*root)[3], const float (*left)[3], const float (*right)[3]);

void construct_compressed_tree(CP32 *flat_data_ptr, uint64_t num_points, Index *idxs_ptr);

const QueryNearest *query_compressed_nearest(const CP32 *flat_data_ptr,
                                             uint64_t num_points,
                                             const float *flat_query_ptr,
                                             uint64_t num_queries);

} // extern "C"
