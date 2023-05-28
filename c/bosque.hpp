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

void construct_compressed_tree(CP32 *flat_data_ptr, uint64_t num_points, Index *idxs_ptr);

/// Queries a compressed tree whose
const QueryNearest *query_compressed_nearest(const CP32 *flat_data_ptr,
                                             uint64_t num_points,
                                             const float *flat_query_ptr,
                                             uint64_t num_queries);

const QueryNearest *query_compressed_nearest_parallel(const CP32 *flat_data_ptr,
                                                      uint64_t num_points,
                                                      const float *flat_query_ptr,
                                                      uint64_t num_queries);

} // extern "C"
