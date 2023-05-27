#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define BUCKET_SIZE 32

/**
 * Auxiliary struct used for ffi
 */
typedef struct DecompressedPair {
  float pos;
  float vel;
} DecompressedPair;

typedef struct MockTree_3 {
  float root[3];
  float left[3];
  float right[3];
} MockTree_3;

/**
 * A wrapper for the `u32` d which holds the compressed position and velocity.
 * It provides utilities for using the positional information.
 *
 * Compressed Position 32-bit -> `CP32`.
 */
typedef struct CP32 {
  uint32_t _0;
} CP32;

typedef uint32_t Index;

typedef struct QueryNearest {
  float dist_sq;
  uint64_t idx_within;
} QueryNearest;

/**
 * Compress a position/velocity `f32/f32` pair in [-0.5, 0.5] x [-6000, 6000] into a `u32`.
 * For AbacusSummit, this corresponds to simulation units for position and km/s for velocity.
 */
uint32_t compress(float position, float velocity);

/**
 * Decompresses a `u32` dword into  a position/velocity `f32/f32` pair in [-0.5, 0.5] x [-6000, 6000].
 * For AbacusSummit, this corresponds to simulation units for position and km/s for velocity.
 */
struct DecompressedPair decompress(uint32_t compressed);

struct MockTree_3 from_abacussummit_compressed(const uint8_t *compressed_bytes,
                                               uintptr_t bytes_len);

void pretty_print_tree(const char *prefix, const struct MockTree_3 *tree);

struct MockTree_3 new_abacus(const float (*root)[3],
                             const float (*left)[3],
                             const float (*right)[3]);

void construct_compressed_tree(struct CP32 *flat_data_ptr, uint64_t num_points, Index *idxs_ptr);

const struct QueryNearest *query_compressed_nearest(const struct CP32 *flat_data_ptr,
                                                    uint64_t num_points,
                                                    const float *flat_query_ptr,
                                                    uint64_t num_queries);
