/* These bindings are autogenerated via cbindgen */

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

/**
 * An auxiliary ffi type for multi-value returns.
 *
 * # SAFETY
 * If the fields of this type ever change review the extern "C" functions in this module for safety.
 */
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

/**
 * Builds a compressed tree made up of the `num_points` points in `flat_data_ptr` inplace.
 *
 * # Safety
 * Slices to the data are made from these raw parts. This pointer and length must be
 * correct and valid.
 */
void construct_compressed_tree(struct CP32 *flat_data_ptr, uint64_t num_points, Index *idxs_ptr);

/**
 * Builds a compressed tree made up of the `num_points` points in `flat_data_ptr` inplace.
 *
 * # Safety
 * Slices to the data are made from these raw parts. This pointer and length must be
 * correct and valid.
 */
void construct_tree_f32(float *flat_data_ptr, uint64_t num_points, Index *idxs_ptr);

/**
 * Queries a compressed tree made up of the `num_points` points in `flat_data_ptr` for the nearest neighbor.
 *
 * # Safety
 * Slices to the data and queries are made from these raw parts. These pointers and lengths must be
 * correct and valid.
 */
const struct QueryNearest *query_compressed_nearest(const struct CP32 *flat_data_ptr,
                                                    uint64_t num_points,
                                                    const float *flat_query_ptr,
                                                    uint64_t num_queries);

/**
 * Queries a f32 tree made up of the `num_points` points in `flat_data_ptr` for the nearest neighbor.
 *
 * # Safety
 * Slices to the data and queries are made from these raw parts. These pointers and lengths must be
 * correct and valid.
 */
const struct QueryNearest *query_f32_nearest(const float *flat_data_ptr,
                                             uint64_t num_points,
                                             const float *flat_query_ptr,
                                             uint64_t num_queries);

/**
 * Queries a compressed tree made up of the `num_points` points in `flat_data_ptr` for the nearest neighbor.
 * This query is parallelized via rayon
 *
 * # Safety
 * Slices to the data and queries are made from these raw parts. These pointers and lengths must be
 * correct and valid.
 */
const struct QueryNearest *query_compressed_nearest_parallel(const struct CP32 *flat_data_ptr,
                                                             uint64_t num_points,
                                                             const float *flat_query_ptr,
                                                             uint64_t num_queries);

/**
 * Queries a f32 tree made up of the `num_points` points in `flat_data_ptr` for the nearest neighbor.
 * This query is parallelized via rayon
 *
 * # Safety
 * Slices to the data and queries are made from these raw parts. These pointers and lengths must be
 * correct and valid.
 */
const struct QueryNearest *query_f32_nearest_parallel(const float *flat_data_ptr,
                                                      uint64_t num_points,
                                                      const float *flat_query_ptr,
                                                      uint64_t num_queries);
