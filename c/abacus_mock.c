#include "bosque.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//#define NUM_POINTS 512*512*512
//#define NUM_POINTS 128*128*128
#define NUM_POINTS 100 * 1000
#define NUM_QUERIES 1000 * 1000

typedef struct {
    float x;
    float y;
    float z;
} Point;


float generateRandomFloat() {
    return (float)rand() / RAND_MAX;
}

int main() {
    // Allocate arrays for mock AbacusSummit sim data
    clock_t start_init_data = clock();
    Point* pos = malloc(NUM_POINTS * sizeof(Point));
    Point* vel = malloc(NUM_POINTS * sizeof(Point));

    // Populate with randoms in allowed range
    for (int i = 0; i < NUM_POINTS; i++) {
        // Position in [-0.5, 0.5]
        pos[i].x = generateRandomFloat() - 0.5;
        pos[i].y = generateRandomFloat() - 0.5;
        pos[i].z = generateRandomFloat() - 0.5;
    
        // Velocity in [-6000.0, 6000.0]
        vel[i].x = 6000 * (generateRandomFloat() - 0.5);
        vel[i].y = 6000 * (generateRandomFloat() - 0.5);
        vel[i].z = 6000 * (generateRandomFloat() - 0.5);
    }
    clock_t end_init_data = clock();
    double init_data_time = ((double) end_init_data - start_init_data) / CLOCKS_PER_SEC;
    printf("Initialized mock data in %f seconds\n", init_data_time);

    // Compress values
    clock_t start_compress = clock();
    CP32* cpos = malloc(3 * NUM_POINTS * sizeof(CP32));
    for (int i = 0; i < NUM_POINTS; i++) {
        cpos[3*i]._0 = compress(pos[i].x, vel[i].x);
        cpos[3*i+1]._0 = compress(pos[i].y, vel[i].y);
        cpos[3*i+2]._0 = compress(pos[i].z, vel[i].z);
    }
    clock_t end_compress = clock();
    double compress_time = ((double) end_compress - start_compress) / CLOCKS_PER_SEC;
    printf("Compressed mock data in %f seconds\n", compress_time);

    // Initialize index array
    Index* idx = malloc(NUM_POINTS * sizeof(Index));
    for (int i = 0; i < NUM_POINTS; i++) {
        idx[i] = i;
    }

    // Create tree on compressed data
    clock_t start_construct = clock();
    construct_compressed_tree(cpos, NUM_POINTS, idx);
    clock_t end_construct = clock();
    double construct_time = ((double) end_construct - start_construct) / CLOCKS_PER_SEC;
    printf("Constructed kdtree inplace in %f seconds\n", construct_time);

    // Query near origin
    float query[3] = {0.0f, 0.0f, 0.0f};
    const QueryNearest* result = query_compressed_nearest(cpos, NUM_POINTS, query, 1);
    printf(
        "Queried at origin: (%f, %f, %f) -> euclidean distance %f\n",
        decompress(cpos[3*result->idx_within]._0).pos,
        decompress(cpos[3*result->idx_within+1]._0).pos,
        decompress(cpos[3*result->idx_within+2]._0).pos,
        sqrt(result->dist_sq)
    );

    // Initialize queries
    float *queries = malloc(3 * NUM_QUERIES * sizeof(float));
    if (queries == NULL) {
        fprintf(stderr, "Failed to allocate memory for queries.\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < NUM_QUERIES; i++) {
        queries[3*i] = (float)rand() / (float)RAND_MAX;
        queries[3*i+1] = (float)rand() / (float)RAND_MAX;
        queries[3*i+2] = (float)rand() / (float)RAND_MAX;
    }
    printf("Initialized queries\n");

    // Query tree many times
    clock_t start_queries = clock();
    const QueryNearest* nearest = query_compressed_nearest(cpos, NUM_POINTS, queries, NUM_QUERIES);
    clock_t end_queries = clock();
    double query_time = ((double) end_queries - start_queries) / CLOCKS_PER_SEC;
    printf("Carried out %d queries in %f millis\n", NUM_QUERIES, 1000 * query_time);

    clock_t start_queries = clock();
    const QueryNearest* nearest = query_compressed_nearest_parallel(cpos, NUM_POINTS, queries, NUM_QUERIES);
    clock_t end_queries = clock();
    double query_time = ((double) end_queries - start_queries) / CLOCKS_PER_SEC;
    printf("Carried out %d queries in parallel in %f millis\n", NUM_QUERIES, 1000 * query_time);

    free(pos);
    free(vel);
    free(idx);
    free(cpos);
    free(queries);
}