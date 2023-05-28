#include "bosque.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

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

long long timeInMicros(void) {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000*1000)+(tv.tv_usec);
}

int main() {
    // Allocate arrays for mock AbacusSummit sim data
    long long start_init_data = timeInMicros();
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
    long long end_init_data = timeInMicros();
    double init_data_time = ((double) end_init_data - start_init_data) / 1000 / 1000;
    printf("Initialized mock data in %f seconds\n", init_data_time);

    // Compress values
    long long start_compress = timeInMicros();
    CP32* cpos = malloc(3 * NUM_POINTS * sizeof(CP32));
    for (int i = 0; i < NUM_POINTS; i++) {
        cpos[3*i]._0 = compress(pos[i].x, vel[i].x);
        cpos[3*i+1]._0 = compress(pos[i].y, vel[i].y);
        cpos[3*i+2]._0 = compress(pos[i].z, vel[i].z);
    }
    long long end_compress = timeInMicros();
    double compress_time = ((double) end_compress - start_compress) / 1000 / 1000;
    printf("Compressed mock data in %f seconds\n", compress_time);

    // Initialize index array
    Index* idx = malloc(NUM_POINTS * sizeof(Index));
    for (int i = 0; i < NUM_POINTS; i++) {
        idx[i] = i;
    }

    // Create tree on compressed data
    long long start_construct = timeInMicros();
    construct_compressed_tree(cpos, NUM_POINTS, idx);
    long long end_construct = timeInMicros();
    double construct_time = ((double) end_construct - start_construct) / 1000 / 1000;;
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
    long long start_queries = timeInMicros();
    const QueryNearest* nearest = query_compressed_nearest(cpos, NUM_POINTS, queries, NUM_QUERIES);
    long long stop_queries = timeInMicros();
    long long query_time = (stop_queries - start_queries) / 1000;
    printf("Carried out %d queries in %lld millis\n", NUM_QUERIES, query_time);

    long long start_par_queries = timeInMicros();
    const QueryNearest* nearest_par = query_compressed_nearest_parallel(cpos, NUM_POINTS, queries, NUM_QUERIES);
    long long stop_par_queries = timeInMicros();
    long long par_query_time = (stop_par_queries - start_par_queries) / 1000;
    printf("Carried out %d queries in parallel in %lld millis\n", NUM_QUERIES, par_query_time);

    free(pos);
    free(vel);
    free(idx);
    free(cpos);
    free(queries);
}