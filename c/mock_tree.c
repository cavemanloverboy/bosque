#include "bosque.h"
#include <stdio.h>

typedef struct {
    float x;
    float y;
    float z;
} Point;

int main() {
    // Define the size of the arrays
    int num = 3;

    // Initialize position and velocity arrays
    Point positions[num];
    Point velocities[num];
    // Positions: A kDTree algorithm will produce some permutation of the particles
    // that uniquely define a queryable kDtree
    positions[0].x = 0.0; positions[0].y = 0.0; positions[0].z = 0.0;
    positions[1].x = 0.125; positions[1].y = 0.125; positions[1].z = 0.125;
    positions[2].x = -0.23; positions[2].y = -0.23; positions[2].z = -0.23;
    printf("Initialized Position Arrays:\n");
    for (int i = 0; i < num; i++) {
        printf("Position %d: (%.2f, %.2f, %.2f)\n", i + 1, positions[i].x, positions[i].y, positions[i].z);
    }

    // Velocities
    velocities[0].x = 0.0; velocities[0].y = 0.0; velocities[0].z = 0.0;
    velocities[1].x = 2000.0; velocities[1].y = 2000.0; velocities[1].z = 2000.0;
    velocities[2].x = -400.0; velocities[2].y = -400.0; velocities[2].z = -400.0;
    printf("\nInitialized Velocity Arrays:\n");
    for (int i = 0; i < num; i++) {
        printf("Velocity %d: (%.2f, %.2f, %.2f)\n", i + 1, velocities[i].x, velocities[i].y, velocities[i].z);
    }

    // Compressed u32 array
    uint32_t* compressed_array;
    int dwords = num*3;
    compressed_array = (uint32_t*)malloc(dwords * sizeof(uint32_t));
    for (int i = 0; i < num; i++) {
        compressed_array[0 + 3*i] = compress(positions[i].x, velocities[i].x);
        compressed_array[1 + 3*i] = compress(positions[i].y, velocities[i].y);
        compressed_array[2 + 3*i] = compress(positions[i].z, velocities[i].z);
    }
    printf("\nCompressed array into 32-bit words:\n");
    for (int i = 0; i < dwords; i++) {
        if (i % 3 == 2) {
            printf("%12d\n", compressed_array[i]);
        } else {
            printf("%12d ", compressed_array[i]);
        }
    }

    // Uncompressed tree
    MockTree_3 uncompressed_tree = new_abacus(&positions[0], &positions[1], &positions[2]);
    pretty_print_tree("\nuncompressed: ", &uncompressed_tree);

    // Decompress tree
    uint8_t* compressed_bytes = (uint8_t*)compressed_array;
    MockTree_3 decompressed_tree = from_abacussummit_compressed(compressed_bytes, dwords * sizeof(uint32_t));
    pretty_print_tree("\ndecompressed: ", &decompressed_tree);

    return 0;
}