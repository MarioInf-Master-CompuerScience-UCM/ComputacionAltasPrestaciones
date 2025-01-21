#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include<stdbool.h>

#include "colormap.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Simulation parameters
#pragma omp declare target
static const unsigned int N = 2048;
static const unsigned int numElement = N*N;

static const float SOURCE_TEMP   = 100.0f;
static const float ENVIROM_TEMP  = 25.0f;
static const float BOUNDARY_TEMP = 5.0f;

static const float MIN_DELTA = 0.01f;
static const unsigned int MAX_ITERATIONS = 2000;
#pragma omp end declare target


static void init(unsigned int source_x, unsigned int source_y, float * matrix) {

    #pragma omp parallel for schedule(guided)
    for (unsigned int y = 0; y < N; ++y){
        #pragma unroll(4)
        for (unsigned int x = 0; x < N; ++x){
            matrix[y*N+x]=ENVIROM_TEMP;
        }
    }
    matrix[source_y*N+source_x] = SOURCE_TEMP;
    
    #pragma omp parallel for schedule(guided)
    #pragma unroll(4)
    for (unsigned int x = 0; x < N; ++x) {
        matrix[        x] = BOUNDARY_TEMP;
        matrix[(N-1)*N+x] = BOUNDARY_TEMP;
    }

    #pragma omp parallel for schedule(guided)
    #pragma unroll(4)
    for (unsigned int y = 0; y < N; ++y) {
        matrix[y*N    ] = BOUNDARY_TEMP;
        matrix[y*N+N-1] = BOUNDARY_TEMP;
    }
}

#pragma omp declare target
static void step(unsigned int source_x, unsigned int source_y, const float *restrict current, float *restrict next) {

    float a = 0.5f; //Diffusion constant
    float dx = 0.01f; float dx2 = dx*dx;
    float dy = 0.01f; float dy2 = dy*dy;
    float dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
    int tempIndex, tempIndex_lastY, tempIndex_nexttY;
    unsigned int y, x;
    
    #pragma omp target teams distribute parallel for collapse(2) private(tempIndex, tempIndex_lastY, tempIndex_nexttY, y, x) shared(next)
    for (y = 1; y < N-1; ++y) {
        for (x = 1; x < N-1; ++x) {       
            tempIndex = y*N+x;
            tempIndex_lastY = (y-1)*N+x;
            tempIndex_nexttY = (y+1)*N+x;
            next[tempIndex] = current[y*N+x] + a * dt *
				((current[tempIndex+1] - 2.0*current[tempIndex] + current[tempIndex-1])/dx2 +
				 (current[tempIndex_nexttY] - 2.0*current[tempIndex] + current[tempIndex_lastY])/dy2);
        }
    }
    #pragma omp target
    {
        next[source_y*N+source_x] = SOURCE_TEMP;
    }

}
#pragma omp end declare target

#pragma omp declare target
static float diff(const float *restrict current, const float *restrict next) {
    float maxdiff = 0.0f;
    float maxdiff_thread = 0.0f;


    for (unsigned int y = 1; y < N-1; ++y) {

        for (unsigned int x = 1; x < N-1; ++x) {
            maxdiff_thread = fmaxf(maxdiff, fabsf(next[y*N+x] - current[y*N+x]));
        }
        

        {
        maxdiff = fmaxf(maxdiff_thread, maxdiff);
        }
    }
    return maxdiff;
}
#pragma omp end declare target

void write_png(float * current, int iter) {
    char file[100];
    uint8_t * image = malloc(3 * N * N * sizeof(uint8_t));
    float maxval = fmaxf(SOURCE_TEMP, BOUNDARY_TEMP);

    #pragma omp parallel for schedule(guided)
    for (unsigned int y = 0; y < N; ++y) {
        for (unsigned int x = 0; x < N; ++x) {
            unsigned int i = y*N+x;
            unsigned int idx1=3*i, idx2=3*i+1, idx3=3*i+2;
            colormap_rgb(COLORMAP_MAGMA, current[i], 0.0f, maxval, &image[idx1], &image[idx2], &image[idx3]);
        }
    }
    sprintf(file,"heat%i.png", iter);
    stbi_write_png(file, N, N, 3, image, 3 * N);

    free(image);
}


int main() {
    
    printf("Dispositivos detectados: %d\n", omp_get_num_devices());
    
    size_t array_size = N * N * sizeof(float);
    float * current = malloc(array_size);
    float * next = malloc(array_size);

    srand(0);
    unsigned int source_x = rand() % (N-2) + 1;
    unsigned int source_y = rand() % (N-2) + 1;
    unsigned int it = 0;

    printf("Heat source at (%u, %u)\n", source_x, source_y);
    init(source_x, source_y, current);
    init(source_x, source_y, next);
    double start = omp_get_wtime();
    float t_diff = SOURCE_TEMP;

        #pragma omp target enter data map(to: source_x, source_y, t_diff, current [0:numElement], next [0:numElement], it, MAX_ITERATIONS, MIN_DELTA)
        for (it = 0; (it < MAX_ITERATIONS) && (t_diff > MIN_DELTA); ++it) {
            step(source_x, source_y, current, next);
            t_diff = diff(current, next);
            if(it%(MAX_ITERATIONS/40)==0){
                printf("%u: %f\n", it, t_diff);
            }

            float * swap = current;
            current = next;
            next = swap;
        }
        #pragma omp target exit data map(from : current [0:numElement], next [0:numElement], it)

    double stop = omp_get_wtime();
    printf("Computing time %f s.\n", stop-start);
    write_png(current, it);
    free(current);
    free(next);

    return 0;
}
