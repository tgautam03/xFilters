#ifndef GPU_CONV2D_CONSTMEM
#define GPU_CONV2D_CONSTMEM

#include <iostream>

#define FILTER_RADIUS 1

// Allocate constant memory for the filter
__constant__ float d_F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void gpu_conv2d_constMem_kernel(float *d_N_ptr, float *d_P_ptr, int n_rows, int n_cols);

void gpu_conv2d_constMem(float *d_N_ptr, float *F, float *d_P_ptr, int n_rows, int n_cols);

#endif