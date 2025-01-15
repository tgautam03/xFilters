#ifndef GPU_CONV2D
#define GPU_CONV2D

#define FILTER_RADIUS 1

__global__ void gpu_conv2d_kernel(float const *d_N_ptr, float const *d_F_ptr, float *d_P_ptr, int const n_rows, int const n_cols);

#endif