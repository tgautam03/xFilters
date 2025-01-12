#ifndef GPU_CONV2D
#define GPU_CONV2D

#define FILTER_RADIUS 1

__global__ void gpu_conv2d_kernel(float *d_N_ptr, float *d_F_ptr, float *d_P_ptr, int n_rows, int n_cols);

#endif