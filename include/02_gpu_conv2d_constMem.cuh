#ifndef GPU_CONV2D_CONSTMEM
#define GPU_CONV2D_CONSTMEM

#define FILTER_RADIUS 1
extern __constant__ float d_F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

__global__ void gpu_conv2d_constMem_kernel(float *d_N_ptr, float *d_P_ptr, int n_rows, int n_cols);

#endif