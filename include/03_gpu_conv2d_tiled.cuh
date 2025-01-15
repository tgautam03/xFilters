#ifndef GPU_CONV2D_TILED
#define GPU_CONV2D_TILED

#define FILTER_RADIUS 1
#define INPUT_TILE_DIM 16
#define OUTPUT_TILE_DIM (INPUT_TILE_DIM - 2*FILTER_RADIUS)

extern __constant__ float d_F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

__global__ void gpu_conv2d_tiled_kernel(float *d_N_ptr, float *d_P_ptr, int n_rows, int n_cols);

#endif