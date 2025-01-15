#include "../include/03_gpu_conv2d_tiled.cuh"

__global__ void gpu_conv2d_tiled_kernel(float *d_N_ptr, float *d_P_ptr, int n_rows, int n_cols)
{
    // Which output element this thread works on
    int out_col = blockIdx.x*OUTPUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int out_row = blockIdx.y*OUTPUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Allocate shared memory
    __shared__ float N_sh[INPUT_TILE_DIM][INPUT_TILE_DIM];

    // Checking for ghost cells and loading tiles into shared memory
    if (out_row >= 0 && out_row < n_rows && out_col >= 0 && out_col < n_cols)
        N_sh[threadIdx.y][threadIdx.x] = d_N_ptr[out_row*n_cols + out_col];
    else
        N_sh[threadIdx.y][threadIdx.x] = 0.0f;
    
    // Ensure all elements are loaded
    __syncthreads();

    // Computing output elements
    int tile_col = threadIdx.x - FILTER_RADIUS;
    int tile_row = threadIdx.y - FILTER_RADIUS;
    
    // Check if output element is valid
    if (out_row >= 0 && out_row < n_rows && out_col >= 0 && out_col < n_cols) 
    {
        // Checking for threads outside the tile bounds
        if (tile_row >= 0 && tile_row < OUTPUT_TILE_DIM && tile_col >= 0 && tile_col < OUTPUT_TILE_DIM) 
        {
            // Result (in thread register)
            float p_val = 0.0f;
            
            // Loop over elements of the filter array
            #pragma unroll
            for (int f_row = 0; f_row < 2*FILTER_RADIUS+1; f_row++) 
            {
                for (int f_col = 0; f_col < 2*FILTER_RADIUS+1; f_col++) 
                {
                    // Input element (in shared memory) to filter element mapping
                    int in_row = tile_row + f_row;
                    int in_col = tile_col + f_col;
                
                    p_val += d_F[f_row*(2*FILTER_RADIUS+1) + f_col] * N_sh[in_row][in_col];
                }
            }

            // Storing the final result
            d_P_ptr[out_row*n_cols + out_col] = p_val;
        }
    }
}