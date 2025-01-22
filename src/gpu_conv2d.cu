#include "../include/01_gpu_conv2d.cuh"

__global__ void gpu_conv2d_kernel(float const *d_N_ptr, float const *d_F_ptr, float *d_P_ptr, int const n_rows, int const n_cols)
{
    // Which output element this thread works on
    int const out_col = blockIdx.x*blockDim.x + threadIdx.x;
    int const out_row = blockIdx.y*blockDim.y + threadIdx.y;
    
    // Check if output element is valid
    if (out_row < n_rows && out_col < n_cols) 
    {
        // Result (in thread register)
        float p_val = 0.0f;
        
        // Loop over elements of the filter array
        #pragma unroll
        for (int f_row = 0; f_row < 2*FILTER_RADIUS+1; f_row++) 
        {
            for (int f_col = 0; f_col < 2*FILTER_RADIUS+1; f_col++) 
            {
                // Input element to filter element mapping
                int in_row = out_row + (f_row - FILTER_RADIUS);
                int in_col = out_col + (f_col - FILTER_RADIUS);
                        
                // Boundary check
                if (in_row >= 0 && in_row < n_rows && in_col >= 0 && in_col < n_cols) 
                    p_val += d_F_ptr[f_row*(2*FILTER_RADIUS+1) + f_col] * d_N_ptr[in_row*n_cols + in_col];
            }
        }
        d_P_ptr[out_row*n_cols + out_col] = p_val;
    }
}