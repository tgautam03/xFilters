#include "../include/01_gpu_conv2d.cuh"

__global__ void gpu_conv2d_kernel(float *d_N_ptr, float *d_F_ptr, float *d_P_ptr, 
                                int r, int n_rows, int n_cols)
{
    // Which output element this thread works on
    int out_col = blockIdx.x*blockDim.x + threadIdx.x;
    int out_row = blockIdx.y*blockDim.y + threadIdx.y;
    
    // Check if output element is valid
    if (out_row < n_rows && out_col < n_cols) 
    {
        // Result (in thread register)
        float p_val = 0.0f;
        
        // Loop over elements of the filter array
        for (int f_row = 0; f_row < 2*r+1; f_row++) 
        {
            for (int f_col = 0; f_col < 2*r+1; f_col++) 
            {
                // Input element to filter element mapping
                int in_row = out_row + (f_row - r);
                int in_col = out_col + (f_col - r);
                        
                // Boundary check
                if (in_row >= 0 && in_row < n_rows && in_col >= 0 && in_col < n_cols) 
                    p_val += d_F_ptr[f_row*(2*r+1) + f_col] * d_N_ptr[in_row*n_cols + in_col];
                }
        }
        d_P_ptr[out_row*n_cols + out_col] = p_val;
    }
}

void gpu_conv2d(float *d_N_ptr, float *d_F_ptr, float *d_P_ptr, int r, int n_rows, int n_cols)
{
    // Kernel execution
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(n_rows/(float)(32)), ceil(n_cols/(float)(32)), 1);

    gpu_conv2d_kernel<<<dim_grid, dim_block>>>(d_N_ptr, d_F_ptr, d_P_ptr, r, n_rows, n_cols);
}