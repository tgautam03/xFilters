#include "../include/02_gpu_conv2d_constMem.cuh"

__global__ void gpu_conv2d_constMem_kernel(float *d_N_ptr, float *d_P_ptr, int n_rows, int n_cols)
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
        for (int f_row = 0; f_row < 2*FILTER_RADIUS+1; f_row++) 
        {
            for (int f_col = 0; f_col < 2*FILTER_RADIUS+1; f_col++) 
            {
                // Input element to filter element mapping
                int in_row = out_row + (f_row - FILTER_RADIUS);
                int in_col = out_col + (f_col - FILTER_RADIUS);
                
                // Boundary check
                if (in_row >= 0 && in_row < n_rows && in_col >= 0 && in_col < n_cols) 
                    p_val += d_F[f_row][f_col] * d_N_ptr[in_row*n_cols + in_col];
            }
        }
        d_P_ptr[out_row*n_cols + out_col] = p_val;
    }
}

void gpu_conv2d_constMem(float *d_N_ptr, float *F, float *d_P_ptr, int n_rows, int n_cols)
{
    // Move filter to GPU constant memory (Destination, Source, Size)
    cudaError_t err = cudaMemcpyToSymbol(d_F, F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    cuda_check(err);

    // Kernel execution
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(n_rows/(float)(32)), ceil(n_cols/(float)(32)), 1);

    gpu_conv2d_constMem_kernel<<<dim_grid, dim_block>>>(d_N_ptr, d_P_ptr, n_rows, n_cols);
}