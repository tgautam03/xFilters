#include <iostream>
#include <chrono>
#include <exception>
#include <assert.h>
#include <random>
#include <iomanip>
#include <fstream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#define FILTER_RADIUS 1
#define INPUT_TILE_DIM 16
#define OUTPUT_TILE_DIM (INPUT_TILE_DIM - 2*FILTER_RADIUS)

__constant__ float d_F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}

void to_grayscale(float* img, float* gray_img, int new_size, int n_rows, int n_cols, int channels)
{
    // Convert to grayscale using weighted average
    #pragma unroll
    for (int i = 0; i < n_rows; ++i) 
    {
        #pragma unroll
        for (int j = 0; j < n_cols; ++j) 
        {
            int idx = (i * n_cols + j) * channels; // Index in the original image data
            // Calculate grayscale value using weighted average
            gray_img[i * new_size + j] = 0.299f * static_cast<float>(img[idx]) +   // Red channel
                                        0.587f * static_cast<float>(img[idx + 1]) + // Green channel
                                        0.114f * static_cast<float>(img[idx + 2])   // Blue channel
            ;
        }
    }
}

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
                #pragma unroll
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

int main(int argc, char const *argv[])
{
    // ---------------------------------------------------------- //
    // ------------------ Load image in memory ------------------ //
    // ---------------------------------------------------------- //
    std::string img_loc = "../data/8k.jpg";

    // Getting frame
    int n_cols, n_rows, channels;
    float *img = stbi_loadf(img_loc.c_str(), &n_cols, &n_rows, &channels, 0);

    std::cout << "Loaded image with Width: " << n_cols << " and Height: " << n_rows << "\n";
    std::cout << "\n";

    // Determine size for square image (preprocessing variables)
    int new_size = std::max(n_rows, n_cols);

    // Allocate memory for the grayscale image (input and output)
    float *N = new float[new_size * new_size];
    float *P = new float[new_size * new_size];

    // Convert to grayscale using weighted average and store in memory
    to_grayscale(img, N, new_size, n_rows, n_cols, channels);

    // ---------------------------------------------------------- //
    // ----------------- GPU memory allocation ------------------ //
    // ---------------------------------------------------------- //
    cudaError_t err;
    
    std::cout << "Allocating GPU memory... \n";
    
    float* d_N;
    err = cudaMalloc((void**) &d_N, new_size*new_size*sizeof(float));
    cuda_check(err);

    float *d_P; 
    err = cudaMalloc((void**) &d_P, new_size*new_size*sizeof(float));
    cuda_check(err);

    // ---------------------------------------------------------- //
    // ------------------- Move input to GPU -------------------- //
    // ---------------------------------------------------------- //
    std::cout << "Moving input to GPU memory... \n";
    
    err = cudaMemcpy(d_N, N, new_size*new_size*sizeof(float), cudaMemcpyHostToDevice);
    cuda_check(err);

    // ------------------------------------------------------------------------- //
    // ----------------------- Initialize filter ------------------------------- //
    // ------------------------------------------------------------------------- //
    std::string filter_type;
    float *F = new float[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];
    F[0] = -1;
    F[1] = -1;
    F[2] = -1;
    F[3] = -1;
    F[4] = 8;
    F[5] = -1;
    F[6] = -1;
    F[7] = -1;
    F[8] = -1;

    // ---------------------------------------------------------- //
    // ------------------ Move filter to GPU -------------------- //
    // ---------------------------------------------------------- //
    std::cout << "Moving filter to GPU memory... \n";
    
    err = cudaMemcpyToSymbol(d_F, F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    cuda_check(err);

    // ---------------------------------------------------------- //
    // --------------------- 2D Convolution --------------------- //
    // ---------------------------------------------------------- //
    float elapsed_time_kernel;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Applying filters
    std::cout << "Applying filter... \n"; 

    dim3 dim_block(INPUT_TILE_DIM, INPUT_TILE_DIM, 1);
    dim3 dim_grid(ceil(new_size/(float)(OUTPUT_TILE_DIM)), ceil(new_size/(float)(OUTPUT_TILE_DIM)), 1);
    // for (size_t i = 0; i < 10; i++)
    //     gpu_conv2d_tiled_kernel<<<dim_grid, dim_block>>>(d_N, d_P, new_size, new_size);

    int n_runs = 1;
    cudaEventRecord(beg);
    for (int i = 0; i < n_runs; i++)
    {
        gpu_conv2d_tiled_kernel<<<dim_grid, dim_block>>>(d_N, d_P, new_size, new_size);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time_kernel, beg, end);
    elapsed_time_kernel /= (1000. * n_runs);
    std::cout << "Time for kernel execution (seconds): " << elapsed_time_kernel << "\n";
    std::cout << "\n";

    // ---------------------------------------------------------- //
    // ---------- Copying result back to host memory -------------//
    // ---------------------------------------------------------- //
    std::cout << "Moving result to CPU memory... \n";
    
    err = cudaMemcpy(P, d_P, new_size*new_size*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);

    // ----------------------------------------------------------------- //
    // -------------------- Saving output as jpg ----------------------- //
    // ----------------------------------------------------------------- //
    // Convert float data to unsigned char for saving as PNG
    std::vector<unsigned char> ucharData(n_rows*n_cols);
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
            ucharData[i*n_cols+j] = static_cast<unsigned char>(std::min(std::max(P[i*new_size+j], 0.0f), 1.0f) * 255.0f);   
    }

    // Write the output image to a file
    stbi_write_png("../data/output_img.png", n_cols, n_rows, 1, ucharData.data(), n_cols);

    delete[] N;
    delete[] F;
    delete[] P;

    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);

    return 0;
}
