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

__global__ void gpu_conv2d_constMem_kernel(float *d_N_ptr, float *d_P_ptr, int n_rows, int n_cols)
{   
    auto idx = [&n_cols](int y,int x){ return y*n_cols+x; };

    // Which output element this thread works on
    int out_col = blockIdx.x*blockDim.x + threadIdx.x;
    int out_row = blockIdx.y*blockDim.y + threadIdx.y;
    
    // Check if output element is valid
    if (out_row < n_rows && out_col < n_cols) 
    {
        // floor of zero and ceiling of nx/ny-1
        int xl = max(0, out_col-1); int yl = max(0, out_row-1);
        int xh = min(n_cols-1,out_col+1); int yh = min(n_rows-1,out_row+1);
        float p_val = d_F[0]*d_N_ptr[idx(yl,xl)] + d_F[1]*d_N_ptr[idx(yl, out_col)] +
                    d_F[2]*d_N_ptr[idx(yl,xh)] + d_F[3]*d_N_ptr[idx(out_row ,xl)] +
                    d_F[4]*d_N_ptr[idx(out_row , out_col)] + d_F[5]*d_N_ptr[idx(out_row ,xh)] +
                    d_F[6]*d_N_ptr[idx(yh,xl)] + d_F[7]*d_N_ptr[idx(yh, out_col)] +
                    d_F[8]*d_N_ptr[idx(yh,xh)];
        // // Result (in thread register)
        // float p_val = 0.0f;
        
        // // Loop over elements of the filter array
        // #pragma unroll
        // for (int f_row = 0; f_row < 2*FILTER_RADIUS+1; f_row++) 
        // {
        //     #pragma unroll
        //     for (int f_col = 0; f_col < 2*FILTER_RADIUS+1; f_col++) 
        //     {
        //         // Input element to filter element mapping
        //         int in_row = out_row + (f_row - FILTER_RADIUS);
        //         int in_col = out_col + (f_col - FILTER_RADIUS);
                        
        //         // Boundary check
        //         if (in_row >= 0 && in_row < n_rows && in_col >= 0 && in_col < n_cols) 
        //             p_val += d_F_ptr[f_row*(2*FILTER_RADIUS+1) + f_col] * d_N_ptr[in_row*n_cols + in_col];
        //         }
        // }
        d_P_ptr[out_row*n_cols + out_col] = p_val;
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

    dim3 dim_block(16, 16, 1);
    dim3 dim_grid(ceil(new_size/(float)(16)), ceil(new_size/(float)(16)), 1);
    cudaEventRecord(beg);
    gpu_conv2d_constMem_kernel<<<dim_grid, dim_block>>>(d_N, d_P, new_size, new_size);
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time_kernel, beg, end);
    elapsed_time_kernel /= 1000.;
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
