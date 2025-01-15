#include <iostream>
#include <chrono>
#include <exception>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "../include/02_gpu_conv2d_constMem.cuh"
#include "../include/utils.hpp"

__constant__ float d_F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}

int main(int argc, char const *argv[])
{
    cudaError_t err;
    // Benchmarking variables
    float elapsed_time_mem_alloc, 
            elapsed_time_mem_t_in, elapsed_time_mem_t_f, elapsed_time_mem_t_out, 
            elapsed_time_kernel;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    // ---------------------------------------------------------- //
    // ------------------ Load image in memory ------------------ //
    // ---------------------------------------------------------- //
    std::string img_loc = "data/8k.jpg";

    // Getting frame
    int n_cols, n_rows, channels;
    float *img = stbi_loadf(img_loc.c_str(), &n_cols, &n_rows, &channels, 0);

    std::cout << "Loaded image with Width: " << n_cols << " and Height: " << n_rows << "\n";
    std::cout << "\n";

    // Determine size for square image (preprocessing variables)
    int new_size = std::max(n_rows, n_cols);

    // Allocate memory for the grayscale image (input and output)
    float* N;
    err = cudaMallocHost((void**)&N, new_size*new_size*sizeof(float));
    cuda_check(err);

    float *P;
    err = cudaMallocHost((void**)&P, new_size*new_size*sizeof(float));
    cuda_check(err);

    // Convert to grayscale using weighted average and store in memory
    to_grayscale(img, N, new_size, n_rows, n_cols, channels);

    // Delete img from memory
    delete[] img;

    // ---------------------------------------------------------- //
    // ----------------- GPU memory allocation ------------------ //
    // ---------------------------------------------------------- //
    std::cout << "Allocating GPU memory... \n";
    cudaEventRecord(beg);
    
    float* d_N;
    err = cudaMalloc((void**) &d_N, new_size*new_size*sizeof(float));
    cuda_check(err);

    float *d_P; 
    err = cudaMalloc((void**) &d_P, new_size*new_size*sizeof(float));
    cuda_check(err);

    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time_mem_alloc, beg, end);
    elapsed_time_mem_alloc /= 1000.;

    std::cout << "Time for GPU memory allocation (seconds): " << elapsed_time_mem_alloc << "\n";
    std::cout << "\n";

    // ---------------------------------------------------------- //
    // ------------------- Move input to GPU -------------------- //
    // ---------------------------------------------------------- //
    std::cout << "Moving input to GPU memory... \n";
    cudaEventRecord(beg);
    
    err = cudaMemcpy(d_N, N, new_size*new_size*sizeof(float), cudaMemcpyHostToDevice);
    cuda_check(err);

    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time_mem_t_in, beg, end);
    elapsed_time_mem_t_in /= 1000.;
    std::cout << "Time for input data transfer (seconds): " << elapsed_time_mem_t_in << "\n";
    std::cout << "\n";

    // ------------------------------------------------------------------------- //
    // ----------------------- Initialize filter ------------------------------- //
    // ------------------------------------------------------------------------- //
    float *F;
    err = cudaMallocHost((void**)&F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    cuda_check(err);
    F[0] = -1; F[1] = -1; F[2] = -1;
    F[3] = -1; F[4] = 8; F[5] = -1;
    F[6] = -1; F[7] = -1; F[8] = -1;
        
    // ---------------------------------------------------------- //
    // ------------------ Move filter to GPU -------------------- //
    // ---------------------------------------------------------- //
    std::cout << "Moving filter to GPU memory... \n";
    cudaEventRecord(beg);
    
    err = cudaMemcpyToSymbol(d_F, F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    cuda_check(err);

    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time_mem_t_f, beg, end);
    elapsed_time_mem_t_f /= 1000.;
    std::cout << "Time for filter data transfer (seconds): " << elapsed_time_mem_t_f << "\n";
    std::cout << "\n";

    // ---------------------------------------------------------- //
    // --------------------- 2D Convolution --------------------- //
    // ---------------------------------------------------------- //

    // Applying filters frame by frame
    std::cout << "Applying filter... \n"; 

    dim3 dim_block(16, 16, 1);
    dim3 dim_grid(ceil(new_size/(float)(16)), ceil(new_size/(float)(16)), 1);
    // Warmup
    for (int i = 0; i < 10; i++)
        gpu_conv2d_constMem_kernel<<<dim_grid, dim_block>>>(d_N, d_P, new_size, new_size);

    // Kernel execution
    cudaEventRecord(beg);

    for (int i = 0; i < 100; i++)
        gpu_conv2d_constMem_kernel<<<dim_grid, dim_block>>>(d_N, d_P, new_size, new_size);

    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time_kernel, beg, end);
    elapsed_time_kernel /= (1000. * 100);
    std::cout << "Time for kernel execution (seconds): " << elapsed_time_kernel << "\n";
    std::cout << "\n";

    // ---------------------------------------------------------- //
    // ---------- Copying result back to host memory -------------//
    // ---------------------------------------------------------- //
    std::cout << "Moving result to CPU memory... \n";
    cudaEventRecord(beg);
    
    err = cudaMemcpy(P, d_P, new_size*new_size*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);
    
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time_mem_t_out, beg, end);
    elapsed_time_mem_t_out /= 1000.;
    std::cout << "Time for output data transfer (seconds): " << elapsed_time_mem_t_out << "\n";
    std::cout << "\n";

    // ---------------------------------------------------------- //
    // --------------------- Benchmarking ------------------------//
    // ---------------------------------------------------------- //

    std::cout << "--------------------- \n";
    std::cout << "Benchmarking details: \n";
    std::cout << "--------------------- \n";
    std::cout << "Time (total): " << elapsed_time_kernel + elapsed_time_mem_alloc + 
                                        elapsed_time_mem_t_in + elapsed_time_mem_t_f + elapsed_time_mem_t_out << "\n";
    std::cout << "FPS (total): " << 1 / (elapsed_time_kernel + elapsed_time_mem_alloc + 
                                        elapsed_time_mem_t_in + elapsed_time_mem_t_f + elapsed_time_mem_t_out) << "\n";
    std::cout << "\n";
    
    std::cout << "Time (kernel): " << elapsed_time_kernel << "\n";
    std::cout << "FPS (kernel): " << 1 / (elapsed_time_kernel) << "\n";
    std::cout << "GFLOPS (kernel): " << 2*new_size*new_size*(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1) * 1e-9 / elapsed_time_kernel << "\n";
    std::cout << "------------------------------------ \n";
    std::cout << "\n";

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
    stbi_write_png("data/gpu_pinnedConst_benchmark_output_img.png", n_cols, n_rows, 1, ucharData.data(), n_cols);

    // ----------------------------------------------------------------- //
    // ------------------ Saving benchmark results --------------------- //
    // ----------------------------------------------------------------- //
    // Opening File
    const std::string filename = "benchmarking/gpu_pinnedConst.txt";
    std::ofstream file(filename);
    if (!file.is_open()) 
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }
    // Writing results
    file << "----------------------------------------------- \n";
    file << "GPU (Pinned with Constant) Benchmarking details \n";
    file << "----------------------------------------------- \n";
    file << "Time for GPU memory allocation (seconds): " << elapsed_time_mem_alloc << "\n";
    file << "\n";
    file << "Time for input data transfer (seconds): " << elapsed_time_mem_t_in << "\n";
    file << "\n";
    file << "Time for filter data transfer (seconds): " << elapsed_time_mem_t_f << "\n";
    file << "\n";
    file << "Time for kernel execution (seconds): " << elapsed_time_kernel << "\n";
    file << "\n";
    file << "Time for output data transfer (seconds): " << elapsed_time_mem_t_out << "\n";
    file << "\n";
    
    file << "Time (total): " << elapsed_time_kernel + elapsed_time_mem_alloc + 
                                        elapsed_time_mem_t_in + elapsed_time_mem_t_f + elapsed_time_mem_t_out << "\n";
    file << "FPS (total): " << 1 / (elapsed_time_kernel + elapsed_time_mem_alloc + 
                                        elapsed_time_mem_t_in + elapsed_time_mem_t_f + elapsed_time_mem_t_out) << "\n";
    file << "\n";
    
    file << "Time (kernel): " << elapsed_time_kernel << "\n";
    file << "FPS (kernel): " << 1 / (elapsed_time_kernel) << "\n";
    file << "GFLOPS (kernel): " << 2*new_size*new_size*(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1) * 1e-9 / elapsed_time_kernel << "\n";
    file << "-----------------------------------------------";

    cudaFreeHost(N);
    cudaFreeHost(F);
    cudaFreeHost(P);

    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);

    return 0;
}
