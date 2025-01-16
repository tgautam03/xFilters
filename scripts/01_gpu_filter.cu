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
    std::string img_loc;
    std::cout << "Enter image location: ";
    std::cin >> img_loc;

    // Getting frame
    int n_cols, n_rows, channels;
    float *img = stbi_loadf(img_loc.c_str(), &n_cols, &n_rows, &channels, 0);

    std::cout << "Loaded image with Width: " << n_cols << " and Height: " << n_rows << "\n";
    std::cout << "\n";

    if (n_cols == 0 || n_rows == 0)
    {
        std::cout << "Image loading failed!" << "\n";
        std::terminate();
    }
    

    // Determine size for square image (preprocessing variables)
    int new_size = std::max(n_rows, n_cols);

    // Allocate memory for the grayscale image (input and output)
    cudaError_t err;
    float* N;
    err = cudaMallocHost((void**)&N, new_size*new_size*sizeof(float));
    cuda_check(err);

    float *P;
    err = cudaMallocHost((void**)&P, new_size*new_size*sizeof(float));
    cuda_check(err);

    // Convert to grayscale using weighted average and store in memory
    to_grayscale(img, N, new_size, n_rows, n_cols, channels);

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
    std::string filter_type;
    float *F;
    err = cudaMallocHost((void**)&F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    cuda_check(err);

    int iter = 0;
    while (true)
    {
        // ------------------------------------------------------------------------- //
        // Which filter; Options: Sharpen, High-pass, Low-pass, Gaussian, d_Gaussian //
        // ------------------------------------------------------------------------- //
        std::cout << "Filter options: Sharpen, High-pass, Low-pass, Gaussian, d_Gaussian \n";
        std::cout << "Enter filter (press 'q' to exit): ";
        std::cin >> filter_type;


        // ---------------------------------------------------------- //
        // ---------------- Defining filter matrix ------------------ //
        // ---------------------------------------------------------- //
        if (filter_type == "Sharpen")
        {
            float alpha = 0.8f;
            std::cout << "Enter alpha between 0 and 1 (default: 0.8): ";
            std::cin >> alpha;
            std::cout << "\n";

            F[0] = -alpha/(9-9*alpha);F[1] = -alpha/(9-9*alpha);F[2] = -alpha/(9-9*alpha);
            F[3] = -alpha/(9-9*alpha);F[4] = (9-alpha)/(9-9*alpha);F[5] = -alpha/(9-9*alpha);
            F[6] = -alpha/(9-9*alpha);F[7] = -alpha/(9-9*alpha);F[8] = -alpha/(9-9*alpha);
            
        }
        else if (filter_type == "High-pass")
        {
            std::cout << "\n";   
            F[0] = -1;F[1] = -1;F[2] = -1;
            F[3] = -1;F[4] = 8;F[5] = -1;
            F[6] = -1;F[7] = -1;F[8] = -1;
        }
        else if (filter_type == "Low-pass")
        {
            float alpha = 9.0f;
            std::cout << "Enter alpha (default: 9.0): ";
            std::cin >> alpha;
            std::cout << "\n";

            F[0] = 1/alpha;F[1] = 1/alpha;F[2] = 1/alpha;
            F[3] = 1/alpha;F[4] = 1/alpha;F[5] = 1/alpha;
            F[6] = 1/alpha;F[7] = 1/alpha;F[8] = 1/alpha;
        }
        else if (filter_type == "Gaussian")
        {
            float alpha = 16.0f;
            std::cout << "Enter alpha (default: 16.0): ";
            std::cin >> alpha;
            std::cout << "\n";

            F[0] = 1/alpha;F[1] = 2/alpha;F[2] = 1/alpha;
            F[3] = 2/alpha;F[4] = 3/alpha;F[5] = 4/alpha;
            F[6] = 1/alpha;F[7] = 2/alpha;F[8] = 1/alpha;
        }
        else if (filter_type == "d_Gaussian")
        {
            std::cout << "\n";
            F[0] = -2;F[1] = 1;F[2] = -2;
            F[3] = 1;F[4] = 4;F[5] = 1;
            F[6] = -2;F[7] = 1;F[8] = -2;
        }
        else if (filter_type == "q")
        {
            break;
        }
        else
        {
            std::cout << "Filter not supported!" << "\n";
            std::terminate();
        }

        
        // ---------------------------------------------------------- //
        // ------------------ Move filter to GPU -------------------- //
        // ---------------------------------------------------------- //
        std::cout << "Moving filter to GPU constant memory... \n";
        cudaEventRecord(beg);
        
        err = cudaMemcpyToSymbol(d_F, F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
        cuda_check(err);
        cudaDeviceSynchronize();

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

        // Kernel execution
        cudaEventRecord(beg);

        dim3 dim_block(16, 16, 1);
        dim3 dim_grid(ceil(new_size/(float)(16)), ceil(new_size/(float)(16)), 1);
        gpu_conv2d_constMem_kernel<<<dim_grid, dim_block>>>(d_N, d_P, new_size, new_size);
        cudaDeviceSynchronize();
        
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
        if (iter == 0)
        {
            std::cout << "Time (total): " << elapsed_time_kernel + elapsed_time_mem_alloc + 
                                                elapsed_time_mem_t_in + elapsed_time_mem_t_f + elapsed_time_mem_t_out << "\n";
            std::cout << "FPS (total): " << 1 / (elapsed_time_kernel + elapsed_time_mem_alloc + 
                                                elapsed_time_mem_t_in + elapsed_time_mem_t_f + elapsed_time_mem_t_out) << "\n";
            std::cout << "\n";
        }
        else
        {
            std::cout << "Time (total): " << elapsed_time_kernel +  elapsed_time_mem_t_f + elapsed_time_mem_t_out << "\n";
            std::cout << "FPS (total): " << 1 / (elapsed_time_kernel +  elapsed_time_mem_t_f+ elapsed_time_mem_t_out) << "\n";
            std::cout << "\n";
        }

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
        stbi_write_png("data/filtered_img.png", n_cols, n_rows, 1, ucharData.data(), n_cols);
        std::cout << "Image saved: 'data/filtered_img.png' \n";
        std::cout << "------------------------------------ \n";
        std::cout << "------------------------------------ \n";
        std::cout << "------------------------------------ \n";
        std::cout << "\n";

        iter += 1;
    }

    cudaFreeHost(N);
    cudaFreeHost(F);
    cudaFreeHost(P);

    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
