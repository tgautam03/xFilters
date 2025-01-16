#include <iostream>
#include <chrono>
#include <exception>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "../include/00_cpu_conv2d.hpp"
#include "../include/utils.hpp"

int main(int argc, char const *argv[])
{
    // Benchmarking variables
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

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
    float *N = new float[new_size * new_size];
    float *P = new float[new_size * new_size];

    // Convert to grayscale using weighted average and store in memory
    to_grayscale(img, N, new_size, n_rows, n_cols, channels);

    // Delete img from memory
    delete[] img;

    // ------------------------------------------------------------------------- //
    // --------------------- Filter: High-pass --------------------------------- //
    // ------------------------------------------------------------------------- //
    float *F = new float[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];
    F[0] = -1; F[1] = -1; F[2] = -1;
    F[3] = -1; F[4] = 8; F[5] = -1;
    F[6] = -1; F[7] = -1; F[8] = -1;
        
    // ---------------------------------------------------------- //
    // --------------------- 2D Convolution --------------------- //
    // ---------------------------------------------------------- //

    // Applying filters frame by frame
    std::cout << "Applying filter... \n"; 

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
        cpu_conv2d(N, F, P, FILTER_RADIUS, new_size, new_size);
    stop = std::chrono::high_resolution_clock::now();

    elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Time for kernel execution (seconds): " << (elapsed_time.count()/1e+6/100) << "\n";
    std::cout << "\n";

    std::cout << "--------------------- \n";
    std::cout << "Benchmarking details: \n";
    std::cout << "--------------------- \n";
    std::cout << "FPS (total): " << 1 / ((elapsed_time.count())/1e+6/100) << "\n";
    std::cout << "GFLOPS (kernel): " << (2*new_size*new_size*(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1) * 1e-9) / (elapsed_time.count()/1e+6/100) << "\n";
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
    stbi_write_png("data/img_output/cpu_benchmark_output_img.png", n_cols, n_rows, 1, ucharData.data(), n_cols);

    // ----------------------------------------------------------------- //
    // ------------------ Saving benchmark results --------------------- //
    // ----------------------------------------------------------------- //
    // Opening File
    const std::string filename = "benchmarking/cpu.txt";
    std::ofstream file(filename);
    if (!file.is_open()) 
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }
    // Writing results
    file << "------------------------ \n";
    file << "CPU Benchmarking details \n";
    file << "------------------------ \n";
    file << "Time for kernel execution (seconds): " << (elapsed_time.count()/1e+6/100) << "\n";
    file << "FPS (total): " << 1 / ((elapsed_time.count())/1e+6/100) << "\n";
    file << "GFLOPS (kernel): " << (2*new_size*new_size*(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1) * 1e-9) / (elapsed_time.count()/1e+6/100) << "\n";
    file << "------------------------";

    delete[] F;
    delete[] N;
    delete[] P;

    return 0;
}
