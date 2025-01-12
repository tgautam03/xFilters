#include <iostream>
#include <chrono>
#include <exception>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "../include/00_cpu_conv2d.hpp"
#include "../include/utils.hpp"

#define FILTER_RADIUS 1

int main(int argc, char const *argv[])
{
    // ------------------------------------------------------------------------- //
    // Which filter; Options: Sharpen, High-pass, Low-pass, Gaussian, d_Gaussian //
    // ------------------------------------------------------------------------- //
    std::string img_loc, filter_type;
    if (argc > 1) 
    {
        img_loc = argv[1];
        std::cout << "Using image: " << img_loc << "\n";
        
        std::cout << "\n";

        filter_type = argv[2];
        std::cout << "Filter in use: " << filter_type << "\n";
    } 
    else 
    {
        std::cout << "Please provide the image location and a filter name." << "\n";
        std::terminate();
    }

    // ---------------------------------------------------------- //
    // ---------------- Defining filter matrix ------------------ //
    // ---------------------------------------------------------- //
    float *F = new float[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];
    if (filter_type == "Sharpen")
    {
        float alpha = 0.8f;
        std::cout << "Enter alpha between 0 and 1 (default: 0.8): ";
        std::cin >> alpha;
        std::cout << "\n";

        F[0] = -alpha/(9-9*alpha);
        F[1] = -alpha/(9-9*alpha);
        F[2] = -alpha/(9-9*alpha);
        F[3] = -alpha/(9-9*alpha);
        F[4] = (9-alpha)/(9-9*alpha);
        F[5] = -alpha/(9-9*alpha);
        F[6] = -alpha/(9-9*alpha);
        F[7] = -alpha/(9-9*alpha);
        F[8] = -alpha/(9-9*alpha);
        
    }
    else if (filter_type == "High-pass")
    {   
        F[0] = -1;
        F[1] = -1;
        F[2] = -1;
        F[3] = -1;
        F[4] = 8;
        F[5] = -1;
        F[6] = -1;
        F[7] = -1;
        F[8] = -1;
    }
    else if (filter_type == "Low-pass")
    {
        float alpha = 9.0f;
        std::cout << "Enter alpha (default: 9.0): ";
        std::cin >> alpha;
        std::cout << "\n";

        F[0] = 1/alpha;
        F[1] = 1/alpha;
        F[2] = 1/alpha;
        F[3] = 1/alpha;
        F[4] = 1/alpha;
        F[5] = 1/alpha;
        F[6] = 1/alpha;
        F[7] = 1/alpha;
        F[8] = 1/alpha;
    }
    else if (filter_type == "Gaussian")
    {
        float alpha = 16.0f;
        std::cout << "Enter alpha (default: 16.0): ";
        std::cin >> alpha;
        std::cout << "\n";

        F[0] = 1/alpha;
        F[1] = 2/alpha;
        F[2] = 1/alpha;
        F[3] = 2/alpha;
        F[4] = 3/alpha;
        F[5] = 4/alpha;
        F[6] = 1/alpha;
        F[7] = 2/alpha;
        F[8] = 1/alpha;
    }
    else if (filter_type == "d_Gaussian")
    {
        F[0] = -2;
        F[1] = 1;
        F[2] = -2;
        F[3] = 1;
        F[4] = 4;
        F[5] = 1;
        F[6] = -2;
        F[7] = 1;
        F[8] = -2;
    }
    else
    {
        std::cout << "Filter not supported!" << "\n";
        std::terminate();
    }

    // ---------------------------------------------------------- //
    // ------------------ Load image in memory ------------------ //
    // ---------------------------------------------------------- //
    // Getting frame info
    int n_cols, n_rows, channels;
    float *img = stbi_loadf(img_loc.c_str(), &n_cols, &n_rows, &channels, 0);

    std::cout << "\n";
    std::cout << "Loaded image with Width: " << n_cols << " and Height: " << n_rows << "\n";
    std::cout << "\n";

    // Determine size for square image (preprocessing variables)
    int new_size = std::max(n_rows, n_cols);

    // Allocate memory for the grayscale image
    float* N = new float[new_size*new_size];

    // Convert to grayscale using weighted average and store in memory
    to_grayscale(img, N, new_size, n_rows, n_cols, channels);

    // ---------------------------------------------------------- //
    // --------------------- 2D Convolution --------------------- //
    // ---------------------------------------------------------- //

    // Applying filters frame by frame
    std::cout << "Applying filter... \n";

    // Allocate memory for the output image
    float *P = new float[new_size * new_size]; 

    // Benchmarking
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    cpu_conv2d(N, F, P, FILTER_RADIUS, new_size, new_size);
    stop = std::chrono::high_resolution_clock::now();

    elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "--------------------- \n";
    std::cout << "Benchmarking details: \n";
    std::cout << "--------------------- \n";
    std::cout << "Time (seconds): " << (elapsed_time.count()/1e+6) << "\n";
    std::cout << "FPS: " << 1 / (elapsed_time.count()/1e+6) << "\n";
    std::cout << "GFLOPS: " << (2*n_rows*n_cols*(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1) * 1e-9) / (elapsed_time.count()/1e+6) << "\n";

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
    stbi_write_png("data/output_img.png", n_cols, n_rows, 1, ucharData.data(), n_cols);

    delete[] F;
    delete[] N;
    delete[] P;
    delete[] img;

    return 0;
}
