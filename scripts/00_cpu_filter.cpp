#include <iostream>
#include <chrono>
#include <exception>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include <opencv2/opencv.hpp>

#include "../include/00_cpu_conv2d.hpp"
#include "../include/utils.hpp"

#define FILTER_RADIUS 1

int main(int argc, char const *argv[])
{
    // Benchmarking variables
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

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
    float* N = new float[new_size * new_size];
    float *P = new float[new_size * new_size];

    // Convert to grayscale using weighted average and store in memory
    to_grayscale(img, N, new_size, n_rows, n_cols, channels);

    // ------------------------------------------------------------------------- //
    // ----------------------- Initialize filter ------------------------------- //
    // ------------------------------------------------------------------------- //
    std::string filter_type;
    float *F = new float[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

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
        // --------------------- 2D Convolution --------------------- //
        // ---------------------------------------------------------- //

        // Applying filters frame by frame
        std::cout << "Applying filter... \n"; 
    
        // Kernel execution
        start = std::chrono::high_resolution_clock::now();
        cpu_conv2d(N, F, P, FILTER_RADIUS, new_size, new_size);
        stop = std::chrono::high_resolution_clock::now();

        elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time for kernel execution (seconds): " << elapsed_time.count()/1e+6 << "\n";
        std::cout << "\n";

        // ---------------------------------------------------------- //
        // --------------------- Benchmarking ------------------------//
        // ---------------------------------------------------------- //

        std::cout << "--------------------- \n";
        std::cout << "Benchmarking details: \n";
        std::cout << "--------------------- \n";
        std::cout << "Time (kernel/total): " << elapsed_time.count()/1e+6 << "\n";
        std::cout << "FPS (kernel/total): " << 1 / (elapsed_time.count()/1e+6) << "\n";
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

    delete[] F;
    delete[] N;
    delete[] P;

    return 0;
}
