#include "../include/utils.hpp"

void to_grayscale(float* img, float* gray_img, int new_size, int n_rows, int n_cols, int channels)
{
    // Convert to grayscale using weighted average
    for (int i = 0; i < n_rows; ++i) 
    {
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

void assert_arr(float *A_mat, float *B_mat, int size, float eps)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(A_mat[i] - B_mat[i]) > eps)
        {
            std::cerr << "Assertion failed for " << "index: " << i << ".\n"
                    << "Absolute Difference: " << fabs(A_mat[i] - B_mat[i]) << "\n";
            assert(fabs(A_mat[i] - B_mat[i]) < eps && "Assertion failed!");
        }
    }
}

void update_benckmark_txt(const std::string& filename, const double recorded_times[], 
                        const double recorded_gflops[], const int mat_sizes[], 
                        const int n_sizes)
{
    // Opening File
    std::ofstream file(filename);
    if (!file.is_open()) 
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "Matrix Sizes" << ": ";
    for (int i = 0; i < n_sizes; i++)
    {
        if (i != n_sizes-1)
            file << mat_sizes[i] << " " ;
        else
            file << mat_sizes[i] << "\n \n" ;
    }

    file << "Time (Seconds)" << ": ";
    for (int i = 0; i < n_sizes; i++)
    {
        if (i != n_sizes-1)
            file << recorded_times[i] << " " ;
        else
            file << recorded_times[i] << "\n \n" ;
    }

    file << "GPLOPS" << ": ";
    for (int i = 0; i < n_sizes; i++)
    {
        if (i != n_sizes-1)
            file << recorded_gflops[i] << " " ;
        else
            file << recorded_gflops[i] << "\n \n" ;
    }
    
}