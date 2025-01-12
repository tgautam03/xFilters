#ifndef UTILS
#define UTILS

#include <iostream>
#include <assert.h>
#include <random>
#include <iomanip>
#include <fstream>
#include <string>


// Convert RGB to grayscale
void to_grayscale(float* img, float* gray_img, int new_size, int n_rows, int n_cols, int channels);


// Asserting matrices are same within the tolerance (eps)
void assert_arr(float *A_mat, float *B_mat, int size, float eps);

// Update benchmark.txt file with recorded times and GFLOPS
void update_benckmark_txt(const std::string& filename, const double recorded_times[], 
                        const double recorded_gflops[], const int mat_sizes[], 
                        const int n_sizes);

#endif