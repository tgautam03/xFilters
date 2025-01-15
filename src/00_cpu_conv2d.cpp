#include "../include/00_cpu_conv2d.hpp"

void cpu_conv2d(float const *N, float const *F, float *P, int const r, int const n_rows, int const n_cols)
{
    // Loop over elements of output matrix P
    for (int out_col = 0; out_col < n_rows; out_col++)
    {
        for (int out_row = 0; out_row < n_cols; out_row++)
        {
            // Output in the register
            float p_val = 0.0f;

            // Loop over elements of the filter F
            for (int f_row = 0; f_row < 2*r+1; f_row++)
            {
                for (int f_col = 0; f_col < 2*r+1; f_col++)
                {
                    // Input-filter mapping
                    // int in_row = out_row + (f_row - r);
                    // int in_col = out_col + (f_col - r);

                    if (((out_row + (f_row - r)) >= 0 && (out_row + (f_row - r)) < n_rows) && ((out_col + (f_col - r)) >= 0 && (out_col + (f_col - r)) < n_cols))
                        p_val += F[f_row*(2*r+1) + f_col] * N[(out_row + (f_row - r))*n_cols + (out_col + (f_col - r))];
                        // p_val += F[f_row*(2*r+1) + f_col] * N[in_row*n_cols + in_col];
                }   
            }
            // Update value in the output matrix
            P[out_row*n_cols + out_col] = p_val;   
        }   
    }
}