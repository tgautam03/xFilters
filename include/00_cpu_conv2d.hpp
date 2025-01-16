#ifndef CPU_CONV2D
#define CPU_CONV2D

#define FILTER_RADIUS 1

void cpu_conv2d(float *N, float *F, float *P, int r, int n_rows, int n_cols);

#endif