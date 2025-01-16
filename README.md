# xFilters
**Convolution** is a popular array operation used in signal processing, digital recording, image/video processing, and computer vision. This repository provides **2D convolution algorithm** written from scratch in **C++ (for CPU)** and **CUDA C++ (for GPU)**, which can be used to apply **filters** to **high resolution** images. 

> Images are first converted to grayscale and then filter is applied.

**Table of content**

0. Naive 2D convolution on a CPU.
1. Naive 2D convolution on a GPU.
2. 2D convolution on a GPU using constant memory for filter matrix.
3. 2D convolution on a GPU using constant memory for filter matrix and tiling for shared memory usage.
4. Naive 2D convolution on a GPU (using pinned memory).
5. 2D convolution on a GPU using constant memory for filter matrix (using pinned memory).
6. 2D convolution on a GPU using constant memory for filter matrix and tiling for shared memory usage (using pinned memory).

## Example Run
**CPU/GPU Filter**
1. In the terminal run: `make filters_cpu` or `make filters_gpu`
2. You will be asked to enter the location of the image. For example, `data/8k.jpg`.
3. You will be asked to type the filter name. Supported filters are as follows:

    ### Supported Filters
    #### Sharpen
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/8k.jpg" width="200" height="150">
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/Sharpen_filtered_img.png" width="200" height="150">

    #### High-pass (edge detection)
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/8k.jpg" width="200" height="150">
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/High-pass_filtered_img.png" width="200" height="150">

    #### Low-pass 
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/8k.jpg" width="200" height="150">
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/Low-pass_filtered_img.png" width="200" height="150">

    #### Gaussian (image blurring)
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/8k.jpg" width="200" height="150">
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/Gaussian_filtered_img.png" width="200" height="150">

    #### Derivative of Gaussian (edge detection)
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/8k.jpg" width="200" height="150">
    <img src="https://raw.githubusercontent.com/tgautam03/xFilters/refs/heads/master/data/d_Gaussian_filtered_img.png" width="200" height="150">

## Running Benchmarks
### Naive CPU
```bash
make 00_cpu_conv2d_benchmark.out 
```
```
Loaded image with Width: 2048 and Height: 1328

Applying filter... 
Time for kernel execution (seconds): 0.0607285

--------------------- 
Benchmarking details: 
--------------------- 
FPS (total): 16.4667
GFLOPS (kernel): 1.2432
------------------------------------ 
```

### Naive GPU
```bash
make 01_gpu_conv2d_benchmark.out
```
```
Loaded image with Width: 2048 and Height: 1328

Allocating GPU memory... 
Time for GPU memory allocation (seconds): 0.00044032

Moving input to GPU memory... 
Time for input data transfer (seconds): 0.0028009

Moving filter to GPU memory... 
Time for filter data transfer (seconds): 8.736e-06

Applying filter... 
Time for kernel execution (seconds): 5.20294e-05

Moving result to CPU memory... 
Time for output data transfer (seconds): 0.00601299

--------------------- 
Benchmarking details: 
--------------------- 
Time (total): 0.00931497
FPS (total): 107.354

Time (kernel): 5.20294e-05
FPS (kernel): 19219.9
GFLOPS (kernel): 1451.05
------------------------------------ 
```

### GPU using constant memory
```bash
make 02_gpu_conv2d_constMem_benchmark.out
```
```
Loaded image with Width: 2048 and Height: 1328

Allocating GPU memory... 
Time for GPU memory allocation (seconds): 0.000191488

Moving input to GPU memory... 
Time for input data transfer (seconds): 0.00271984

Moving filter to GPU memory... 
Time for filter data transfer (seconds): 0.000128704

Applying filter... 
Time for kernel execution (seconds): 5.16403e-05

Moving result to CPU memory... 
Time for output data transfer (seconds): 0.00601722

--------------------- 
Benchmarking details: 
--------------------- 
Time (total): 0.00910889
FPS (total): 109.783

Time (kernel): 5.16403e-05
FPS (kernel): 19364.7
GFLOPS (kernel): 1461.99
------------------------------------ 
```

### GPU using constant memory and tiling
```bash
make 03_gpu_conv2d_tiled_benchmark.out 
```
```
Loaded image with Width: 2048 and Height: 1328

Allocating GPU memory... 
Time for GPU memory allocation (seconds): 0.000313344

Moving input to GPU memory... 
Time for input data transfer (seconds): 0.00283443

Moving filter to GPU memory... 
Time for filter data transfer (seconds): 0.0002504

Applying filter... 
Time for kernel execution (seconds): 5.53062e-05

Moving result to CPU memory... 
Time for output data transfer (seconds): 0.0065999

--------------------- 
Benchmarking details: 
--------------------- 
Time (total): 0.0100534
FPS (total): 99.469

Time (kernel): 5.53062e-05
FPS (kernel): 18081.1
GFLOPS (kernel): 1365.08
------------------------------------ 
```

### Naive GPU (pinned memory)
```bash
make 04_gpu_conv2d_pinnedMem_benchmark.out
```
```
Loaded image with Width: 2048 and Height: 1328

Allocating GPU memory... 
Time for GPU memory allocation (seconds): 0.000217088

Moving input to GPU memory... 
Time for input data transfer (seconds): 0.00265677

Moving filter to GPU memory... 
Time for filter data transfer (seconds): 9.632e-06

Applying filter... 
Time for kernel execution (seconds): 4.50765e-05

Moving result to CPU memory... 
Time for output data transfer (seconds): 0.00249299

--------------------- 
Benchmarking details: 
--------------------- 
Time (total): 0.00542156
FPS (total): 184.449

Time (kernel): 4.50765e-05
FPS (kernel): 22184.5
GFLOPS (kernel): 1674.88
------------------------------------ 
```

### GPU using constant memory (pinned memory)
```bash
make 05_gpu_conv2d_pinnedConstMem_benchmark.out 
```
```
Loaded image with Width: 2048 and Height: 1328

Allocating GPU memory... 
Time for GPU memory allocation (seconds): 0.000176064

Moving input to GPU memory... 
Time for input data transfer (seconds): 0.00267555

Moving filter to GPU memory... 
Time for filter data transfer (seconds): 0.000199776

Applying filter... 
Time for kernel execution (seconds): 4.3735e-05

Moving result to CPU memory... 
Time for output data transfer (seconds): 0.00250381

--------------------- 
Benchmarking details: 
--------------------- 
Time (total): 0.00559894
FPS (total): 178.605

Time (kernel): 4.3735e-05
FPS (kernel): 22865
GFLOPS (kernel): 1726.25
------------------------------------ 
```

### GPU using constant memory and tiling (pinned memory)
```bash
make 06_gpu_conv2d_pinnedTiled_benchmark.out
```
```
Loaded image with Width: 2048 and Height: 1328

Allocating GPU memory... 
Time for GPU memory allocation (seconds): 0.000154464

Moving input to GPU memory... 
Time for input data transfer (seconds): 0.0026567

Moving filter to GPU memory... 
Time for filter data transfer (seconds): 0.000105152

Applying filter... 
Time for kernel execution (seconds): 5.37395e-05

Moving result to CPU memory... 
Time for output data transfer (seconds): 0.0024945

--------------------- 
Benchmarking details: 
--------------------- 
Time (total): 0.00546456
FPS (total): 182.997

Time (kernel): 5.37395e-05
FPS (kernel): 18608.3
GFLOPS (kernel): 1404.88
------------------------------------ 
```

## References
- Image load/save done using [stb single-file public domain libraries for C/C++](https://github.com/nothings/stb). Check out [lib](https://github.com/tgautam03/xFilters/tree/master/lib) for the specific source code.

- Example images in [data](https://github.com/tgautam03/xFilters/tree/master/data):
    - [Image by Eberhard Grossgasteiger](https://www.pexels.com/photo/mountain-at-night-under-a-starry-sky-1624496/)
    - [Image by Pok Rie](https://www.pexels.com/photo/seawaves-on-sands-982263/)