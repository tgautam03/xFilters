CC = nvcc -gencode arch=compute_86,code=sm_86

DEVICE_USAGE = --ptxas-options=-v
HOST_COMPILE_FLAG = -c
DEVICE_COMPILE_FLAG = -dc

CPU_OPTIMIZE = -O3 -Xcompiler "-Ofast -march=native -funroll-loops -ffast-math -msse2 -msse3 -msse4 -mavx -mavx2 -flto=4"


# Utils
build/utils.o: src/utils.cpp
	$(CC) $(HOST_COMPILE_FLAG) src/utils.cpp -o build/utils.o

# Benchmark Naive CPU
00_cpu_conv2d_benchmark.out: benchmarking/00_cpu_conv2d_benchmark.cpp src/00_cpu_conv2d.cpp build/utils.o
	$(CC) $(CPU_OPTIMIZE) build/utils.o src/00_cpu_conv2d.cpp benchmarking/00_cpu_conv2d_benchmark.cpp -o bin/00_cpu_conv2d_benchmark.out

# Benchmark Naive GPU
01_gpu_conv2d_benchmark.out: benchmarking/01_gpu_conv2d_benchmark.cu src/01_gpu_conv2d.cu build/utils.o
	$(CC) -w build/utils.o src/01_gpu_conv2d.cu benchmarking/01_gpu_conv2d_benchmark.cu -o bin/01_gpu_conv2d_benchmark.out

# Benchmark Constant Memory GPU
02_gpu_conv2d_constMem_benchmark.out: benchmarking/02_gpu_conv2d_constMem_benchmark.cu src/02_gpu_conv2d_constMem.cu build/utils.o
	$(CC) -w -rdc=true build/utils.o src/02_gpu_conv2d_constMem.cu benchmarking/02_gpu_conv2d_constMem_benchmark.cu -o bin/02_gpu_conv2d_constMem_benchmark.out

# Benchmark Tiled GPU
03_gpu_conv2d_tiled_benchmark.out: benchmarking/03_gpu_conv2d_tiled_benchmark.cu src/03_gpu_conv2d_tiled.cu build/utils.o
	$(CC) -w -rdc=true build/utils.o src/03_gpu_conv2d_tiled.cu benchmarking/03_gpu_conv2d_tiled_benchmark.cu -o bin/03_gpu_conv2d_tiled_benchmark.out


# Naive CPU
00_cpu_conv2d_run.out: scripts/00_cpu_conv2d_run.cpp src/00_cpu_conv2d.cpp build/utils.o
	$(CC) $(CPU_OPTIMIZE) build/utils.o src/00_cpu_conv2d.cpp scripts/00_cpu_conv2d_run.cpp -o bin/00_cpu_conv2d_run.out

# Naive GPU
01_gpu_conv2d_run.out: scripts/01_gpu_conv2d_run.cu src/01_gpu_conv2d.cu build/utils.o
	$(CC) -w build/utils.o src/01_gpu_conv2d.cu scripts/01_gpu_conv2d_run.cu -o bin/01_gpu_conv2d_run.out

# Constant memory GPU
02_gpu_conv2d_constMem_run.out: scripts/02_gpu_conv2d_constMem_run.cu src/02_gpu_conv2d_constMem.cu build/utils.o
	$(CC) -w -rdc=true scripts/02_gpu_conv2d_constMem_run.cu src/02_gpu_conv2d_constMem.cu build/utils.o -o bin/02_gpu_conv2d_constMem_run.out

# Clean executable files
clean: 
	@echo "Removing object files..."
	rm bin/*.out build/*.o