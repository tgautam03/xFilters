CC = nvcc -gencode arch=compute_86,code=sm_86

DEVICE_USAGE = --ptxas-options=-v
HOST_COMPILE_FLAG = -c
DEVICE_COMPILE_FLAG = -dc

CPU_OPTIMIZE = -O3 -Xcompiler "-Ofast -march=native -funroll-loops -ffast-math -msse2 -msse3 -msse4 -mavx -mavx2 -flto=4"

# Utils
build/utils.o: src/utils.cpp
	@$(CC) $(HOST_COMPILE_FLAG) src/utils.cpp -o build/utils.o

# Benchmark Naive CPU
00_cpu_conv2d_benchmark.out: benchmarking/00_cpu_conv2d_benchmark.cpp src/00_cpu_conv2d.cpp build/utils.o
	@$(CC) $(CPU_OPTIMIZE) build/utils.o src/00_cpu_conv2d.cpp benchmarking/00_cpu_conv2d_benchmark.cpp -o bin/00_cpu_conv2d_benchmark.out

# Benchmark Naive GPU
01_gpu_conv2d_benchmark.out: benchmarking/01_gpu_conv2d_benchmark.cu src/01_gpu_conv2d.cu build/utils.o
	@$(CC) -w build/utils.o src/01_gpu_conv2d.cu benchmarking/01_gpu_conv2d_benchmark.cu -o bin/01_gpu_conv2d_benchmark.out

# Benchmark Constant Memory GPU
02_gpu_conv2d_constMem_benchmark.out: benchmarking/02_gpu_conv2d_constMem_benchmark.cu src/02_gpu_conv2d_constMem.cu build/utils.o
	@$(CC) -w -rdc=true build/utils.o src/02_gpu_conv2d_constMem.cu benchmarking/02_gpu_conv2d_constMem_benchmark.cu -o bin/02_gpu_conv2d_constMem_benchmark.out

# Benchmark Tiled GPU
03_gpu_conv2d_tiled_benchmark.out: benchmarking/03_gpu_conv2d_tiled_benchmark.cu src/03_gpu_conv2d_tiled.cu build/utils.o
	@$(CC) -w -rdc=true build/utils.o src/03_gpu_conv2d_tiled.cu benchmarking/03_gpu_conv2d_tiled_benchmark.cu -o bin/03_gpu_conv2d_tiled_benchmark.out

# Benchmark Pinned GPU
04_gpu_conv2d_pinnedMem_benchmark.out: benchmarking/04_gpu_conv2d_pinnedMem_benchmark.cu src/01_gpu_conv2d.cu build/utils.o
	@$(CC) -w build/utils.o src/01_gpu_conv2d.cu benchmarking/04_gpu_conv2d_pinnedMem_benchmark.cu -o bin/04_gpu_conv2d_pinnedMem_benchmark.out

# Benchmark Pinned & Constant GPU
05_gpu_conv2d_pinnedConstMem_benchmark.out: benchmarking/05_gpu_conv2d_pinnedConstMem_benchmark.cu src/02_gpu_conv2d_constMem.cu build/utils.o
	@$(CC) -w -rdc=true build/utils.o src/02_gpu_conv2d_constMem.cu benchmarking/05_gpu_conv2d_pinnedConstMem_benchmark.cu -o bin/05_gpu_conv2d_pinnedConstMem_benchmark.out

# Benchmark Pinned & Constant GPU
06_gpu_conv2d_pinnedTiled_benchmark.out: benchmarking/06_gpu_conv2d_pinnedTiled_benchmark.cu src/03_gpu_conv2d_tiled.cu build/utils.o
	@$(CC) -w -rdc=true build/utils.o src/03_gpu_conv2d_tiled.cu benchmarking/06_gpu_conv2d_pinnedTiled_benchmark.cu -o bin/06_gpu_conv2d_pinnedTiled_benchmark.out

##################################################################################################################################################################################
############################################################################# Applications #######################################################################################
##################################################################################################################################################################################
# Naive CPU
cpu_filters: scripts/00_cpu_filter.cpp src/00_cpu_conv2d.cpp build/utils.o
	@$(CC) -w $(CPU_OPTIMIZE) $(shell pkg-config --cflags --libs opencv4) build/utils.o src/00_cpu_conv2d.cpp scripts/00_cpu_filter.cpp -o bin/00_cpu_filter.out && ./bin/00_cpu_filter.out

# Naive GPU
gpu_filter: scripts/01_gpu_filter.cu src/02_gpu_conv2d_constMem.cu build/utils.o
	@$(CC) -w -rdc=true build/utils.o src/02_gpu_conv2d_constMem.cu scripts/01_gpu_filter.cu -o bin/01_gpu_filter.out && ./bin/01_gpu_filter.out

# Clean executable files
clean: 
	@echo "Removing object files..."
	rm bin/*.out build/*.o