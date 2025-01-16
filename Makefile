CC = nvcc -gencode arch=compute_86,code=sm_86

DEVICE_USAGE = --ptxas-options=-v
HOST_COMPILE_FLAG = -c
DEVICE_COMPILE_FLAG = -dc

CPU_OPTIMIZE = -O3 -Xcompiler "-Ofast -march=native -funroll-loops -ffast-math -msse2 -msse3 -msse4 -mavx -mavx2 -flto=4"

##################################################################################################################################################################################
############################################################################# Source files #######################################################################################
##################################################################################################################################################################################
# Utils
utils.o: src/utils.cpp
	@$(CC) $(HOST_COMPILE_FLAG) $(CPU_OPTIMIZE) src/utils.cpp -o build/utils.o

# CPU Conv2D 
cpu_conv2d.o: src/cpu_conv2d.cpp
	@$(CC) $(HOST_COMPILE_FLAG) $(CPU_OPTIMIZE) src/cpu_conv2d.cpp -o build/cpu_conv2d.o

# GPU Conv2D 
gpu_conv2d.o: src/gpu_conv2d.cu
	@$(CC) $(DEVICE_COMPILE_FLAG) src/gpu_conv2d.cu -o build/gpu_conv2d.o

# GPU Conv2D with constant memory 
gpu_conv2d_constMem.o: src/gpu_conv2d_constMem.cu
	@$(CC) $(DEVICE_COMPILE_FLAG) src/gpu_conv2d_constMem.cu -o build/gpu_conv2d_constMem.o

# GPU Conv2D with constant memory and tiling
gpu_conv2d_tiled.o: src/gpu_conv2d_tiled.cu
	@$(CC) $(DEVICE_COMPILE_FLAG) src/gpu_conv2d_tiled.cu -o build/gpu_conv2d_tiled.o

##################################################################################################################################################################################
############################################################################# Benchmarking #######################################################################################
##################################################################################################################################################################################

# Benchmark Naive CPU
00_cpu_conv2d_benchmark.out: benchmarking/00_cpu_conv2d_benchmark.cpp cpu_conv2d.o utils.o
	@$(CC) $(CPU_OPTIMIZE) build/utils.o build/cpu_conv2d.o benchmarking/00_cpu_conv2d_benchmark.cpp -o bin/00_cpu_conv2d_benchmark.out && ./bin/00_cpu_conv2d_benchmark.out

# Benchmark Naive GPU
01_gpu_conv2d_benchmark.out: benchmarking/01_gpu_conv2d_benchmark.cu gpu_conv2d.o utils.o
	@$(CC) -w build/utils.o build/gpu_conv2d.o benchmarking/01_gpu_conv2d_benchmark.cu -o bin/01_gpu_conv2d_benchmark.out && ./bin/01_gpu_conv2d_benchmark.out

# Benchmark Constant Memory GPU
02_gpu_conv2d_constMem_benchmark.out: benchmarking/02_gpu_conv2d_constMem_benchmark.cu gpu_conv2d_constMem.o utils.o
	@$(CC) -w -rdc=true build/utils.o build/gpu_conv2d_constMem.o benchmarking/02_gpu_conv2d_constMem_benchmark.cu -o bin/02_gpu_conv2d_constMem_benchmark.out && ./bin/02_gpu_conv2d_constMem_benchmark.out

# Benchmark Tiled GPU
03_gpu_conv2d_tiled_benchmark.out: benchmarking/03_gpu_conv2d_tiled_benchmark.cu gpu_conv2d_tiled.o utils.o
	@$(CC) -w -rdc=true build/utils.o build/gpu_conv2d_tiled.o benchmarking/03_gpu_conv2d_tiled_benchmark.cu -o bin/03_gpu_conv2d_tiled_benchmark.out && ./bin/03_gpu_conv2d_tiled_benchmark.out

# Benchmark Naive GPU (Pinned Memory)
04_gpu_conv2d_pinnedMem_benchmark.out: benchmarking/04_gpu_conv2d_pinnedMem_benchmark.cu gpu_conv2d.o utils.o
	@$(CC) -w build/utils.o build/gpu_conv2d.o benchmarking/04_gpu_conv2d_pinnedMem_benchmark.cu -o bin/04_gpu_conv2d_pinnedMem_benchmark.out && ./bin/04_gpu_conv2d_pinnedMem_benchmark.out

# Benchmark Constant Memory GPU (Pinned Memory)
05_gpu_conv2d_pinnedConstMem_benchmark.out: benchmarking/05_gpu_conv2d_pinnedConstMem_benchmark.cu gpu_conv2d_constMem.o utils.o
	@$(CC) -w -rdc=true build/utils.o build/gpu_conv2d_constMem.o benchmarking/05_gpu_conv2d_pinnedConstMem_benchmark.cu -o bin/05_gpu_conv2d_pinnedConstMem_benchmark.out && ./bin/05_gpu_conv2d_pinnedConstMem_benchmark.out

# Benchmark Tiled GPU (Pinned Memory)
06_gpu_conv2d_pinnedTiled_benchmark.out: benchmarking/06_gpu_conv2d_pinnedTiled_benchmark.cu gpu_conv2d_tiled.o utils.o
	@$(CC) -w -rdc=true build/utils.o build/gpu_conv2d_tiled.o benchmarking/06_gpu_conv2d_pinnedTiled_benchmark.cu -o bin/06_gpu_conv2d_pinnedTiled_benchmark.out && ./bin/06_gpu_conv2d_pinnedTiled_benchmark.out

##################################################################################################################################################################################
############################################################################# Applications #######################################################################################
##################################################################################################################################################################################
# Naive CPU
filters_cpu: scripts/00_cpu_filter.cpp cpu_conv2d.o utils.o
	@$(CC) -w $(CPU_OPTIMIZE) $(shell pkg-config --cflags --libs opencv4) build/utils.o build/cpu_conv2d.o scripts/00_cpu_filter.cpp -o bin/00_cpu_filter.out && ./bin/00_cpu_filter.out

# Naive GPU
filters_gpu: scripts/01_gpu_filter.cu gpu_conv2d_constMem.o utils.o
	@$(CC) -w -rdc=true build/utils.o build/gpu_conv2d_constMem.o scripts/01_gpu_filter.cu -o bin/01_gpu_filter.out && ./bin/01_gpu_filter.out

##################################################################################################################################################################################
########################################################################## Clean executables #####################################################################################
##################################################################################################################################################################################
clean: 
	@echo "Removing object files..."
	rm bin/*.out build/*.o