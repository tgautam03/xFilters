CC = nvcc -gencode arch=compute_86,code=sm_86

DEVICE_USAGE = --ptxas-options=-v
HOST_COMPILE_FLAG = -c
DEVICE_COMPILE_FLAG = -dc

CPU_OPTIMIZE = -O3 -Xcompiler "-Ofast -march=native -funroll-loops -ffast-math -msse2 -msse3 -msse4 -mavx -mavx2 -flto=4"


# Utils
build/utils.o: src/utils.cpp
	$(CC) $(HOST_COMPILE_FLAG) src/utils.cpp -o build/utils.o

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