CC = nvcc

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

# Clean executable files
clean: 
	@echo "Removing object files..."
	rm *.out build/*.o