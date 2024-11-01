REM File: benchmark.bat

@echo off
echo Compiling CPU benchmarking code...
REM For OpenBLAS
gcc cpu_benchmark.c -o cpu_benchmark -fopenmp -lopenblas
REM For Intel MKL (uncomment if using MKL)
REM gcc cpu_benchmark.c -o cpu_benchmark.exe -fopenmp -lmkl_rt

echo Compiling GPU benchmarking code...
nvcc gpu_benchmark.cu -o gpu_benchmark -lcublas

echo Starting CPU Benchmarking...
./cpu_benchmark

echo CPU Benchmarking completed. Results saved to cpu_benchmark_results.csv

echo Starting GPU Benchmarking...
./gpu_benchmark.exe

echo GPU Benchmarking completed. Results saved to gpu_benchmark_results.csv
