// File: gpu_benchmark.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define NUM_SIZES 5

int main() {
    int sizes[NUM_SIZES] = {256, 512, 1024, 2048, 4096};

    // Open the output file
    FILE *output_file = fopen("gpu_benchmark_results.csv", "w");
    if (output_file == NULL) {
        printf("Error opening output file!\n");
        return -1;
    }

    // Write CSV header
    fprintf(output_file, "NumGPUs,MatrixSize,TimeSeconds\n");

    // Check number of devices
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("Number of CUDA devices: %d\n", device_count);

    for(int s=0; s<NUM_SIZES; s++) {
        int N = sizes[s];

        // Allocate host matrices
        float *h_A = (float*)malloc(N*N*sizeof(float));
        float *h_B = (float*)malloc(N*N*sizeof(float));
        float *h_C = (float*)malloc(N*N*sizeof(float));

        // Initialize matrices with random numbers
        srand(0); // For reproducibility
        for(int i=0; i<N*N; i++) {
            h_A[i] = (float)rand() / RAND_MAX;
            h_B[i] = (float)rand() / RAND_MAX;
            h_C[i] = 0.0f;
        }

        // Ensure data is not cached
        for(int i=0; i<N*N; i++) {
            volatile float tmp = h_A[i] + h_B[i];
        }

        // Single GPU computation
        {
            cudaSetDevice(0);

            // Allocate device matrices
            float *d_A, *d_B, *d_C;
            cudaMalloc((void**)&d_A, N*N*sizeof(float));
            cudaMalloc((void**)&d_B, N*N*sizeof(float));
            cudaMalloc((void**)&d_C, N*N*sizeof(float));

            // Copy matrices to device
            cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

            // Ensure data is not cached
            cudaDeviceSynchronize();

            // Use cuBLAS sgemm
            cublasHandle_t handle;
            cublasCreate(&handle);
            float alpha = 1.0f;
            float beta = 0.0f;

            cudaDeviceSynchronize();
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N, &alpha,
                        d_B, N,
                        d_A, N,
                        &beta,
                        d_C, N);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);

            // Write the result to the output file
            fprintf(output_file, "%d,%d,%f\n", 1, N, elapsedTime / 1000.0f);

            // Cleanup
            cublasDestroy(handle);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Multi-GPU computation (if device_count >= 2)
        if(device_count >= 2) {
            int num_devices = 2;
            int N_per_device = N / num_devices;

            // Allocate device matrices on each GPU
            float *d_A[2], *d_B[2], *d_C[2];
            cublasHandle_t handles[2];

            for(int d=0; d<num_devices; d++) {
                cudaSetDevice(d);
                int start_row = d * N_per_device;
                int rows = (d == num_devices - 1) ? N - start_row : N_per_device;

                cudaMalloc((void**)&d_A[d], N*rows*sizeof(float));
                cudaMalloc((void**)&d_B[d], N*N*sizeof(float));
                cudaMalloc((void**)&d_C[d], N*rows*sizeof(float));

                // Copy relevant parts of matrices to each device
                cudaMemcpy(d_A[d], h_A + start_row*N, N*rows*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_B[d], h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

                cublasCreate(&handles[d]);
            }

            // Ensure data is not cached
            cudaDeviceSynchronize();

            // Start timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);

            // Launch computations on each device
            for(int d=0; d<num_devices; d++) {
                cudaSetDevice(d);

                int start_row = d * N_per_device;
                int rows = (d == num_devices - 1) ? N - start_row : N_per_device;

                float alpha = 1.0f;
                float beta = 0.0f;

                cublasSetStream(handles[d], 0);

                cublasSgemm(handles[d], CUBLAS_OP_N, CUBLAS_OP_N,
                            N, rows, N, &alpha,
                            d_B[d], N,
                            d_A[d], N,
                            &beta,
                            d_C[d], N);
            }

            // Synchronize devices
            for(int d=0; d<num_devices; d++) {
                cudaSetDevice(d);
                cudaDeviceSynchronize();
            }

            // End timing
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);

            // Write the result to the output file
            fprintf(output_file, "%d,%d,%f\n", num_devices, N, elapsedTime / 1000.0f);

            // Cleanup
            for(int d=0; d<num_devices; d++) {
                cublasDestroy(handles[d]);
                cudaFree(d_A[d]);
                cudaFree(d_B[d]);
                cudaFree(d_C[d]);
            }
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        free(h_A);
        free(h_B);
        free(h_C);
    }

    fclose(output_file);

    return 0;
}
