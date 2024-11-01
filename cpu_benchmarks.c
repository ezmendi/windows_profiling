// File: cpu_benchmark.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cblas.h> // Include BLAS library

#define NUM_SIZES 5

int main() {
    int sizes[NUM_SIZES] = {256, 512, 1024, 2048, 4096};
    int num_threads_list[] = {1, 2, 4, 8, 16, 24};
    int num_num_threads = sizeof(num_threads_list)/sizeof(num_threads_list[0]);

    // Open the output file
    FILE *output_file = fopen("cpu_benchmark_results.csv", "w");
    if (output_file == NULL) {
        printf("Error opening output file!\n");
        return -1;
    }

    // Write CSV header
    fprintf(output_file, "NumThreads,MatrixSize,TimeSeconds\n");

    for(int t=0; t<num_num_threads; t++) {
        int num_threads = num_threads_list[t];
        // Set the number of threads for OpenMP and BLAS libraries
        omp_set_num_threads(num_threads);
        #ifdef OPENBLAS
            openblas_set_num_threads(num_threads);
        #endif
        #ifdef MKL
            mkl_set_num_threads(num_threads);
        #endif

        for(int s=0; s<NUM_SIZES; s++) {
            int N = sizes[s];

            // Allocate matrices
            float *A = (float*)malloc(N*N*sizeof(float));
            float *B = (float*)malloc(N*N*sizeof(float));
            float *C = (float*)malloc(N*N*sizeof(float));

            // Initialize matrices with random numbers
            srand(0); // For reproducibility
            for(int i=0; i<N*N; i++) {
                A[i] = (float)rand() / RAND_MAX;
                B[i] = (float)rand() / RAND_MAX;
                C[i] = 0.0f;
            }

            // Ensure data is not cached
            for(int i=0; i<N*N; i++) {
                volatile float tmp = A[i] + B[i];
            }

            double start_time = omp_get_wtime();

            // Default sgemm using BLAS library
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, 1.0f,
                        A, N,
                        B, N, 0.0f, C, N);

            double end_time = omp_get_wtime();
            double elapsed_time = end_time - start_time;

            // Write the result to the output file
            fprintf(output_file, "%d,%d,%f\n", num_threads, N, elapsed_time);

            free(A);
            free(B);
            free(C);
        }
    }

    fclose(output_file);
    return 0;
}
