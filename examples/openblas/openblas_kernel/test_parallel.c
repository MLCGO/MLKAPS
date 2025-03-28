#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>  // OpenBLAS header

void check_openblas_parallel() {
    // Get the parallelization type
    int parallel_type = openblas_get_parallel();
    
    // Get number of threads
    int num_threads = openblas_get_num_threads();
    
    // Print the results
    printf("OpenBLAS Parallelization Check:\n");
    printf("Number of threads: %d\n", num_threads);
    
    switch(parallel_type) {
        case 0:
            printf("Parallelization: Sequential (no threading)\n");
            break;
        case 1:
            printf("Parallelization: OpenMP\n");
            break;
        case 2:
            printf("Parallelization: POSIX threads (pthreads)\n");
            break;
        default:
            printf("Parallelization: Unknown (%d)\n", parallel_type);
    }
}

int main() {
    // Example usage in your benchmark
    check_openblas_parallel();
    
    // Your outer product benchmark code would go here
    // For example:
    double a[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    double c[9];
    
    // C = alpha * A * B^T + beta * C (outer product)
    cblas_dger(CblasRowMajor, 3, 3, 1.0, a, 1, b, 1, c, 3);
    
    return 0;
}
