#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>  // OpenBLAS header

void check_openblas_parallel(const char* stage) {
    int th_model = openblas_get_parallel();
    int num_threads = openblas_get_num_threads();

    printf("OpenBLAS Parallelization Check (%s):\n", stage);
    printf("Number of threads: %d\n", num_threads);
    switch(th_model) {
        case OPENBLAS_SEQUENTIAL:
            printf("OpenBLAS is compiled sequentially.\n");
            break;
        case OPENBLAS_THREAD:
            printf("OpenBLAS is compiled using the normal threading model (pthreads).\n");
            break;
        case OPENBLAS_OPENMP:
            printf("OpenBLAS is compiled using OpenMP.\n");
            break;
        default:
            printf("OpenBLAS threading model unknown (%d).\n", th_model);
    }
    printf("\n");
}

int main() {
    // Check initial state
    check_openblas_parallel("Before setting threads");

    // Set number of threads to 4
    openblas_set_num_threads(4);

    // Check after setting
    check_openblas_parallel("After setting to 4 threads");

    // Simple BLAS test
    double a[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    double c[9] = {0};

    cblas_dger(CblasRowMajor, 3, 3, 1.0, a, 1, b, 1, c, 3);
    printf("BLAS operation completed (c[0] = %f)\n", c[0]);

    return 0;
}
