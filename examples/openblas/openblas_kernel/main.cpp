
#include <iostream>
#include <openblas/cblas.h>
#include <vector>
#include <chrono>
#include <omp.h>

namespace chrono = std::chrono;

int main(int argc, char **argv)
{

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <vec_size> <nthreads>" << std::endl;
        return 1;
    }

    const int vec_size = std::atoi(argv[1]);
    const int nthreads = std::atoi(argv[2]);

    double* a = static_cast<double*>(aligned_alloc(64, vec_size * sizeof(double)));
    double* b = static_cast<double*>(aligned_alloc(64, vec_size * sizeof(double)));
    double* c = static_cast<double*>(aligned_alloc(64, vec_size * vec_size * sizeof(double)));

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < vec_size; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }


    openblas_set_num_threads(nthreads);
    omp_set_num_threads(nthreads);

    // Warmup
    cblas_dger(CblasRowMajor, vec_size, vec_size, 1.0, a, 1, b, 1, c, vec_size);

    constexpr int nrepetitions = 10;
    double total = 0;
    int nrepet = 0;
    while (true) {
        auto begin = chrono::high_resolution_clock::now();
        cblas_dger(CblasRowMajor, vec_size, vec_size, 1.0, a, 1, b, 1, c, vec_size);
        auto end = chrono::high_resolution_clock::now();
        double delta = chrono::duration<double>(end - begin).count();
        total += delta;
        nrepet++;
        
        // We want atleast 1s of total time to get a good average for small times
        // and atleast nrepet repetitions for large times
        if (total > 1.0 && nrepet >= nrepetitions) {
            break;
        }
    }

    std::cout << total / nrepet << std::endl;
    free(a);
    free(b);
    free(c);

    return 0;
}