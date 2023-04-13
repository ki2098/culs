#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../src/matrix.cuh"
#include "../src/dom.cuh"
#include "../src/poisson.cuh"

#define N 100000

__global__ void vector_add(MatrixCp<double> &a, MatrixCp<double> &b, MatrixCp<double> &c) {
    int index = Util::get_global_idx();
    int stride = Util::get_global_size();
    for (int i = index; i < a._num; i += stride) {
        c(i) = a(i) + b(i);
    }
}

int main() {
    dim3 size(N, 1, 1);
    dim3 origin(0, 0, 0);
    Dom dom(size, origin);

    Matrix<double> a(size, 1, LOCATION::HOST);
    Matrix<double> b(size, 1, LOCATION::HOST);

    for (int i = 0; i < N; i ++) {
        a(i) = 2;
        b(i) = i;
    }

    a.sync_h2d();
    b.sync_h2d();

    double product = dot(a, b, dom);

    a.release(LOCATION::BOTH);
    b.release(LOCATION::BOTH);

    printf("%lf\n", product);

    return 0;
}