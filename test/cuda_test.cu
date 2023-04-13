#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 100

template<class T>
struct Vector{
    int dim;
    T *data;
};

__device__ unsigned get_global_idx() {
    unsigned block_size = blockDim.x * blockDim.y * blockDim.z;
    unsigned block_idx  = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    unsigned thread_idx = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
    return block_idx * block_size + thread_idx;
}

__device__ unsigned get_global_size() {
    unsigned block_size = blockDim.x * blockDim.y * blockDim.z;
    unsigned grid_size  = gridDim.x  * gridDim.y  * gridDim.z;
    return block_size * grid_size;
}


__global__ void vector_add(Vector<double> &a, Vector<double> &b, Vector<double> &c) {
    int index = get_global_idx();
    int stride = get_global_size();
    for (int i = index; i < a.dim; i += stride) {
        c.data[i] = a.data[i] + b.data[i];
    }

    if (index == 0) {
        printf("%u %u %u %u %u %u %d %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, index, stride);
    }
}

int main() {
    Vector<double> a, b, c;
    a.dim = b.dim = c.dim = N;
    a.data = (double*)malloc(sizeof(double) * N);
    b.data = (double*)malloc(sizeof(double) * N);
    c.data = (double*)malloc(sizeof(double) * N);

    for (int i = 0; i < N; i ++) {
        a.data[i] = i;
        b.data[i] = 2 * i;
    }

    Vector<double> d_a, d_b, d_c;
    d_a.dim = d_b.dim = d_c.dim = N;

    cudaMalloc(&(d_a.data), sizeof(double) * N);
    cudaMalloc(&(d_b.data), sizeof(double) * N);
    cudaMalloc(&(d_c.data), sizeof(double) * N);

    cudaMemcpy(d_a.data, a.data, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.data, b.data, sizeof(double) * N, cudaMemcpyHostToDevice);

    Vector<double> *dd_a, *dd_b, *dd_c;
    cudaMalloc(&dd_a, sizeof(Vector<double>));
    cudaMalloc(&dd_b, sizeof(Vector<double>));
    cudaMalloc(&dd_c, sizeof(Vector<double>));
    cudaMemcpy(dd_a, &d_a, sizeof(Vector<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_b, &d_b, sizeof(Vector<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_c, &d_c, sizeof(Vector<double>), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    dim3 grid(4, 1, 1);
    dim3 block(256, 1, 1);

    vector_add<<<grid,block>>>(*dd_a, *dd_b, *dd_c);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaMemcpy(c.data, d_c.data, sizeof(double) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_a.data);
    cudaFree(d_b.data);
    cudaFree(d_c.data);
    cudaFree(dd_a);
    cudaFree(dd_b);
    cudaFree(dd_c);

    for (int i = 0; i < N; i ++) {
        if (fabs(a.data[i] + b.data[i] - c.data[i]) > 1e-5) {
            printf("%d %lf %lf %lf\n", i, a.data[i], b.data[i], c.data[i]);
        }
    }

    free(a.data);
    free(b.data);
    free(c.data);

    float millisec;
    cudaEventElapsedTime(&millisec, start, end);
    printf("%f\n", millisec);

    return 0;
}