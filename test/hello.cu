#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <stdio.h>

__global__ void hello() {
    printf("Hello!\n");
}

int main() {
    
    cudaProfilerStart();
    hello<<<1, 1>>>();
    fflush(stdout);
    cudaProfilerStop();
    return 0;
}