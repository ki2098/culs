#ifndef _UTIL_H_
#define _UTIL_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>

namespace Util {

__host__ __device__ static inline void idx2ijk(int idx, int &i, int &j, int &k, dim3 &size) {
    int size_y  = size.y;
    int size_z  = size.z;
    int size_yz = size_y * size_z;
    i   = idx / size_yz;
    idx = idx % size_yz;
    j   = idx / size_z;
    k   = idx % size_z;
}

__host__ __device__ static inline int ijk2idx(int i, int j, int k, dim3 &size) {
    return i * size.y * size.z + j * size.z + k;
}

__device__ static inline int get_global_idx() {
    return blockDim.x * blockIdx.x + threadIdx.x;
}

__device__ static inline int get_global_size() {
    return blockDim.x * gridDim.x;
}

}

#endif