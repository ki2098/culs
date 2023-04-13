#ifndef _DOM_H
#define _DOM_H 1

#include <cuda.h>
#include <cuda_runtime.h>

struct Dom {
    dim3        _size;
    dim3      _origin;
    dim3   *_size_ptr;
    dim3 *_origin_ptr;
    int          _num;

    Dom(dim3 &size, dim3 &origin);
    ~Dom();
};

Dom::Dom(dim3 &size, dim3 &origin) : _size(size), _origin(origin) {
    _num = size.x * size.y * size.z;
    cudaMalloc(  &_size_ptr, sizeof(dim3));
    cudaMalloc(&_origin_ptr, sizeof(dim3));
    cudaMemcpy(  _size_ptr,   &_size, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(_origin_ptr, &_origin, sizeof(dim3), cudaMemcpyHostToDevice);
}

Dom::~Dom() {
    cudaFree(_size_ptr);
    cudaFree(_origin_ptr);
}

#endif