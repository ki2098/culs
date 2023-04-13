#ifndef _MATRIX_H
#define _MATRIX_H 1

#include <float.h>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dom.cuh"
#include "util.cuh"
#include "param.cuh"

namespace LOCATION {
static const int NONE   = 0;
static const int HOST   = 1;
static const int DEVICE = 2;
static const int BOTH   = HOST | DEVICE;
}

template<class T>
struct MatrixCp {
    T  *_arr;
    int _row;
    int _col;
    int _num;
    int _loc;
    MatrixCp(dim3 &size, int col, int loc);
    void release();
    __host__ __device__ T& operator()(int idx) {return _arr[idx];}
    __host__ __device__ T& operator()(int row_idx, int col_idx) {return _arr[col_idx * _row + row_idx];}
};

template<class T>
MatrixCp<T>::MatrixCp(dim3 &size, int col, int loc) : _row(size.x * size.y * size.z), _col(col), _num(size.x * size.y * size.z * col), _loc(loc) {
    if (loc == LOCATION::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
    } else if (loc == LOCATION::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
    }
}

template<class T>
void MatrixCp<T>::release() {
    if (_loc == LOCATION::HOST) {
        free(_arr);
    } else if (_loc == LOCATION::DEVICE) {
        cudaFree(_arr);
    }
    _loc = LOCATION::NONE;
}

template<class T>
struct Matrix {
    MatrixCp<T>  _hh;
    MatrixCp<T>  _hd;
    MatrixCp<T> *_dd;
    int         _row;
    int         _col;
    int         _num;
    int         _loc;
    Matrix(dim3 &size, int col, int loc);
    void release(int loc);
    void sync_h2d();
    void sync_d2h();
    __host__ __device__ T& operator()(int idx) {return _hh(idx);}
    __host__ __device__ T& operator()(int row_idx, int col_idx) {return _hh(row_idx, col_idx);}
};

template<class T>
Matrix<T>::Matrix(dim3 &size, int col, int loc) : _row(size.x * size.y * size.z), _col(col), _num(size.x * size.y * size.z * col), _loc(loc), _hh(size, col, (loc & LOCATION::HOST)), _hd(size, col, (loc & LOCATION::DEVICE)), _dd(nullptr) {
    if (loc & LOCATION::DEVICE) {
        cudaMalloc(&_dd, sizeof(MatrixCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(MatrixCp<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
void Matrix<T>::release(int loc) {
    if ((loc & LOCATION::HOST) && (_loc & LOCATION::HOST)) {
        _hh.release();
        _loc &= (~LOCATION::HOST);
    }
    if ((loc & LOCATION::DEVICE) && (_loc & LOCATION::DEVICE)) {
        _hd.release();
        cudaFree(_dd);
        _loc &= (~LOCATION::DEVICE);
    }

}

template<class T>
void Matrix<T>::sync_h2d() {
    if (_loc == LOCATION::BOTH) {
        cudaMemcpy(_hd._arr, _hh._arr, sizeof(T) * _hh._num, cudaMemcpyHostToDevice);
    } else if (_loc == LOCATION::HOST) {
        cudaMalloc(&(_hd._arr), sizeof(T) * _num);
        cudaMemcpy(_hd._arr, _hh._arr, sizeof(T) * _num, cudaMemcpyHostToDevice);
        _hd._loc |= LOCATION::DEVICE;
        _loc     |= LOCATION::DEVICE;
        cudaMalloc(&_dd, sizeof(MatrixCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(MatrixCp<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
void Matrix<T>::sync_d2h() {
    if (_loc == LOCATION::BOTH) {
        cudaMemcpy(_hh._arr, _hd._arr, sizeof(T) * _num, cudaMemcpyDeviceToHost);
    } else if (_loc == LOCATION::DEVICE) {
        _hh._arr = (T*)malloc(sizeof(T) * _num);
        cudaMemcpy(_hh._arr, _hd._arr, sizeof(T) * _num, cudaMemcpyDeviceToHost);
        _hh._loc |= LOCATION::HOST;
        _loc     |= LOCATION::HOST;
    }
}

namespace MatrixUtil {

__global__ static void get_max_diag_kernel(MatrixCp<double> &a, double *max_partial, dim3 &size) {
    __shared__ double cache[n_threads];
    int stride = Util::get_global_size();
    double temp_max = - DBL_MAX;
    for (int idx = Util::get_global_idx(); idx < a._num; idx += stride) {
        double _value = a(idx, 0);
        if (_value > temp_max) {
            temp_max = _value;
        }
    }
    cache[threadIdx.x] = temp_max;

    int length = n_threads;
    while (length > 1) {
        int cut = length / 2;
        int reduce = length - cut;
        if (threadIdx.x < cut) {
            double _value = cache[threadIdx.x + reduce];
            if (_value > cache[threadIdx.x]) {
                cache[threadIdx.x] = _value;
            }
        }
        __syncthreads();
        length = reduce;
    }

    if (threadIdx.x == 0) {
        max_partial[blockIdx.x] = cache[0];
    }
}

static double get_max_diag(Matrix<double> &a, Dom &dom) {
    double *max_partial, *max_partial_dev;
    cudaMalloc(&max_partial_dev, sizeof(double) * n_blocks);
    max_partial = (double*)malloc(sizeof(double) * n_blocks);

    get_max_diag_kernel<<<n_blocks, n_threads>>>(*(a._dd), max_partial_dev, *(dom._size_ptr));

    cudaMemcpy(max_partial, max_partial_dev, sizeof(double) * n_blocks, cudaMemcpyDeviceToHost);

    double max_diag = max_partial[0];
    for (int i = 1; i < n_blocks; i ++) {
        if (max_partial[i] > max_diag) {
            max_diag = max_partial[i];
        }
    }

    free(max_partial);
    cudaFree(max_partial_dev);

    return max_diag;
}

__global__ static void calc_norm_kernel(MatrixCp<double> &a, double *sum_partial, dim3 &size) {
    __shared__ double cache[n_threads];
    int stride = Util::get_global_size();
    double temp_sum = 0;
    for (int idx = Util::get_global_idx(); idx < a._num; idx += stride) {
        temp_sum += a(idx) * a(idx);
    }
    cache[threadIdx.x] = temp_sum;
    __syncthreads();

    int length = n_threads;
    while (length > 1) {
        int cut = length / 2;
        int reduce = length - cut;
        if (threadIdx.x < cut) {
            cache[threadIdx.x] += cache[threadIdx.x + reduce];
        }
        __syncthreads();
        length = reduce;
    }
    
    if (threadIdx.x == 0) {
        sum_partial[blockIdx.x] = cache[0];
    }
}

static double calc_norm(Matrix<double> &a, Dom &dom) {
    double *sum_partial, *sum_partial_dev;
    cudaMalloc(&sum_partial_dev, sizeof(double) * n_blocks);
    sum_partial = (double*)malloc(sizeof(double) * n_blocks);

    calc_norm_kernel<<<n_blocks, n_threads>>>(*(a._dd), sum_partial_dev, *(dom._size_ptr));

    cudaMemcpy(sum_partial, sum_partial_dev, sizeof(double) * n_blocks, cudaMemcpyDeviceToHost);

    double sum = sum_partial[0];
    for (int i = 1; i < n_blocks; i ++) {
        sum += sum_partial[i];
    }

    free(sum_partial);
    cudaFree(sum_partial_dev);

    return sqrt(sum);
}

void assign(Matrix<double> &dst, Matrix<double> &src, int loc) {
    assert(dst._num == src._num);
    if (loc & LOCATION::HOST) {
        assert(dst._loc & src._loc & LOCATION::HOST);
        memcpy(dst._hh._arr, src._hh._arr, sizeof(double) * dst._num);
    }
    if (loc & LOCATION::DEVICE) {
        assert(dst._loc & src._loc & LOCATION::DEVICE);
        cudaMemcpy(dst._hd._arr, src._hd._arr, sizeof(double) * dst._num, cudaMemcpyDeviceToDevice);
    }
}

void clear(Matrix<double> &dst, int loc) {
    if (loc & LOCATION::HOST) {
        assert(dst._loc & LOCATION::HOST);
        memset(dst._hh._arr, 0, sizeof(double) * dst._num);
    }
    if (loc & LOCATION::DEVICE) {
        assert(dst._loc & LOCATION::DEVICE);
        cudaMemset(dst._hd._arr, 0, sizeof(double) * dst._num);
    }
}

}

#endif