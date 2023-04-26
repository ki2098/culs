#ifndef _MATRIX_H_
#define _MATRIX_H_ 1

#include <float.h>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dom.cuh"
#include "util.cuh"
#include "param.h"

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
    int _lab;
    MatrixCp(dim3 &size, int col, int loc, int lab);
    MatrixCp(int row, int col, int loc, int lab);
    MatrixCp();
    ~MatrixCp();
    void init(dim3 &size, int col, int loc, int lab);
    void init(int row, int col, int loc, int lab);
    void release();
    __host__ __device__ T& operator()(int idx) {return _arr[idx];}
    __host__ __device__ T& operator()(int row_idx, int col_idx) {return _arr[col_idx * _row + row_idx];}
};

template<class T>
MatrixCp<T>::MatrixCp() : _row(0), _col(0), _num(0), _loc(LOCATION::NONE), _lab(0), _arr(nullptr) {/* printf("Default constructor of MatrixCp called\n"); */}

template<class T>
MatrixCp<T>::MatrixCp(dim3 &size, int col, int loc, int lab) : _row(size.x * size.y * size.z), _col(col), _num(size.x * size.y * size.z * col), _loc(loc), _lab(lab) {
    if (loc == LOCATION::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
    } else if (loc == LOCATION::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
    }
}

template<class T>
MatrixCp<T>::MatrixCp(int row, int col, int loc, int lab) : _row(row), _col(col), _num(row * col), _loc(loc), _lab(lab) {
    if (loc == LOCATION::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
    } else if (loc == LOCATION::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
    }
}

template<class T>
void MatrixCp<T>::init(dim3 &size, int col, int loc, int lab) {
    assert(_loc == LOCATION::NONE);
    _row = size.x * size.y * size.z;
    _col = col;
    _num = _row * _col;
    _loc = loc;
    _lab = lab;
    if (loc == LOCATION::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
        // printf("initializer of MatrixCp %d called to free on HOST\n", _lab);
    } else if (loc == LOCATION::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
        // printf("initializer of MatrixCp %d called to free on DEVICE\n", _lab);
    }
}

template<class T>
void MatrixCp<T>::init(int row, int col, int loc, int lab) {
    assert(_loc == LOCATION::NONE);
    _row = row;
    _col = col;
    _num = _row * _col;
    _loc = loc;
    _lab = lab;
    if (loc == LOCATION::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
        // printf("initializer of MatrixCp %d called to init on HOST\n", _lab);
    } else if (loc == LOCATION::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
        // printf("initializer of MatrixCp %d called to init on DEVICE\n", _lab);
    }
}

template<class T>
void MatrixCp<T>::release() {
    if (_loc == LOCATION::HOST) {
        // printf("release of MatrixCp %d called to free on HOST\n", _lab);
        free(_arr);
    } else if (_loc == LOCATION::DEVICE) {
        // printf("release of MatrixCp %d called to free on DEVICE\n", _lab);
        cudaFree(_arr);
    }
    _loc = LOCATION::NONE;
}

template<class T>
MatrixCp<T>::~MatrixCp() {
    if (_loc == LOCATION::HOST) {
        // printf("destructor of MatrixCp %d called to free on HOST\n", _lab);
        free(_arr);
        _loc &= (~LOCATION::HOST);
    } else if (_loc == LOCATION::DEVICE) {
        // printf("destructor of MatrixCp %d called to free on DEVICE\n", _lab);
        cudaFree(_arr);
        _loc &= (~LOCATION::DEVICE);
    }
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
    int         _lab;
    Matrix(dim3 &size, int col, int loc, int lab);
    Matrix(int row, int col, int loc, int lab);
    Matrix();
    ~Matrix();
    void init(dim3 &size, int col, int loc, int lab);
    void init(int row, int col, int loc, int lab);
    void release(int loc);
    void sync_h2d();
    void sync_d2h();
    __host__ __device__ T& operator()(int idx) {return _hh(idx);}
    __host__ __device__ T& operator()(int row_idx, int col_idx) {return _hh(row_idx, col_idx);}
};

template<class T>
Matrix<T>::Matrix() : _row(0), _col(0), _num(0), _loc(LOCATION::NONE), _lab(0), _dd(nullptr) {/* printf("Default constructor of Matrix called\n"); */}

template<class T>
Matrix<T>::Matrix(dim3 &size, int col, int loc, int lab) : _row(size.x * size.y * size.z), _col(col), _num(size.x * size.y * size.z * col), _loc(loc), _lab(lab), _hh(size, col, (loc & LOCATION::HOST), lab), _hd(size, col, (loc & LOCATION::DEVICE), lab), _dd(nullptr) {
    if (loc & LOCATION::DEVICE) {
        cudaMalloc(&_dd, sizeof(MatrixCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(MatrixCp<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
Matrix<T>::Matrix(int row, int col, int loc, int lab) : _row(row), _col(col), _num(row * col), _loc(loc), _lab(lab), _hh(row, col, (loc & LOCATION::HOST), lab), _hd(row, col, (loc & LOCATION::DEVICE), lab), _dd(nullptr) {
    if (loc & LOCATION::DEVICE) {
        cudaMalloc(&_dd, sizeof(MatrixCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(MatrixCp<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
void Matrix<T>::init(dim3 &size, int col, int loc, int lab) {
    assert(_loc == LOCATION::NONE);
    _row = size.x * size.y * size.z;
    _col = col;
    _num = _row * _col;
    _loc = loc;
    _lab = lab;
    _hh.init(_row, _col, _loc & LOCATION::HOST  , _lab);
    _hd.init(_row, _col, _loc & LOCATION::DEVICE, _lab);
    if (loc & LOCATION::DEVICE) {
        cudaMalloc(&_dd, sizeof(MatrixCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(MatrixCp<T>), cudaMemcpyHostToDevice);
        // printf("initializer of Matrix %d called to init on DEVICE\n", _lab);
    }
}

template<class T>
void Matrix<T>::init(int row, int col, int loc, int lab) {
    assert(_loc == LOCATION::NONE);
    _row = row;
    _col = col;
    _num = _row * _col;
    _loc = loc;
    _lab = lab;
    _hh.init(_row, _col, _loc & LOCATION::HOST  , _lab);
    _hd.init(_row, _col, _loc & LOCATION::DEVICE, _lab);
    if (loc & LOCATION::DEVICE) {
        cudaMalloc(&_dd, sizeof(MatrixCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(MatrixCp<T>), cudaMemcpyHostToDevice);
        // printf("initializer of Matrix %d called to init on DEVICE\n", _lab);
    }
}

template<class T>
void Matrix<T>::release(int loc) {
    if ((loc & LOCATION::HOST) && (_loc & LOCATION::HOST)) {
        // printf("release of Matrix %d called to free on HOST\n", _lab);
        _hh.release();
        _loc &= (~LOCATION::HOST);
    }
    if ((loc & LOCATION::DEVICE) && (_loc & LOCATION::DEVICE)) {
        // printf("release of Matrix %d called to free on DEVICE\n", _lab);
        _hd.release();
        cudaFree(_dd);
        _loc &= (~LOCATION::DEVICE);
    }

}

template<class T>
Matrix<T>::~Matrix() {
    if (_loc & LOCATION::DEVICE) {
        // printf("destructor of Matrix %d called to free on DEVICE\n", _lab);
        cudaFree(_dd);
    }
    _loc = LOCATION::NONE;
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

/* template<class T>
struct SliceCp {
    int _anchor_row;
    int _anchor_col;
    int _row;
    int _col;
    int _num;
    int _loc;
    MatrixCp<T> *_original;
    __host__ __device__ T& operator()(int row_idx, int col_idx) {return _original->operator()(row_idx + _anchor_row, col_idx + _anchor_col);}
    __host__ __device__ T& operator()(int idx) {
        assert(_col == 1);
        return _original->operator()(idx + _anchor_row, _anchor_col);
    }
    SliceCp(MatrixCp<T> &original, int row_start, int row_end, int col_start, int col_end) : _original(&original), _anchor_row(row_start), _anchor_col(col_start), _row(row_end - row_start), _col(col_end - col_start), _loc(original._loc) {
        _num = _row * _col;
    }
};

template<class T>
struct Slice {
    SliceCp<T>  _hh;
    SliceCp<T>  _hd;
    SliceCp<T> *_dd;
    int _anchor_row;
    int _anchor_col;
    int _row;
    int _col;
    int _num;
    int _loc;
    Matrix<T> *_original;
    __host__ __device__ T& operator()(int row_idx, int col_idx) {return _original->operator()(row_idx + _anchor_row, col_idx + _anchor_col);}
    __host__ __device__ T& operator()(int idx) {
        assert(_col == 1);
        return _original->operator()(idx + _anchor_row, _anchor_col);
    }
    Slice(Matrix<T> &original, int row_start, int row_end, int col_start, int col_end) :_hh(original._hh, row_start, row_end, col_start, col_end), _hd(original._hd, row_start, row_end, col_start, col_end), _original(&original), _anchor_row(row_start), _anchor_col(col_start), _row(row_end - row_start), _col(col_end - col_start), _loc(original._loc) {
        _num = _row * _col;
        if (original._loc & LOCATION::DEVICE) {
            cudaMalloc(&_dd, sizeof(SliceCp<T>));
            SliceCp<T> temp(original._hd);
            temp._original = original._dd;
            cudaMemcpy(_dd, &temp, sizeof(SliceCp<T>), cudaMemcpyHostToDevice);
        }
    }
}; */

namespace MatrixUtil {

__global__ static void get_max_diag_kernel(MatrixCp<double> &a, double *max_partial, dim3 &size) {
    __shared__ double cache[n_threads];
    int stride = Util::get_global_size();
    double temp_max = 0;
    for (int idx = Util::get_global_idx(); idx < a._row; idx += stride) {
        double _value = fabs(a(idx, 0));
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