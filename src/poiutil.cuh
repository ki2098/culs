#ifndef _POIUTIL_H_
#define _POIUTIL_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "matrix.cuh"
#include "mesh.cuh"
#include "util.cuh"

struct LS_State {
    double re;
    int    it;
};

__global__ static void dot_kernel(MatrixCp<double> &a, MatrixCp<double> &b, double *sum_partial, dim3 &size) {
    __shared__ double cache[n_threads];
    int stride = Util::get_global_size();
    double temp_sum = 0;
    for (int idx = Util::get_global_idx(); idx < a._num; idx += stride) {
        temp_sum += a(idx) * b(idx);
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

static double dot(Matrix<double> &a, Matrix<double> &b, Dom &dom) {
    double *sum_partial, *sum_partial_dev;
    cudaMalloc(&sum_partial_dev, sizeof(double) * n_blocks);
    sum_partial = (double*)malloc(sizeof(double) * n_blocks);

    dot_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(b._dd), sum_partial_dev, *(dom._size_ptr));

    cudaMemcpy(sum_partial, sum_partial_dev, sizeof(double) * n_blocks, cudaMemcpyDeviceToHost);

    double sum = sum_partial[0];
    for (int i = 1; i < n_blocks; i ++) {
        sum += sum_partial[i];
    }

    free(sum_partial);
    cudaFree(sum_partial_dev);

    return sum;
}

__global__ static void prepare_poisson_eq_kernel(MatrixCp<double> &a, MatrixCp<double> &b, MatrixCp<double> &x, MatrixCp<double> &h, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        if (idx == 0) {
            double dxr = x(idx + 1, 0) - x(idx, 0);
            double dx  = h(idx, 0);
            double ar  = kappa / (dx * dxr);
            double al  = 0;
            double ac  = - (ar + 2 * kappa / (dx * dx));
            double bc  = - (2 * kappa * TL) / (dx * dx);
            a(idx, 0)  = ac;
            a(idx, 1)  = al;
            a(idx, 2)  = ar;
            b(idx)     = bc;
        } else if (idx == size.x - 1) {
            double dxl = x(idx, 0) - x(idx - 1, 0);
            double dx  = h(idx, 0);
            double ar  = 0;
            double al  = kappa / (dx * dxl);
            double ac  = - (al + 2 * kappa / (dx * dx));
            double bc  = - (2 * kappa * TR) / (dx * dx);
            a(idx, 0)  = ac;
            a(idx, 1)  = al;
            a(idx, 2)  = ar;
            b(idx)     = bc;
        } else {
            double dxr = x(idx + 1, 0) - x(idx, 0);
            double dxl = x(idx, 0) - x(idx - 1, 0);
            double dx  = h(idx, 0);
            double ar  = kappa / (dx * dxr);
            double al  = kappa / (dx * dxl);
            double ac  = - (ar + al);
            double bc  = 0;
            a(idx, 0)  = ac;
            a(idx, 1)  = al;
            a(idx, 2)  = ar;
            b(idx)     = bc;
        }
    }
}

__global__ static void scale_eq_kernel(MatrixCp<double> &a, MatrixCp<double> &b, dim3 &size, double max_diag) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        a(idx, 0) /= max_diag;
        a(idx ,1) /= max_diag;
        a(idx, 2) /= max_diag;
        b(idx)    /= max_diag;
    }
}

static void prepare_poisson_eq(Matrix<double> &a, Matrix<double> &b, Mesh &mesh, Dom &dom) {
    prepare_poisson_eq_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(b._dd), *(mesh.x._dd), *(mesh.h._dd), *(dom._size_ptr));

    double max_diag = MatrixUtil::get_max_diag(a, dom);
    printf("max diag = %lf\n", max_diag);

    scale_eq_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(b._dd), *(dom._size_ptr), max_diag);
}

__global__ static void calc_res_kernel(MatrixCp<double> &a, MatrixCp<double> &x, MatrixCp<double> &b, MatrixCp<double> &r, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        double ac = a(idx, 0);
        double al = a(idx, 1);
        double ar = a(idx, 2);
        int    ic = idx;
        int    il = (idx > 0         )? idx - 1 : idx;
        int    ir = (idx < size.x - 1)? idx + 1 : idx;
        double xc = x(ic);
        double xl = x(il);
        double xr = x(ir);
        r(idx) = b(idx) - (ac * xc + al * xl + ar * xr);
    }
}

__global__ static void poisson_sor_kernel(MatrixCp<double> &a, MatrixCp<double> &x, MatrixCp<double> &b, double omega, int color, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        double ac = a(idx, 0);
        double al = a(idx, 1);
        double ar = a(idx, 2);
        int    ic = idx;
        int    il = (idx > 0         )? idx - 1 : idx;
        int    ir = (idx < size.x - 1)? idx + 1 : idx;
        double xc = x(ic);
        double xl = x(il);
        double xr = x(ir);
        double cc = (b(idx) - (ac * xc + al * xl + ar * xr)) / ac;
        if (idx % 2 != color) {
            cc = 0;
        }
        x(idx) = xc + omega * cc;
    }
}

__global__ static void poisson_jacobi_kernel(MatrixCp<double> &a, MatrixCp<double> &xn, MatrixCp<double> &xp, MatrixCp<double> &b, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        double ac = a(idx, 0);
        double al = a(idx, 1);
        double ar = a(idx, 2);
        int    ic = idx;
        int    il = (idx > 0         )? idx - 1 : idx;
        int    ir = (idx < size.x - 1)? idx + 1 : idx;
        double xc = xp(ic);
        double xl = xp(il);
        double xr = xp(ir);
        double cc = (b(idx) - (ac * xc + al * xl + ar * xr)) / ac;
        xn(idx) = xc + cc;
    }
}

static void preconditioner_sor(const Matrix<double> &a, Matrix<double> &x, const Matrix<double> &b, int maxit, Dom &dom) {
    for (int it = 1; it <= maxit; it ++) {
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, 0, *(dom._size_ptr));
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, 1, *(dom._size_ptr));
    }
}

static void preconditioner_jacobi(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, int maxit, Dom &dom) {
    Matrix<double> xp(dom._size, 1, LOCATION::DEVICE, -1);
    for (int it = 1; it <= maxit; it ++) {
        MatrixUtil::assign(xp, x, LOCATION::DEVICE);
        poisson_jacobi_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(xp._dd), *(b._dd), *(dom._size_ptr));
    }
    xp.release(LOCATION::DEVICE);
}

__global__ static void calc_ax_kernel(MatrixCp<double> &a, MatrixCp<double> &x, MatrixCp<double> &ax, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        double ac = a(idx, 0);
        double al = a(idx, 1);
        double ar = a(idx, 2);
        int    ic = idx;
        int    il = (idx > 0         )? idx - 1 : idx;
        int    ir = (idx < size.x - 1)? idx + 1 : idx;
        double xc = x(ic);
        double xl = x(il);
        double xr = x(ir);
        ax(idx)   = ac * xc + al * xl + ar * xr;
    }
}

#endif