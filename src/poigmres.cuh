#ifndef _POIGREMS_H_
#define _POIGREMS_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "matrix.cuh"
#include "util.cuh"
#include "mesh.cuh"
#include "poiutil.cuh"

__global__ static void fgmres_kernel_1(MatrixCp<double> &v, MatrixCp<double> &r, double beta, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        v(idx) = r(idx) / beta;
    }
}

__global__ static void fgmres_kernel_2(MatrixCp<double> &w, MatrixCp<double> &v, double h, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        w(idx) -= h * v(idx);
    }
}

__global__ static void fgmres_kernel_3(MatrixCp<double> &x, MatrixCp<double> &z, double rm, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        x(idx) += rm * z(idx);
    }
}

static void poisson_fgmres(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, double e1, double e2, int maxit, int restart, Dom &dom, LS_State &state) {
    int    n;
    bool   jump1;
    std::vector<Matrix<double>> v(restart + 1, Matrix<double>(dom._size, 1, LOCATION::DEVICE));
    std::vector<Matrix<double>> z(restart + 1, Matrix<double>(dom._size, 1, LOCATION::DEVICE));
    std::vector<std::vector<double>> h(restart + 1, std::vector<double>(restart + 1, 0));
    std::vector<double> s(restart + 1, 0);
    std::vector<double> c(restart + 1, 0);
    Matrix<double> w(dom._size, 1, LOCATION::DEVICE);

    calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
    state.re = MatrixUtil::calc_norm(r, dom) / x._num;

    for (state.it = 0; state.it < maxit; state.it ++) {
        printf("\r%12.5e %d", state.re, state.it);  
        if (state.re < e1) {
            return;
        }
        
        std::vector<double> rm(restart + 1, 0);
        rm[0] = state.re;
        fgmres_kernel_1<<<n_blocks, n_threads>>>(*(v[0]._dd), *(r._dd), state.re, *(dom._size_ptr));

        jump1 = false;

        for (int i = 0; i < restart; i ++) {
            MatrixUtil::clear(z[i], LOCATION::DEVICE);
            preconditioner_sor(a, z[i], v[i], 5, dom);
            // MatrixUtil::assign(z[i], v[i], LOCATION::DEVICE);
            calc_ax_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(z[i]._dd), *(w._dd), *(dom._size_ptr));
            for (int k = 0; k <= i; k ++) {
                h[k][i] = dot(w, v[k], dom);
                fgmres_kernel_2<<<n_blocks, n_threads>>>(*(w._dd), *(v[k]._dd), h[k][i], *(dom._size_ptr));
            }
            h[i + 1][i] = MatrixUtil::calc_norm(w, dom);

            if (h[i + 1][i] < e1) {
                /* pseudo converge */
                n = i - 1;
                jump1 = true;
                break;
            }

            fgmres_kernel_1<<<n_blocks, n_threads>>>(*(v[i + 1]._dd), *(w._dd), h[i + 1][i], *(dom._size_ptr));

            for (int k = 1; k <= i; k ++) {
                double ht = h[k - 1][i];
                h[k - 1][i] = c[k - 1] * ht + s[k - 1] * h[k][i];
                h[k    ][i] = s[k - 1] * ht + c[k - 1] * h[k][i];
            }

            double d = sqrt(h[i][i] * h[i][i] + h[i + 1][i] * h[i + 1][i]);
            if (d < e1) {
                n = i - 1;
                jump1 = true;
                break;
            }
            c[i] = h[i    ][i] / d;
            s[i] = h[i + 1][i] / d;
            h[i][i] = c[i] * h[i][i] + s[i] * h[i + 1][i];
            
            rm[i + 1] = - s[i] * rm[i];
            rm[i    ] =   c[i] * rm[i];

            if (fabs(rm[i + 1]) < e2) {
                n = i;
                jump1 = true;
            }
        }

        if (!jump1) {
            n = restart - 1;
        }

        for (int i = n; i >= 1; i --) {
            rm[i] = rm[i] / h[i][i];
            for (int k = 0; k < i; k ++) {
                rm[k] -= rm[i] * h[k][i];
            }
        }
        rm[0] = rm[0] / h[0][0];

        for (int i = 0; i <= n; i ++) {
            fgmres_kernel_3<<<n_blocks, n_threads>>>(*(x._dd), *(z[i]._dd), rm[i], *(dom._size_ptr));
        }

        calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
        state.re = MatrixUtil::calc_norm(r, dom) / x._num;
    }
    printf("\n\n");
    w.release(LOCATION::DEVICE);
    for (int i = 0; i < restart + 1; i ++) {
        v[i].release(LOCATION::DEVICE);
        z[i].release(LOCATION::DEVICE);
    }
}

#endif