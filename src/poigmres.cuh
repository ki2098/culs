#ifndef _POIGREMS_H_
#define _POIGREMS_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
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
    std::vector<Matrix<double>> v(restart + 1);
    std::vector<Matrix<double>> z(restart + 1);
    for (int i = 0; i < restart + 1; i ++) {
        v[i].init(dom._size, 1, LOCATION::DEVICE, 40 + i);
        z[i].init(dom._size, 1, LOCATION::DEVICE, 60 + i);
    }
    /* for (int i = 0; i < restart + 1; i ++) {
        printf("%d %d %d %d %d %d %d %d\n", v[i]._hh._arr == nullptr, v[i]._loc, v[i]._hd._loc, v[i]._hh._loc, z[i]._hh._arr == nullptr, z[i]._loc, z[i]._hd._loc, z[i]._hh._loc);
    } */
    Matrix<double> H(restart + 1, restart + 1, LOCATION::HOST, 31);
    Matrix<double> s(restart + 1, 1, LOCATION::HOST, 32);
    Matrix<double> c(restart + 1, 1, LOCATION::HOST, 33);
    Matrix<double> w(dom._size, 1, LOCATION::DEVICE, 34);
    Matrix<double> rm(restart + 1, 1, LOCATION::HOST, 35);

    FILE *resfile = fopen("./gmres_res.csv", "w");
    fprintf(resfile, "it,res\n");

    calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
    double beta = MatrixUtil::calc_norm(r, dom);
    state.re = beta / x._num;

    for (state.it = 0; state.it < maxit; state.it ++) {
        fprintf(resfile, "%d,%.5e\n", state.it, state.re);
        printf("\r%12.5e %d", state.re, state.it);  
        if (state.re < e1) {
            break;
        }
        MatrixUtil::clear(rm, LOCATION::HOST);
        rm(0) = beta;
        fgmres_kernel_1<<<n_blocks, n_threads>>>(*(v[0]._dd), *(r._dd), beta, *(dom._size_ptr));
        for (int i = 0; i < restart; i ++) {
            MatrixUtil::clear(z[i], LOCATION::DEVICE);
            preconditioner_sor(a, z[i], v[i], 5, dom);
            calc_ax_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(z[i]._dd), *(w._dd), *(dom._size_ptr));
            for (int k = 0; k <= i; k ++) {
                H(k, i) = dot(w, v[k], dom);
                fgmres_kernel_2<<<n_blocks, n_threads>>>(*(w._dd), *(v[k]._dd), H(k, i), *(dom._size_ptr));
            }
            H(i + 1, i) = MatrixUtil::calc_norm(w, dom);

            if (H(i + 1, i) < e1) {
                n = i - 1;
                jump1 = true;
                break;
            }

            fgmres_kernel_1<<<n_blocks, n_threads>>>(*(v[i + 1]._dd), *(w._dd), H(i + 1, i), *(dom._size_ptr));

            for (int k = 1; k <= i; k ++) {
                double ht = H(k - 1, i);
                H(k - 1, i) = c(k - 1) * ht + s(k - 1) * H(k, i);
                H(k    , i) = s(k - 1) * ht + c(k - 1) * H(k, i);
            }

            double d = sqrt(H(i, i) * H(i, i) + H(i + 1, i) * H(i + 1, i));
            if (d < e1) {
                n = i - 1;
                jump1 = true;
                break;
            }
            c(i) = H(i    , i) / d;
            s(i) = H(i + 1, i) / d;
            H(i, i) = c(i) * H(i, i) + s(i) * H(i + 1, i);
            rm(i + 1) = - s(i) * rm(i);
            rm(i    ) =   c(i) * rm(i);
            double rho = fabs(rm(i + 1));
            if (rho < e2) {
                n = i;
                jump1 = true;
                break;
            }
        }
        if(!jump1) {
            n = restart - 1;
        }

        for (int i = n; i >= 1; i --) {
            rm(i) = rm(i) / H(i, i);
            for (int k = 0; k < i; k ++) {
                rm(k) -= rm(i) * H(k, i);
            }
        }
        rm(0) = rm(0) / H(0, 0);

        for (int i = 0; i <= n; i ++) {
            fgmres_kernel_3<<<n_blocks, n_threads>>>(*(x._dd), *(z[i]._dd), rm(i), *(dom._size_ptr));
        }

        calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
        beta = MatrixUtil::calc_norm(r, dom);
        state.re = beta / x._num;
    }
    printf("\n\n");
    fclose(resfile);
}

/* double sign(double x) {
    if (x == 0) {
        return 1;
    }
    return x / fabs(x);
}

__global__ static void householder_kernel_1(MatrixCp<double> &w, double norm, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        w(idx) /= norm;
    }
}

__global__ static void householder_kernel_2(MatrixCp<double> &v, MatrixCp<double> &w, int inner, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        v(idx) = - 2 * w(inner) * w(idx);
        if (idx == inner) {
            v(idx) += 1;
        }
    }
}

__global__ static void apply_householder_kernel(MatrixCp<double> &v, MatrixCp<double> &w, double vwp, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        v(idx) -= 2 * vwp * w(idx);
    }
}

static void gmres_householder(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, double e, int maxit, int restart, Dom &dom, LS_State &state) {
    Matrix<double> H(restart, restart, LOCATION::HOST, 31);
    Matrix<double> s(restart, 1, LOCATION::HOST, 32);
    Matrix<double> c(restart, 1, LOCATION::HOST, 33);
    Matrix<double> w(dom._size, 1, LOCATION::DEVICE, 34);
    Matrix<double> v(dom._size, 1, LOCATION::DEVICE, 35);
    Matrix<double> vt(dom._size, 1, LOCATION::DEVICE, 36);
    std::vector<Matrix<double>> M(restart + 1);
    for (int i = 0; i < restart + 1; i ++) {
        M[i].init(dom._size, 1, LOCATION::DEVICE, 40 + i);
    }
    Matrix<double> g(dom._size, 1, LOCATION::HOST, 37);

    calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
    double normr = MatrixUtil::calc_norm(r, dom);
    state.re = normr / x._num;

    for (state.it = 0; state.it < maxit; state.it ++) {
        printf("\r%12.5e %d", state.re, state.it);  
        if (state.re < e) {
            break;
        }

        MatrixUtil::assign(w, r, LOCATION::DEVICE);
        double w0, beta;
        cudaMemcpy(&w0, w._hd._arr, sizeof(double), cudaMemcpyDeviceToHost);
        beta = sign(w0) * normr;
        w0 += beta;
        cudaMemcpy(w._hd._arr, &w0, sizeof(double), cudaMemcpyHostToDevice);
        double normw = MatrixUtil::calc_norm(w, dom);
        householder_kernel_1<<<n_blocks, n_threads>>>(*(w._dd), normw, *(dom._size_ptr));
        MatrixUtil::assign(M[0], w, LOCATION::DEVICE);
        g(0) = - beta;

        for (int i = 0; i < restart; i ++) {
            householder_kernel_2<<<n_blocks, n_threads>>>(*(v._dd), *(w._dd), i, *(dom._size_ptr));
            for (int k = i - 1; k >= 0; k --) {
                double vwp = dot(v, M[k], dom);
                apply_householder_kernel<<<n_blocks, n_threads>>>(*(v._dd), *(M[k]._dd), vwp, *(dom._size_ptr));
            }

            calc_ax_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(v._dd), *(vt._dd), *(dom._size_ptr));

            MatrixUtil::clear(v, LOCATION::DEVICE);
            preconditioner_sor(a, v, vt, 5, dom);

            for (int k = 0; k <= i; k ++) {
                double vwp = dot(v, M[k], dom);
                apply_householder_kernel<<<n_blocks, n_threads>>>(*(v._dd), *(M[k]._dd), vwp, *(dom._size_ptr));
            }

            if (i < a._row - 1) {
                
            }
        }
    }
} */

#endif