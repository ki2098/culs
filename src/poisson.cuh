#ifndef _POISSON_H_
#define _POISSON_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "mesh.cuh"
#include "util.cuh"
#include "param.h"
#include "poiutil.cuh"

static void poisson_sor(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, double e, int maxit, Dom &dom, LS_State &state) {
    for (state.it = 0; state.it < maxit;) {
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, 0, *(dom._size_ptr));
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, 1, *(dom._size_ptr));
        calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
        state.re = MatrixUtil::calc_norm(r, dom) / x._num;
        state.it ++;

        printf("\r%12.5e %d", state.re, state.it);
        if (state.re < e) {
            break;
        }
    }
    printf("\n\n");
}



__global__ static void pbicgstab_kernel_1(MatrixCp<double> &p, MatrixCp<double> &q, MatrixCp<double> &r, double beta, double omega, dim3 &size) {
    // int stride = Util::get_global_size();
    // for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
    //     p(idx) = r(idx) + beta * (p(idx) - omega * q(idx));
    // }
    int idx = Util::get_global_idx();
    if (idx < size.x) {
        p(idx) = r(idx) + beta * (p(idx) - omega * q(idx));
    }
}

__global__ static void pbicgstab_kernel_2(MatrixCp<double> &s, MatrixCp<double> &q, MatrixCp<double> &r, double alpha, dim3 &size) {
    // int stride = Util::get_global_size();
    // for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
    //     s(idx) = r(idx) - alpha * q(idx);
    // }
    int idx = Util::get_global_idx();
    if (idx < size.x) {
        s(idx) = r(idx) - alpha * q(idx);
    }
}

__global__ static void pbicgstab_kernel_3(MatrixCp<double> &x, MatrixCp<double> &pp, MatrixCp<double> &ss, double alpha, double omega, dim3 &size) {
    // int stride = Util::get_global_size();
    // for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
    //     x(idx) += (alpha * pp(idx) + omega * ss(idx));
    // }
    int idx = Util::get_global_idx();
    if (idx < size.x) {
        x(idx) += (alpha * pp(idx) + omega * ss(idx));
    }
}

__global__ static void pbicgstab_kernel_4(MatrixCp<double> &r, MatrixCp<double> &s, MatrixCp<double> &t, double omega, dim3 &size) {
    // int stride = Util::get_global_size();
    // for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
    //     r(idx) = s(idx) - omega * t(idx);
    // }
    int idx = Util::get_global_idx();
    if (idx < size.x) {
        r(idx) = s(idx) - omega * t(idx);
    }
}



static void poisson_pbicgstab(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, double e, int maxit, Dom &dom, LS_State &state) {
    Matrix<double> rr(dom._size, 1, LOCATION::DEVICE, 21);
    Matrix<double>  p(dom._size, 1, LOCATION::DEVICE, 22);
    Matrix<double>  q(dom._size, 1, LOCATION::DEVICE, 23);
    Matrix<double>  s(dom._size, 1, LOCATION::DEVICE, 24);
    Matrix<double> pp(dom._size, 1, LOCATION::DEVICE, 25);
    Matrix<double> ss(dom._size, 1, LOCATION::DEVICE, 26);
    Matrix<double>  t(dom._size, 1, LOCATION::DEVICE, 27);

    double rho, rrho, alpha, beta, omega;

    calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
    state.re = MatrixUtil::calc_norm(r, dom) / x._num;
    MatrixUtil::assign(rr, r, LOCATION::DEVICE);

    FILE *resfile = fopen("./pbicgstab_res.csv", "w");
    fprintf(resfile, "it,res\n");

    rrho  = 1;
    alpha = 0;
    omega = 1;

    for (state.it = 0; state.it < maxit; state.it ++) {
        fprintf(resfile, "%d,%.5e\n", state.it, state.re);
        printf("\r%12.5e %d", state.re, state.it);  
        if (state.re < e) {
            break;
        }
        rho = dot(r, rr, dom);
        if (fabs(rho) < __FLT_MIN__) {
            printf("\nsmall rho: %12.5e\n", rho);
            state.re = rho;
            break;
        }

        if (state.it == 0) {
            MatrixUtil::assign(p, r, LOCATION::DEVICE);
        } else {
            beta = (rho * alpha) / (rrho * omega);
            pbicgstab_kernel_1<<<n_blocks, n_threads>>>(*(p._dd), *(q._dd), *(r._dd), beta, omega, *(dom._size_ptr));
        }
        MatrixUtil::clear(pp, LOCATION::DEVICE);
        preconditioner_sor(a, pp, p, 5, dom);
        calc_ax_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(pp._dd), *(q._dd), *(dom._size_ptr));
        alpha = rho / dot(rr, q, dom);

        pbicgstab_kernel_2<<<n_blocks, n_threads>>>(*(s._dd), *(q._dd), *(r._dd), alpha, *(dom._size_ptr));
        MatrixUtil::clear(ss, LOCATION::DEVICE);
        preconditioner_sor(a, ss, s, 5, dom);
        calc_ax_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(ss._dd), *(t._dd), *(dom._size_ptr));
        omega = dot(t, s, dom) / dot(t, t, dom);

        pbicgstab_kernel_3<<<n_blocks, n_threads>>>(*(x._dd), *(pp._dd), *(ss._dd), alpha, omega, *(dom._size_ptr));
        pbicgstab_kernel_4<<<n_blocks, n_threads>>>(*(r._dd), *(s._dd), *(t._dd), omega, *(dom._size_ptr));

        rrho = rho;
        state.re = MatrixUtil::calc_norm(r, dom) / x._num; 
    }
    printf("\n\n");
    fclose(resfile);
    // rr.release(LOCATION::DEVICE);
    // p.release(LOCATION::DEVICE);
    // q.release(LOCATION::DEVICE);
    // s.release(LOCATION::DEVICE);
    // pp.release(LOCATION::DEVICE);
    // ss.release(LOCATION::DEVICE);
    // t.release(LOCATION::DEVICE);
}

#endif