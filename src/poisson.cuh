#ifndef _POISSON_H
#define _POISSON_H 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "mesh.cuh"
#include "util.cuh"
#include "param.cuh"

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

static void poisson_sor(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, double e, int maxit, Dom &dom, LS_State &state) {
    for (state.it = 1; state.it <= maxit; state.it ++) {
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, 0, *(dom._size_ptr));
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, 1, *(dom._size_ptr));
        calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
        state.re = MatrixUtil::calc_norm(r, dom) / x._num;

        printf("\r%12.5e %d", state.re, state.it);
        if (state.re < e) {
            break;
        }
    }
    printf("\n\n");
}

static void preconditioner_sor(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, int maxit, Dom &dom) {
    for (int it = 1; it <= maxit; it ++) {
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, 0, *(dom._size_ptr));
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, 1, *(dom._size_ptr));
    }
}

static void preconditioner_jacobi(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, int maxit, Dom &dom) {
    Matrix<double> xp(dom._size, 1, LOCATION::DEVICE);
    for (int it = 1; it <= maxit; it ++) {
        MatrixUtil::assign(xp, x, LOCATION::DEVICE);
        poisson_jacobi_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(xp._dd), *(b._dd), *(dom._size_ptr));
    }
    xp.release(LOCATION::DEVICE);
}

__global__ static void pbicgstab_kernel_1(MatrixCp<double> &p, MatrixCp<double> &q, MatrixCp<double> &r, double beta, double omega, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        p(idx) = r(idx) + beta * (p(idx) - omega * q(idx));
    }
}

__global__ static void pbicgstab_kernel_2(MatrixCp<double> &s, MatrixCp<double> &q, MatrixCp<double> &r, double alpha, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        s(idx) = r(idx) - alpha * q(idx);
    }
}

__global__ static void pbicgstab_kernel_3(MatrixCp<double> &x, MatrixCp<double> &pp, MatrixCp<double> &ss, double alpha, double omega, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        x(idx) += (alpha * pp(idx) + omega * ss(idx));
    }
}

__global__ static void pbicgstab_kernel_4(MatrixCp<double> &r, MatrixCp<double> &s, MatrixCp<double> &t, double omega, dim3 &size) {
    int stride = Util::get_global_size();
    for (int idx = Util::get_global_idx(); idx < size.x; idx += stride) {
        r(idx) = s(idx) - omega * t(idx);
    }
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

static void poisson_pbicgstab(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, double e, int maxit, Dom &dom, LS_State &state) {
    Matrix<double> rr(dom._size, 1, LOCATION::DEVICE);
    Matrix<double>  p(dom._size, 1, LOCATION::DEVICE);
    Matrix<double>  q(dom._size, 1, LOCATION::DEVICE);
    Matrix<double>  s(dom._size, 1, LOCATION::DEVICE);
    Matrix<double> pp(dom._size, 1, LOCATION::DEVICE);
    Matrix<double> ss(dom._size, 1, LOCATION::DEVICE);
    Matrix<double>  t(dom._size, 1, LOCATION::DEVICE);

    double rho, rrho, alpha, beta, omega;

    calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
    
    MatrixUtil::assign(rr, r, LOCATION::DEVICE);

    rrho  = 1;
    alpha = 0;
    omega = 1;

    for (state.it = 1; state.it <= maxit; state.it ++) {
        rho = dot(r, rr, dom);
        if (fabs(rho) < __FLT_MIN__) {
            printf("\nsmall rho: %12.5e\n", rho);
            state.re = rho;
            state.it -= 1;
            break;
        }

        if (state.it == 1) {
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

        printf("\r%12.5e %d", state.re, state.it);
        if (state.re < e) {
            break;
        }
    }
    printf("\n\n");

    rr.release(LOCATION::DEVICE);
    p.release(LOCATION::DEVICE);
    q.release(LOCATION::DEVICE);
    s.release(LOCATION::DEVICE);
    pp.release(LOCATION::DEVICE);
    ss.release(LOCATION::DEVICE);
    t.release(LOCATION::DEVICE);
}

#endif