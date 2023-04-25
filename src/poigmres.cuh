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

    calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
    double beta = MatrixUtil::calc_norm(r, dom);
    state.re = beta / x._num;

    for (state.it = 0; state.it < maxit; state.it ++) {
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

    // for (state.it = 0; state.it < maxit; state.it ++) {
    //     // printf("\r%12.5e %d", state.re, state.it);  
    //     if (state.re < e1) {
    //         return;
    //     }
        
    //     printf("%d_______________________________________________________________________________________________\n", state.it);

        
    //     rm[0] = state.re;
    //     fgmres_kernel_1<<<n_blocks, n_threads>>>(*(v[0]._dd), *(r._dd), state.re, *(dom._size_ptr));
        
    //     v[0].sync_d2h();
    //     printf("rm= ");
    //     for(int _ = 0; _ < N; _ ++) {
    //         printf("%lf ", rm[_]);
    //     }
    //     printf("\n");
    //     printf("v0= ");
    //     for(int _ = 0; _ < N; _ ++) {
    //         printf("%lf ", v[0](_));
    //     }
    //     printf("\n");

    //     jump1 = false;

    //     for (int i = 0; i < restart; i ++) {
    //         printf("%d.%d_____________________________________________________________________________________________\n", state.it, i);
    //         printf("v= ");
    //         v[i].sync_d2h();
    //         for(int _ = 0; _ < N; _ ++) {
    //             printf("%lf ", v[i](_));
    //         }
    //         printf("\n");

    //         MatrixUtil::clear(z[i], LOCATION::DEVICE);
    //         printf("v= ");
    //         v[i].sync_d2h();
    //         for(int _ = 0; _ < N; _ ++) {
    //             printf("%lf ", v[i](_));
    //         }
    //         printf("\n");
    //         for (int _ = 1; _ <= 5; _ ++) {
    //             poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(z[i]._dd), *(v[i]._dd), sor_omega, 0, *(dom._size_ptr));
    //             poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(z[i]._dd), *(v[i]._dd), sor_omega, 1, *(dom._size_ptr));
    //         }

    //         /* printf("v= ");
    //         v[i].sync_d2h();
    //         for(int _ = 0; _ < N; _ ++) {
    //             printf("%lf ", v[i](_));
    //         }
    //         printf("\n"); */

    //         calc_ax_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(z[i]._dd), *(w._dd), *(dom._size_ptr));

            
    //         printf("z= ");
    //         z[i].sync_d2h();
    //         for(int _ = 0; _ < N; _ ++) {
    //             printf("%lf ", z[i](_));
    //         }
    //         printf("\n");
    //         printf("w= ");
    //         w.sync_d2h();
    //         for(int _ = 0; _ < N; _ ++) {
    //             printf("%lf ", w(_));
    //         }
    //         printf("\n");

    //         for (int k = 0; k <= i; k ++) {
    //             h[k][i] = dot(w, v[k], dom);
    //             fgmres_kernel_2<<<n_blocks, n_threads>>>(*(w._dd), *(v[k]._dd), h[k][i], *(dom._size_ptr));

    //             printf("dot= %lf\n",h[k][i]);
    //             printf("w= ");
    //             w.sync_d2h();
    //             for(int _ = 0; _ < N; _ ++) {
    //                 printf("%lf ", w(_));
    //             }
    //             printf("\n");
    //         }
    //         h[i + 1][i] = MatrixUtil::calc_norm(w, dom);

    //         if (h[i + 1][i] < e1) {
    //             /* pseudo converge */
    //             n = i - 1;
    //             jump1 = true;
    //             break;
    //         }

    //         fgmres_kernel_1<<<n_blocks, n_threads>>>(*(v[i + 1]._dd), *(w._dd), h[i + 1][i], *(dom._size_ptr));

    //         for (int k = 1; k <= i; k ++) {
    //             double ht = h[k - 1][i];
    //             h[k - 1][i] = c[k - 1] * ht + s[k - 1] * h[k][i];
    //             h[k    ][i] = s[k - 1] * ht + c[k - 1] * h[k][i];
    //         }

    //         double d = sqrt(h[i][i] * h[i][i] + h[i + 1][i] * h[i + 1][i]);
    //         if (d < e1) {
    //             n = i - 1;
    //             jump1 = true;
    //             break;
    //         }
    //         c[i] = h[i    ][i] / d;
    //         s[i] = h[i + 1][i] / d;
    //         h[i][i] = c[i] * h[i][i] + s[i] * h[i + 1][i];

    //         rm[i + 1] = - s[i] * rm[i];
    //         rm[i    ] =   c[i] * rm[i];

    //         if (fabs(rm[i + 1]) < e2) {
    //             n = i;
    //             jump1 = true;
    //         }
    //     }

    //     if (!jump1) {
    //         n = restart - 1;
    //     }

    //     for (int i = n; i >= 1; i --) {
    //         rm[i] = rm[i] / h[i][i];
    //         for (int k = 0; k < i; k ++) {
    //             rm[k] -= rm[i] * h[k][i];
    //         }
    //     }
    //     rm[0] = rm[0] / h[0][0];

    //     for (int i = 0; i <= n; i ++) {
    //         fgmres_kernel_3<<<n_blocks, n_threads>>>(*(x._dd), *(z[i]._dd), rm[i], *(dom._size_ptr));
    //     }

    //     calc_res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), *(dom._size_ptr));
    //     state.re = MatrixUtil::calc_norm(r, dom);

    //     // for (int i = 0; i < restart + 1; i ++) {
    //     //     printf("%lf ", rm[i]);
    //     // }
    //     // printf("\n");
    // }
    // printf("\n\n");

    // for (int i = 0; i < restart + 1; i ++) {
    //     printf("%lf %lf %lf\n", s[i], c[i], rm[i]);
    // }
    // for (int i = 0; i < restart + 1; i ++) {
    //     for (int j = 0; j < restart + 1; j ++) {
    //         printf("%lf ", h[i][j]);
    //     }
    //     printf("\n");
    // }
}

#endif