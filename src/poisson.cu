#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "poisson.cuh"
#include "poigmres.cuh"

int main() {
    dim3 size(N, 1, 1);
    dim3 origin(0, 0, 0);
    Dom dom(size, origin);
    Mesh mesh(dom);

    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, 0);
    printf("%d %d %d %d\n", dprop.multiProcessorCount, dprop.maxThreadsPerBlock, dprop.maxThreadsPerMultiProcessor, dprop.maxBlocksPerMultiProcessor);

    double dx = L / N;

    for (int idx = 0; idx < N; idx ++) {
        mesh.x(idx, 0) = (idx + 0.5) * dx;
        mesh.x(idx, 1) = 0;
        mesh.x(idx, 2) = 0;
        mesh.h(idx, 0) = dx;
        mesh.h(idx, 1) = 1e-2;
        mesh.h(idx, 2) = 1e-2;
        mesh.v(idx   ) = dx * 1e-4; 
    }

    mesh.sync_h2d();

    Matrix<double> a(size, 3, LOCATION::BOTH);
    Matrix<double> t(size, 1, LOCATION::BOTH);
    Matrix<double> b(size, 1, LOCATION::BOTH);
    Matrix<double> r(size, 1, LOCATION::BOTH);

    prepare_poisson_eq(a, b, mesh, dom);

    LS_State ls_state;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // poisson_pbicgstab(a, t, b, r, 1e-9, 100000, dom, ls_state);
    poisson_fgmres(a, t, b, r, 1e-9, 1e-9, 10000, 10, dom, ls_state);
    cudaEventRecord(stop);

    a.sync_d2h();
    t.sync_d2h();
    b.sync_d2h();

    /* for (int idx = 0; idx < N; idx ++) {
        printf("%12.5e %12.5e %12.5e, %12.5e\n", a(idx, 1), a(idx, 0), a(idx, 2), b(idx));
    }
    printf("---------------------------------------\n");
    for (int idx = 0; idx < N; idx ++) {
        printf("%12.5e\n", t(idx));
    }
    printf("---------------------------------------\n");
    printf("%12.5e, %d\n", ls_state.re, ls_state.it); */

    float wall_time;
    cudaEventElapsedTime(&wall_time, start, stop);
    printf("%f\n", wall_time / 1000.0);

    mesh.release(LOCATION::BOTH);
    a.release(LOCATION::BOTH);
    t.release(LOCATION::BOTH);
    b.release(LOCATION::BOTH);

    return 0;
}