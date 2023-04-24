#ifndef _MESH_H_
#define _MESH_H_ 1

#include "matrix.cuh"
#include "dom.cuh"

struct Mesh {
    Matrix<double> x;
    Matrix<double> h;
    Matrix<double> v;

    Mesh(Dom &dom);
    void sync_h2d();
    void sync_d2h();
    void release(int loc);
};

Mesh::Mesh(Dom &dom) : x(dom._size, 3, LOCATION::HOST), h(dom._size, 3, LOCATION::HOST), v(dom._size, 1, LOCATION::HOST) {}

void Mesh::sync_d2h() {
    x.sync_d2h();
    h.sync_d2h();
    v.sync_d2h();
}

void Mesh::sync_h2d() {
    x.sync_h2d();
    h.sync_h2d();
    v.sync_h2d();
}

void Mesh::release(int loc) {
    x.release(loc);
    h.release(loc);
    v.release(loc);
}

#endif