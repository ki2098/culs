#ifndef _SMALS_H_
#define _SMALS_H_ 1

#include <cstdio>
#include <assert.h>
#include "matrix.cuh"

namespace SMALS {

static double det(Matrix<double> &a) {
    if (a._row == 2) {
        return a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
    }
    double _det = 0;
    for (int j = 0; j < a._col; j ++) {
        // std::vector<std::vector<double>> sub(a.size() - 1, std::vector<double>(a.size() - 1));
        Matrix<double> sub(a._row - 1, a._col - 1, LOCATION::HOST, 0);
        for (int ix = 1; ix < a._row; ix ++) {
            for (int jx = 0; jx < a._col; jx ++) {
                if (jx == j) {
                    continue;
                }
                int it = ix - 1;
                int jt = (jx < j)? jx : jx - 1;
                sub(it, jt) = a(ix, jx);
            }
        }
        int sig = (j % 2 == 0)? 1 : - 1;
        _det += sig * a(0, j) * det(sub);
    }
    return _det;
}

static void solve(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b) {
    assert(a._row == a._col && a._col == x._row && a._col == b._row && x._col == 1 && b._col == 1);
    double _adet = det(a);
    if (_adet == 0) {
        printf("singularity!\n");
    }
    
    for (int j = 0; j < a._col; j ++) {
        Matrix<double> ai(a._row, a._col, LOCATION::HOST, 0);
        MatrixUtil::assign(ai, a, LOCATION::HOST);
        for (int i = 0; i < a._row; i ++) {
            ai(i, j) = b(i);
        }
        x(j) = det(ai) / _adet;
    }
}

}

#endif