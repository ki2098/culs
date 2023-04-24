#ifndef _SMALS_H_
#define _SMALS_H_ 1

#include <cstdio>
#include <vector>

namespace SMALS {

static double det(std::vector<std::vector<double>> &a) {
    if (a.size() == 2) {
        return a[0][0] * a[1][1] - a[0][1] * a[1][0];
    }
    double _det = 0;
    for (int j = 0; j < a.size(); j ++) {
        std::vector<std::vector<double>> sub(a.size() - 1, std::vector<double>(a.size() - 1));
        for (int ix = 1; ix < a.size(); ix ++) {
            for (int jx = 0; jx < a.size(); jx ++) {
                if (jx == j) {
                    continue;
                }
                int it = ix - 1;
                int jt = (jx < j)? jx : jx - 1;
                sub[it][jt] = a[ix][jx];
            }
        }
        int sig = (j % 2 == 0)? 1 : - 1;
        _det += sig * a[0][j] * det(sub);
    }
    return _det;
}

static void solve(std::vector<std::vector<double>> &a, std::vector<double> &x, std::vector<double> &b) {
    double _adet = det(a);
    if (_adet == 0) {
        printf("singularity!\n");
    }
    
    for (int j = 0; j < a.size(); j ++) {
        std::vector<std::vector<double>> ai(a.size(), std::vector<double>(a.size()));
        ai = a;
        for (int i = 0; i < a.size(); i ++) {
            ai[i][j] = b[i];
        }
        x[j] = det(ai) / _adet;
    }
}

}

#endif