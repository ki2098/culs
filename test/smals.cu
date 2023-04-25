#include <iostream>
#include <cstring>
#include "../src/smals.h"

using namespace std;

int main() {
    /* double _a[4][4] = {
        { 3,  4, -2, 2},
        { 4,  9, -3, 5},
        {-2, -3,  7, 6},
        { 1,  4,  6, 7}
    }; */
    double _a[16] = {3, 4, -2, 1, 4, 9, -3, 4, -2, -3, 7, 6, 2, 5, 6, 7};
    double _b[4] = {2, 8, 10, 2};
    Matrix<double> a(4, 4, LOCATION::HOST, 1);
    Matrix<double> b(4, 1, LOCATION::HOST, 0);
    Matrix<double> x(4, 1, LOCATION::HOST, 0);
    memcpy(a._hh._arr, _a, sizeof(double) * 16);
    memcpy(b._hh._arr, _b, sizeof(double) * 4);
    SMALS::solve(a, x, b);
    for (int i = 0; i < a._row; i ++) {
        for (int j = 0; j < a._col; j ++) {
            printf("%10.6lf ", a(i, j));
        }
        printf(" , ");
        printf("%10.6lf ", x(i));
        printf(" , ");
        printf("%10.6lf ", b(i));
        printf("\n");
    }
    printf("\n");

    return 0;
}
