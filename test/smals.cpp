#include <iostream>
#include "../src/smals.h"

using namespace std;

int main() {
    vector<vector<double>> a {
        { 3,  4, -2, 2},
        { 4,  9, -3, 5},
        {-2, -3,  7, 6},
        { 1,  4,  6, 7}
    };
    vector<double> b = {2, 8, 10, 2};
    vector<double> x(4);
    SMALS::solve(a, x, b);
    for (int i = 0; i < x.size(); i ++) {
        printf("%lf ", x[i]);
    }
    printf("\n");

    return 0;
}