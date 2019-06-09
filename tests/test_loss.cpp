#include "snn/loss.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

    tensor<real> a({4, 1}, {0, 0, 1, 1});
    tensor<real> b({4, 1}, {0, 1, 0, 1});
    
	loss lq;
    lq.create("quadratic()");
    cout << lq.f(a, b) << '\n';
    cout << lq.df(a, b) << '\n';
    
	loss lce;
    lce.create("cross_entropy()");
    cout << lce.f(a, b) << '\n';
    cout << lce.df(a, b) << '\n';
}
