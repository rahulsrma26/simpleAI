#include "snn/activator.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

    tensor<real> a({2, 3}, {-9, -1, 0, 0.5, 1, 9});

    activator as;
	as.create("sigmoid()");
    cout << as.f(a) << '\n';
    cout << as.df(a) << '\n';

    activator ar;
    ar.create("relu()");
    cout << ar.f(a) << '\n';
    cout << ar.df(a) << '\n';
}
