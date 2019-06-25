#include "snn/math/tmath.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

    tensor<int> b({2, 2, 2});
    int i = 0;
    b.initialize([&]() { return 1 + (i++) % 9; });
    cout << b << endl;

    auto a = math::pad(b, {{1, 1}, {1, 1}, {1, 1}});
    cout << a << endl;
}
