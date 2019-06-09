#include "snn/layer.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

    layer d;
    d.create("dense(input=784, units=100, use_bias=true, kernel_initializer=xavier(), bias_initializer=zeros(), activation=sigmoid())");
    tensor<real> input({1, 784});
    input.fill(1);
    auto output = d.forward(input);
    cout << output << '\n';
}
