#include "snn/optimizer.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

    tensor<real> t({2, 3});
    tensor<real> g({2, 3}, {-9, -1, 0, 0.5, 1, 9});

    optimizer sgd;
    sgd.create(t, "sgd(learning_rate=0.1)");
    sgd.update(t, g);
	cout << t << '\n';
}
