#include "snn/variable.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

	variable v;
    v.create("shape=(2,3), initializers=normal(mean=1.0)");
	cout << v.var << '\n';

	v.set_optimizer("sgd()");

	tensor<real> g({2, 3}, {-9, -1, 0, 0.5, 1, 9});
	v.optimize(g);
    cout << v.var << '\n';
}
