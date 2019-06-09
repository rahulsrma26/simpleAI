#include "snn/initializer.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

    tensor<real> a({3, 5});

    initializer xa;
    xa.create("xavier()");
    xa.init(a);
    cout << a << '\n';

    initializer no;
    no.create("normal(mean=1)");
    no.init(a);
    cout << a << '\n';
}
