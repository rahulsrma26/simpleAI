#include "snn/model/sequential.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

    models::sequential m;
    m.add("dense(input=2, units=4)");
    m.add("dense(units=1)");
    m.compile("loss=cross_entropy(), optimizer=sgd(learning_rate=0.1)");

    // create a dataset
    const shapeType data_size = 128;
    tensor<real> dx({data_size, 2});
    tensor<real> dy({data_size, 1});

    for (size_t i = 0; i < 128; i++) {
        int a = rand() & 1, b = rand() & 1;
        dx[2 * i] = a;
        dx[2 * i + 1] = b;
        dy[i] = a ^ b;
    }

    // create a test dataset
    tensor<real> tx({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    tensor<real> ty({4, 1}, {0, 1, 1, 0});

    for (int epoch = 0; epoch < 128; epoch++) {
        m.run(dx, dy, "batch_size=4");
        m.run(tx, ty, "batch_size=4, train=false");
    }
    cout << "XOR test \n";
    cout << "test input = " << tx << '\n';
    cout << "test label = " << ty.squeeze() << '\n';
    cout << "test output = " << m.predict(tx).squeeze() << '\n';
}
