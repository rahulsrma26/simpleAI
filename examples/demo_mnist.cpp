#include "snn/model/sequential.hpp"
#include "snn/dataset/mnist.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace snn;

    if (argc < 2) {
        cout << "Usages:" << '\n';
        cout << argv[0] << " <mnist-dir>" << '\n';
        return 0;
    }

    auto data_dir = string(argv[1]);
    auto [trainX, trainY] = dataset::mnist::load(data_dir + "/train");
    auto [testX, testY] = dataset::mnist::load(data_dir + "/t10k");
    // flatten 2D images to 1D
    trainX.reshape({0, 784});
    testX.reshape({0, 784});

    models::sequential m;
    m.add("dense(units=300, input=784)");
    m.add("dense(units=10)");
    m.compile("loss=cross_entropy(), optimizer=sgd(learning_rate=0.5)");
    m.summary();

    const int epochs = 20;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "Epoch: " << epoch << '/' << epochs << endl;
        m.run(trainX, trainY, "batch_size=32");
        m.run(testX, testY, "batch_size=128, train=false");
    }
}
