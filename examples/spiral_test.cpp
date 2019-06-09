#include "snn/model/sequential.hpp"
#include "snn/dataset/spiral.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

    seed(12345);
    models::sequential m;
    m.add("dense(units=8, activation=tanh(), input=2)");
    m.add("dense(units=8, activation=tanh())");
    m.add("dense(units=5, activation=tanh())");
    m.add("dense(units=1)");
    m.compile("loss=cross_entropy(), optimizer=sgd(learning_rate=0.03)");
    m.summary();

    auto [trainX, trainY, testX, testY] = dataset::spiral::generate(8192, false);

	const int epochs = 256;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "Epoch: " << epoch << '/' << epochs << endl;
        m.run(trainX, trainY, "batch_size=64");
        m.run(testX, testY, "batch_size=64, train=false");
    }
}
