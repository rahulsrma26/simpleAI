#include "snn/snn.hpp"
#include "snn/dataset/spiral.hpp"
#include "snn/util/image.hpp"
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
    m.compile("loss=cross_entropy(), optimizer=adam(learning_rate=0.001)");
    m.summary();

    auto [trainX, trainY, testX, testY] = dataset::spiral::generate(8192, false);

    const int epochs = 128;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "Epoch: " << epoch << '/' << epochs << endl;
        m.run(trainX, trainY, "batch_size=32");
        m.run(testX, testY, "batch_size=256, train=false");
    }

    int radius = 256;
    auto image = m.predict(dataset::spiral::generate_grid(radius));
    image.reshape({radius * 2 + 1, radius * 2 + 1});
    save_pgm("spiral.pgm", image);
}
