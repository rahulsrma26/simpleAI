/*
This is an implementation of spiral dataset example. You can check tensorflow implementation running live from tensorflow playground site.
https://playground.tensorflow.org/#activation=tanh&batchSize=30&dataset=spiral&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=0&networkShape=8,8,5,2&seed=0.48787&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

After ruuning the binary see spiral.pgm for results.
*/

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
