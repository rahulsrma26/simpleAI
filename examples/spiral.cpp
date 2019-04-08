#include "spiralDataSet.h"
#include "simpleNN.h"
#include <cstdio>
#include <chrono>

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace chrono;
    using namespace simpleNN;

    constexpr int SWEEPS = 500;
    constexpr size_t BATCH_SIZE = 10;
    real ETA = 0.03f;

    NeuralNetwork nn(2, loss::crossEntropy);
    nn.add(make_unique<DenseLayer>(8, activators::tanh));
    nn.add(make_unique<DenseLayer>(8, activators::tanh));
    nn.add(make_unique<DenseLayer>(5, activators::tanh));
    nn.add(make_unique<DenseLayer>(1, activators::sigmoid));
    auto [trainX, trainY, testX, testY] = dataset::spiralDataset<real>(5000, false);

    printf("+-------+---------------------+---------------------+------------+\n");
    printf("|       |        train        |         test        |            |\n");
    printf("| sweep |     loss | accuracy |     loss | accuracy | time (sec) |\n");
    printf("|-------+----------+----------+----------+----------+------------|\n");

    for (int i = 1; i <= SWEEPS; ++i) {
        auto start = high_resolution_clock::now();
        nn.train(trainX, trainY, BATCH_SIZE, ETA);
        printf("| %5d |", i);
        auto stats = nn.getStats(trainX, trainY);
        printf(" %8.5f | %0.6f |", stats.loss, stats.accuracy);
        stats = nn.getStats(testX, testY);
        printf(" %8.5f | %0.6f |", stats.loss, stats.accuracy);
        // ETA *= .99f;
        double time =
            duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() / 1e9;
        printf(" %10.5f |\n", time);
    }

    int radius = 100;
    dataset::saveGrid<real>("out.pgm", nn.predict(dataset::spiralGrid<real>(radius)),
                            2 * radius + 1);
    return 0;
}
