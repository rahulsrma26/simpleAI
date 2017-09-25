// SNNLib.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "mnistDataSet.h"
#include "simpleNN.h"
#include <cstdio>
using namespace std;
using namespace simpleNN;

auto spiralDataset(size_t sample = 500, real noise = 0, bool addSin = false) {
    real pi = 3.14159265358979;
    size_t n = sample / 2;
    realMatrix xdata, ydata;

    auto gen = [&](real deltaT, real label) {
        for (size_t i = 0; i < n; ++i) {
            real r = 5.0 * i / n;
            real t = 1.75 * i / n * 2.0 * pi + deltaT;
            real x = r * sin(t);
            real y = r * cos(t);
            if (addSin)
                xdata.push_back({ x, y, sin(x), cos(y) });
            else
                xdata.push_back({ x, y });
            ydata.push_back({ label });
        }
    };

    gen(0, 1.0);
    gen(pi, 0.0);
    return make_pair(xdata, ydata);
}

void spiralTest() {
    NeuralNetwork nn(2);
    nn.add(make_unique<DenseLayer>(8));
    nn.add(make_unique<DenseLayer>(8));
    nn.add(make_unique<DenseLayer>(5));
    nn.add(make_unique<DenseLayer>(1, activators::sigmoid));

    cout << nn.numLayers() << " layers NN created."
        " input: " << nn.inputSize() << ", output: " << nn.outputSize() << '\n';

    auto[xdata, ydata] = spiralDataset(500, 0.0, false);

    for (int i = 0; i < 100000; ++i) {
        nn.train(xdata, ydata, 10, .03);
        if (i % 100 == 0)
            cout << "iteration: " << i + 1
            << ", training loss: " << nn.getStats(xdata, ydata).loss << '\n';
    }
}

auto mnistDataset(float trainingSize) {
    dataset::MnistDataSet data;
    data.Load(".\\data\\train-images.idx3-ubyte", ".\\data\\train-labels.idx1-ubyte");

    realMatrix trainX, trainY;
    realMatrix testX, testY;
    {
        auto[xdata, ydata] = data.getData();
        const size_t samples = xdata.size();
        const size_t splitIndex = samples * trainingSize;

        trainX.reserve(splitIndex);
        move(xdata.begin(), xdata.begin() + splitIndex, back_inserter(trainX));
        trainY.reserve(splitIndex);
        move(ydata.begin(), ydata.begin() + splitIndex, back_inserter(trainY));

        testX.reserve(samples - splitIndex);
        move(xdata.begin() + splitIndex, xdata.end(), back_inserter(testX));
        testY.reserve(samples - splitIndex);
        move(ydata.begin() + splitIndex, ydata.end(), back_inserter(testY));
    }
    return make_tuple(trainX, trainY, testX, testY);
}

int main()
{
    using namespace std;
    using namespace simpleNN;

    NeuralNetwork nn(784, loss::crossEntropy);
    nn.add(make_unique<DenseLayer>(100));
    //nn.add(make_unique<DenseLayer>(20));
    nn.add(make_unique<DenseLayer>(10, activators::sigmoid));

    auto[trainX, trainY, testX, testY] = mnistDataset(.7);

    printf("+-------+---------------------+---------------------+\n");
    printf("|       |        train        |         test        |\n");
    printf("| sweep |     loss | accuracy |     loss | accuracy |\n");
    printf("|-------+----------+----------+----------+----------|\n");

    for (int i = 1; i <= 30; ++i) {
        nn.train(trainX, trainY, 10, 0.5);
        printf("| %5d |", i);
        auto stats = nn.getStats(trainX, trainY);
        printf(" %0.6f | %0.6f |", stats.loss, stats.accuracy);
        stats = nn.getStats(testX, testY);
        printf(" %0.6f | %0.6f |\n", stats.loss, stats.accuracy);
    }

    return 0;
}

/*

Performance
+---------------+------+-------+---------------------+---------------------+
|               |      |       |        train        |         test        |
|               |      |       |---------------------+---------------------|
|          type |  eta | sweep |     loss | accuracy |     loss | accuracy |
|---------------+------+-------+----------+----------+----------+----------|
|     quadratic |  0.5 |     1 | 0.302699 | 0.486524 | 0.307688 | 0.486500 |
|     quadratic |  0.5 |    30 | 0.031712 | 0.958929 | 0.062650 | 0.933333 |
|     hillinger |  0.5 |     1 | 0.368739 | 0.584952 | 0.373407 | 0.583333 |
|     hillinger |  0.5 |    30 | 0.032595 | 0.969476 | 0.076785 | 0.940667 |
| cross-entropy |  0.5 |     1 | 3.029185 | 0.905333 | 3.140360 | 0.902222 |
| cross-entropy |  0.5 |    30 | 0.651235 | 0.982929 | 1.135570 | 0.961389 |
+---------------+------+-------+---------------------+---------------------+

*/