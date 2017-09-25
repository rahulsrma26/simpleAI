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

    NeuralNetwork nn(784);
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
+-------+---------------------+---------------------+
|       |        train        |         test        |
| sweep |     loss | accuracy |     loss | accuracy |
|-------+----------+----------+----------+----------|
|     1 | 0.302699 | 0.486524 | 0.307688 | 0.486500 |
|     2 | 0.286805 | 0.497405 | 0.292384 | 0.497611 |
|     3 | 0.279060 | 0.503167 | 0.284921 | 0.504778 |
|     4 | 0.273935 | 0.503881 | 0.281145 | 0.504833 |
|     5 | 0.270593 | 0.506833 | 0.279227 | 0.506333 |
|     6 | 0.244430 | 0.583905 | 0.252869 | 0.582000 |
|     7 | 0.229154 | 0.595762 | 0.238951 | 0.593500 |
|     8 | 0.224517 | 0.597405 | 0.235333 | 0.597222 |
|     9 | 0.216566 | 0.626310 | 0.229264 | 0.622667 |
|    10 | 0.197928 | 0.656024 | 0.213434 | 0.648556 |
|    11 | 0.191517 | 0.667310 | 0.207790 | 0.659444 |
|    12 | 0.170824 | 0.727262 | 0.186624 | 0.716111 |
|    13 | 0.153646 | 0.749595 | 0.171308 | 0.735778 |
|    14 | 0.147386 | 0.754262 | 0.165549 | 0.741778 |
|    15 | 0.142308 | 0.762476 | 0.160505 | 0.748111 |
|    16 | 0.127636 | 0.816619 | 0.145487 | 0.801722 |
|    17 | 0.103750 | 0.851262 | 0.122662 | 0.837778 |
|    18 | 0.061740 | 0.930714 | 0.081606 | 0.914222 |
|    19 | 0.055883 | 0.937595 | 0.076758 | 0.918111 |
|    20 | 0.050195 | 0.941214 | 0.072185 | 0.924333 |
|    21 | 0.046290 | 0.945429 | 0.070026 | 0.925167 |
|    22 | 0.043803 | 0.947881 | 0.068211 | 0.928222 |
|    23 | 0.041216 | 0.950429 | 0.066420 | 0.928500 |
|    24 | 0.039106 | 0.952214 | 0.065912 | 0.929444 |
|    25 | 0.038249 | 0.953500 | 0.065358 | 0.930500 |
|    26 | 0.037916 | 0.954190 | 0.065054 | 0.930944 |
|    27 | 0.035445 | 0.956190 | 0.063849 | 0.931778 |
|    28 | 0.034029 | 0.957262 | 0.063640 | 0.933278 |
|    29 | 0.032927 | 0.958524 | 0.062644 | 0.932111 |
|    30 | 0.031712 | 0.958929 | 0.062650 | 0.933333 |
*/