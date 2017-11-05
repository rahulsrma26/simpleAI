/*
SNNLib.cpp : Defines the entry point for the console application.
author: Rahul Sharma
*/

#include "stdafx.h"
#include "mnistDataSet.h"
#include "spiralDataSet.h"
#include "simpleNN.h"
#include <cstdio>
#include <chrono>

// disable to run spiral test
#define MNIST_TEST_RUN

int main()
{
    using namespace std;
    using namespace chrono;
    using namespace simpleNN;

#ifdef MNIST_TEST_RUN
    constexpr size_t SWEEPS = 30;
    constexpr size_t BATCH_SIZE = 10;
    real ETA = 0.5f;

    // 97.48% accuracy
    NeuralNetwork nn(784, loss::crossEntropy);
    nn.add(make_unique<DenseLayer>(300));
    nn.add(make_unique<Dropout>(0.5f));
    nn.add(make_unique<DenseLayer>(10));
    auto[trainX, trainY] = dataset::MnistDataSet::get(
        ".\\data\\train-images.idx3-ubyte", ".\\data\\train-labels.idx1-ubyte");
    auto[testX, testY] = dataset::MnistDataSet::get(
        ".\\data\\t10k-images.idx3-ubyte", ".\\data\\t10k-labels.idx1-ubyte");

#else // SPIRAL_TEST_RUN
    constexpr size_t SWEEPS = 500;
    constexpr size_t BATCH_SIZE = 10;
    real ETA = 0.03f;

    NeuralNetwork nn(2, loss::crossEntropy);
    nn.add(make_unique<DenseLayer>(8, activators::tanh));
    nn.add(make_unique<DenseLayer>(8, activators::tanh));
    nn.add(make_unique<DenseLayer>(5, activators::tanh));
    nn.add(make_unique<DenseLayer>(1, activators::sigmoid));
    auto[trainX, trainY, testX, testY] = dataset::spiralDataset<real>(1000, false);
#endif

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
        //ETA *= .97f;
        double time = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() / 1e9;
        printf(" %10.5f |\n", time);
    }

#ifndef MNIST_TEST_RUN
    int radius = 100;
    dataset::saveGrid<real>("out.pgm", nn.predict(dataset::spiralGrid<real>(radius)), 2 * radius + 1);
#endif

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
|  ce + dropout |  0.5 |     1 | 2.742396 | 0.917317 | 2.643539 | 0.918700 |
|  ce + dropout |  0.5 |    30 | 0.883343 | 0.976600 | 1.094968 | 0.963700 |
+---------------+------+-------+---------------------+---------------------+

1 iteration:
Without OpenMP = 18.4469 seconds.
   With OpenMP = 5.85805 seconds.
*/