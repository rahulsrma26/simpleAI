/*
SNNLib.cpp : Defines the entry point for the console application.
author: Rahul Sharma
*/

#include "stdafx.h"
#include "mnistDataSet.h"
#include "simpleNN.h"
#include <cstdio>
#include <chrono>
using namespace std;
using namespace simpleNN;

auto spiralDataset(size_t samples = 1000, bool addSin = false) {
    constexpr real pi = 3.14159265358979f;
    realMatrix trainX, trainY, testX, testY;

    auto gen = [&](real deltaT, const realVector& label, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            real r = 5.0f * i / n;
            real t = 1.75f * i / n * 2.0f * pi + deltaT;
            real x = r * sin(t);
            real y = r * cos(t);
            realMatrix& xdata = i & 1 ? testX : trainX;
            realMatrix& ydata = i & 1 ? testY : trainY;
            if (addSin)
                xdata.push_back({ x, y, sin(x), cos(y) });
            else
                xdata.push_back({ x, y });
            ydata.push_back(label);
        }
    };

    gen(0, { 0 }, samples / 2);
    gen(pi, { 1 }, samples - samples / 2);
    return make_tuple(trainX, trainY, testX, testY);
}

auto mnistDataset() {
    auto[trainX, trainY] = dataset::MnistDataSet::get(".\\data\\train-images.idx3-ubyte", ".\\data\\train-labels.idx1-ubyte");
    auto[testX, testY] = dataset::MnistDataSet::get(".\\data\\t10k-images.idx3-ubyte", ".\\data\\t10k-labels.idx1-ubyte");
    return make_tuple(trainX, trainY, testX, testY);
}

int main()
{
    using namespace std;
    using namespace chrono;
    using namespace simpleNN;

    constexpr size_t SWEEPS = 1;
    constexpr size_t BATCH_SIZE = 10;
    real ETA = 0.5;

    /*NeuralNetwork nn(2, loss::crossEntropy);
    nn.add(make_unique<DenseLayer>(8));
    nn.add(make_unique<DenseLayer>(8));
    nn.add(make_unique<DenseLayer>(5));
    nn.add(make_unique<DenseLayer>(1, activators::sigmoid));
    auto[trainX, trainY, testX, testY] = spiralDataset(1000, false);*/

    NeuralNetwork nn(784, loss::crossEntropy);
    nn.add(make_unique<DenseLayer>(100, activators::sigmoid));
    //nn.add(make_unique<Dropout>(0.2f));
    nn.add(make_unique<DenseLayer>(10, activators::sigmoid));
    auto[trainX, trainY, testX, testY] = mnistDataset();

    printf("+-------+---------------------+---------------------+\n");
    printf("|       |        train        |         test        |\n");
    printf("| sweep |     loss | accuracy |     loss | accuracy |\n");
    printf("|-------+----------+----------+----------+----------|\n");

    auto start = high_resolution_clock::now();
    for (int i = 1; i <= SWEEPS; ++i) {
        nn.train(trainX, trainY, BATCH_SIZE, ETA);
        printf("| %5d |", i);
        auto stats = nn.getStats(trainX, trainY);
        printf(" %0.6f | %0.6f |", stats.loss, stats.accuracy);
        stats = nn.getStats(testX, testY);
        printf(" %0.6f | %0.6f |\n", stats.loss, stats.accuracy);
        ETA *= .96f;
    }

    cout << "Done in "
        << duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() / 1e9
        << " seconds." << endl;

    /*int n = 100;
    ofstream fout;
    fout.open("out.pgm");
    fout << "P2" << endl;
    fout << 2*n+1 << ' ' << 2*n+1 << endl;
    fout << 99 << endl;
    for (int y = -n; y <= n; ++y) {
        for (int x = -n; x <= n; ++x) {
            auto output = nn.feedForward({ 5.0f * x / n, 5.0f * y / n });
            fout << min(max(0, (int)(output[0] * 99)), 99) << ' ';
        }
        fout << endl;
    }
    fout.close();*/

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

*/