// SNNLib.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "mnistDataSet.h"
#include "simpleNN.h"
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

int main()
{
    using namespace std;

    dataset::MnistDataSet data;
    data.Load(".\\data\\train-images.idx3-ubyte", ".\\data\\train-labels.idx1-ubyte");

    using namespace simpleNN;
    NeuralNetwork nn(784);
    nn.add(make_unique<DenseLayer>(100));
    nn.add(make_unique<DenseLayer>(20));
    nn.add(make_unique<DenseLayer>(10, activators::sigmoid));

    cout << nn.numLayers() << " layers NN created."
        " input: " << nn.inputSize() << ", output: " << nn.outputSize() << '\n';

    real trainingSize = .7;
    auto [xdata, ydata] = data.getData();

    for (int i = 0; i < 100; ++i) {
        cout << "iteration: " << i + 1;
        nn.train(xdata, ydata, 100, .5);
        cout << ", ";
        auto stats = nn.getStats(xdata, ydata);
        cout << "loss: " << stats.loss << ", accuracy: " << stats.accuracy << '\n';
    }

    return 0;
}

/*
4 layers NN created. input: 784, output: 10
iteration: 1, loss: 0.207803, accuracy: 0.704983
iteration: 2, loss: 0.148474, accuracy: 0.80925
iteration: 3, loss: 0.110427, accuracy: 0.85995
iteration: 4, loss: 0.0942541, accuracy: 0.879833
iteration: 5, loss: 0.086462, accuracy: 0.89095
iteration: 6, loss: 0.0788267, accuracy: 0.900283
iteration: 7, loss: 0.0724331, accuracy: 0.908267
iteration: 8, loss: 0.067941, accuracy: 0.914983
iteration: 9, loss: 0.0649109, accuracy: 0.918517
iteration: 10, loss: 0.0601316, accuracy: 0.925367
iteration: 11, loss: 0.0574244, accuracy: 0.928417
iteration: 12, loss: 0.054275, accuracy: 0.932767
iteration: 13, loss: 0.0533889, accuracy: 0.93315
iteration: 14, loss: 0.04977, accuracy: 0.93825
iteration: 15, loss: 0.0484691, accuracy: 0.9398
iteration: 16, loss: 0.0476328, accuracy: 0.940517
iteration: 17, loss: 0.0454334, accuracy: 0.9439
iteration: 18, loss: 0.0440913, accuracy: 0.94625
iteration: 19, loss: 0.0423745, accuracy: 0.947817
iteration: 20, loss: 0.042015, accuracy: 0.947867
iteration: 21, loss: 0.039632, accuracy: 0.951617
*/