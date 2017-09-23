// SNNLib.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "mnistDataSet.h"
#include "simpleNN.h"

int main()
{
    using namespace std;

    dataset::MnistDataSet data;
    data.Load(".\\data\\train-images.idx3-ubyte", ".\\data\\train-labels.idx1-ubyte");
    for (int i = 0; i < 1; i++)
        cout << data.show(i) << '\n';

    using namespace simpleNN;
    NeuralNetwork nn(2);
    nn.add(make_unique<DenseLayer>(3));
    nn.add(make_unique<DenseLayer>(1));

    cout << nn.numLayers() << " layers NN created."
        " input: " << nn.inputSize() << ", output: " << nn.outputSize() << '\n';

    cout << nn.feedForward({ 1,0 }) << '\n';

    return 0;
}

