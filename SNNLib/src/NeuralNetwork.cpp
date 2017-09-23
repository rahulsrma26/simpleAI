#include "stdafx.h"
#include "NeuralNetwork.h"

namespace simpleNN {

    NeuralNetwork::NeuralNetwork(size_t input) 
        : inputNeurons(input)
    {
    }

    size_t NeuralNetwork::inputSize() const {
        return inputNeurons;
    }

    size_t NeuralNetwork::numLayers() const {
        return layers.size() + 1;
    }

    size_t NeuralNetwork::outputSize() const {
        return layers.size()? layers.back()->size(): -1;
    }

    void NeuralNetwork::add(std::unique_ptr<NNLayer> layer) {
        layer->initialize(layers.size()? layers.back()->size(): inputNeurons);
        layers.push_back(std::move(layer));
    }

    realVector NeuralNetwork::feedForward(const realVector& input) {
        auto output = input;
        for (auto& layer : layers)
            output = layer->forward(output);
        return output;
    }

}