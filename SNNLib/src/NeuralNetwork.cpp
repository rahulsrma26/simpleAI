#include "stdafx.h"
#include "NeuralNetwork.h"

namespace simpleNN {

    NeuralNetwork::NeuralNetwork(size_t input, LossFunction _cost)
        : inputNeurons(input), cost(_cost)
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

    void NeuralNetwork::backPropogate(const realVector& output, const realVector& label) {
        auto delta = cost.second(output, label);
        for (size_t i = layers.size(); i; --i)
            delta = layers[i - 1]->backward(delta);
    }

    void NeuralNetwork::update(real eta) {
        for (auto& layer : layers)
            layer->update(eta);
    }

    void NeuralNetwork::train(const realMatrix& input, const realMatrix& label,
        size_t batchSize, real learningRate) 
    {
        const size_t samples = input.size();
        for (size_t i = 0; i < samples; ) {
            for (size_t j = 0; j < batchSize; ++j, ++i) {
                auto output = feedForward(input[i]);
                backPropogate(output, label[i]);
            }
            update(learningRate);
        }
    }

    inline uint32_t getClass(const realVector& data) {
        return std::max_element(data.begin(), data.end()) - data.begin();
    }

    nnStats NeuralNetwork::getStats(const realMatrix& input, const realMatrix& label){
        real loss = 0.0;
        uint32_t correct = 0;

        const size_t samples = input.size();
        for (size_t i = 0; i < samples; ++i) {
            auto output = feedForward(input[i]);
            loss += cost.first(output, label[i]);
            if (getClass(output) == getClass(label[i]))
                ++correct;
        }
        return { loss / samples, (real)correct / samples };
    }
}