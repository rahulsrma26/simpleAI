#include "stdafx.h"
#include "NeuralNetwork.h"

namespace simpleNN {

    NeuralNetwork::NeuralNetwork(uint input, LossFunction _cost)
        : inputNeurons(input), cost(_cost)
    {
    }

    uint NeuralNetwork::inputSize() const {
        return inputNeurons;
    }

    size_t NeuralNetwork::numLayers() const {
        return layers.size() + 1;
    }

    uint NeuralNetwork::outputSize() const {
        return layers.size()? layers.back()->size(): -1;
    }

    void NeuralNetwork::add(std::unique_ptr<NNLayer> layer) {
        layer->initialize(layers.size()? layers.back()->size(): inputNeurons);
        layers.push_back(std::move(layer));
    }

    realVector NeuralNetwork::predict(const realVector& input) {
        auto output = input;
        for (auto& layer : layers)
            output = layer->predict(output);
        return output;
    }

    realMatrix NeuralNetwork::predict(const realMatrix& inputs) {
        realMatrix outputs;
        for (auto& input : inputs)
            outputs.push_back(predict(input));
        return outputs;
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
        uint batchSize, real learningRate) 
    {
        const size_t samples = input.size();

        uintVector indices(samples);
        for (uint i = 0; i < samples; ++i)
            indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), nnRandomEngine);
        
        for (uint i = 0; i < samples; ) {
            for (uint j = 0; j < batchSize; ++j, ++i) {
                uint idx = indices[i];
                auto output = feedForward(input[idx]);
                backPropogate(output, label[idx]);
            }
            update(learningRate);
        }
    }

    inline size_t getClass(const realVector& data) {
        if (data.size() == 1)
            return data.front() < .5 ? 0 : 1;
        return std::max_element(data.begin(), data.end()) - data.begin();
    }

    nnStats NeuralNetwork::getStats(const realMatrix& input, const realMatrix& label){
        real loss = 0.0;
        size_t correct = 0;

        const size_t samples = input.size();
        for (size_t i = 0; i < samples; ++i) {
            auto output = predict(input[i]);
            loss += cost.first(output, label[i]);
            if (getClass(output) == getClass(label[i]))
                ++correct;
        }
        return { (real)loss / samples, (real)correct / samples };
    }
}