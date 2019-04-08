#ifndef __NEURAL_NETWORK_DENSE_LAYER_H_
#define __NEURAL_NETWORK_DENSE_LAYER_H_

#include "nntypes.h"
#include "NNLayer.h"
#include "activators.h"

namespace simpleNN {

    class DenseLayer : public NNLayer
    {
        Activator activator;
        realVector bias;
        realMatrix weights;
        realVector input;
        realVector weightedInput;
        realVector deltaB;
        realMatrix deltaW;
        int deltaN;

    public:
        DenseLayer(int numNeurons, Activator activator=activators::sigmoid);

        /*DenseLayer(const DenseLayer&) = default;
        DenseLayer(DenseLayer&&) = default;
        DenseLayer& operator=(const DenseLayer&) = default;
        DenseLayer& operator=(DenseLayer&&) = default;*/

        virtual void initialize(int numInputConnections) override;
        virtual realVector getInput() const override;
        virtual realVector predict(const realVector& previousActivation) override;
        virtual realVector forward(const realVector& previousActivation) override;
        virtual realVector backward(const realVector& activationDelta) override;
        virtual void update(real eta) override;
    };

}
#endif // !__NEURAL_NETWORK_DENSE_LAYER_H_



