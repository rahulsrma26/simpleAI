#ifndef __NEURAL_NETWORK_DENSE_LAYER_H_
#define __NEURAL_NETWORK_DENSE_LAYER_H_

#include "nntypes.h"
#include "NNLayer.h"
#include "activators.h"

namespace simpleNN {

    class DenseLayer : public NNLayer
    {
        Activator activator;
        size_t inputConnections;
        realVector bias;
        realMatrix weights;
        realVector input;
        realVector weightedInput;
        realVector deltaB;
        realMatrix deltaW;
        size_t deltaN;

    public:
        DenseLayer(size_t numNeurons, Activator activator=activators::tanh);

        /*DenseLayer(const DenseLayer&) = default;
        DenseLayer(DenseLayer&&) = default;
        DenseLayer& operator=(const DenseLayer&) = default;
        DenseLayer& operator=(DenseLayer&&) = default;*/

        virtual void initialize(size_t numInputConnections) override;
        virtual realVector forward(const realVector& previousActivation) override;
        virtual realVector backward(const realVector& weightsTransposeDelta) override;
        virtual void update(real eta) override;
    };

}
#endif // !__NEURAL_NETWORK_DENSE_LAYER_H_



