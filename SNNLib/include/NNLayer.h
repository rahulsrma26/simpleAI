#ifndef __NEURAL_NETWORK_LAYER_H__
#define __NEURAL_NETWORK_LAYER_H__

#include "nntypes.h"

namespace simpleNN {
    class NNLayer {
    protected:
        uint numNeurons;
        uint inputConnections;

    public:
        NNLayer(uint _numNeurons) :
            numNeurons(_numNeurons) {}

        /*NNLayer(const NNLayer&) = default;
        NNLayer(NNLayer&&) = default;
        NNLayer& operator=(const NNLayer&) = default;
        NNLayer& operator=(NNLayer&&) = default;*/

        uint size() const {
            return numNeurons;
        }

        virtual void initialize(uint numInputConnections) {
            inputConnections = numInputConnections;
        }

        virtual realVector predict(const realVector& input) {
            return input;
        }

        virtual realVector forward(const realVector& input) { 
            return input; 
        }

        virtual realVector backward(const realVector& weightsTransposeDelta) {
            return weightsTransposeDelta;
        }

        virtual realVector getInput() const {
            return realVector(numNeurons);
        }

        virtual void update(real eta) {}
    };
}

#endif // !__NEURAL_NETWORK_LAYER_H__
