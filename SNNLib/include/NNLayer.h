#ifndef __NEURAL_NETWORK_LAYER_H__
#define __NEURAL_NETWORK_LAYER_H__

#include "nntypes.h"

namespace simpleNN {
    class NNLayer {
    protected:
        size_t numNeurons;

    public:
        NNLayer(size_t _numNeurons) : numNeurons(_numNeurons) {}

        /*NNLayer(const NNLayer&) = default;
        NNLayer(NNLayer&&) = default;
        NNLayer& operator=(const NNLayer&) = default;
        NNLayer& operator=(NNLayer&&) = default;*/

        virtual void initialize(size_t inputs) {}

        virtual realVector forward(const realVector& input) { 
            return input; 
        }

        virtual realVector backward(const realVector& weightsTransposeDelta) {
            return weightsTransposeDelta;
        }

        virtual void update(real eta) {}

        size_t size() const {
            return numNeurons;
        }
    };
}

#endif // !__NEURAL_NETWORK_LAYER_H__
