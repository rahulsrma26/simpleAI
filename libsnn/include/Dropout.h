#ifndef __NEURAL_NETWORK_DROPOUT_H_
#define __NEURAL_NETWORK_DROPOUT_H_

#include "nntypes.h"
#include "NNLayer.h"
#include "activators.h"

namespace simpleNN {

    class Dropout : public NNLayer
    {
        real probability;
        realVector active;
        std::uniform_real_distribution<real> uniform;

    public:
        Dropout(real dropoutProbability);

        virtual void initialize(int numInputConnections) override;
        virtual realVector getInput() const override;
        virtual realVector predict(const realVector& previousActivation) override;
        virtual realVector forward(const realVector& previousActivation) override;
        virtual realVector backward(const realVector& activationDelta) override;
        virtual void update(real eta) override;
    };

}
#endif // !__NEURAL_NETWORK_DROPOUT_H_
