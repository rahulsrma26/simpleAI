#ifndef __NEURAL_NETWORK_TYPES_H__
#define __NEURAL_NETWORK_TYPES_H__

//#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <functional>

namespace simpleNN {
    using uint = uint32_t;
    using uintVector = std::vector<uint>;

    using real = float;
    using realVector = std::vector<real>;
    using realMatrix = std::vector<realVector>;
    using VectorProcessor = std::function<realVector(const realVector&)>;

    static std::default_random_engine nnRandomEngine; // (
        //eng.seed(chrono::system_clock::now().time_since_epoch().count()));
    static std::normal_distribution<real> Gaussian;

    template<class Ch, class Tr>
    std::basic_ostream<Ch, Tr>& 
        operator << (std::basic_ostream<Ch, Tr>& os, const realVector& container) {
        os << '{';

        auto start = std::begin(container), end = std::end(container);
        if (start != end) {
            os << *start;
            for (start = std::next(start); start != end; start = std::next(start))
                os << ", " << *start;
        }

        return os << '}';
    }
}

#endif // !__NEURAL_NETWORK_TYPES_H__
