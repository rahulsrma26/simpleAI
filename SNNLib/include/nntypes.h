#ifndef __NEURAL_NETWORK_TYPES_H__
#define __NEURAL_NETWORK_TYPES_H__

#include <vector>
#include <random>
#include <chrono>

namespace simpleNN {
    using real = double;
    using realVector = std::vector<real>;
    using realMatrix = std::vector<realVector>;

    static std::default_random_engine nnRandomEngine;
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
