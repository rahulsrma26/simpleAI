#pragma once

namespace snn {

#include <unordered_map>

template <class A, class B>
auto map_reverser(const std::unordered_map<A, B> &ab) {
    std::unordered_map<B, A> ba;
    for (auto [k, v] : ab)
        ba.insert({v, k});
    return ba;
}

} // namespace snn