#pragma once
#include <regex>
#include <string>
#include <unordered_map>
#include "snn/misc/constant.hpp"

namespace snn {

bool is_string(const std::string&);

bool is_int(const std::string&);

bool is_double(const std::string&);

template <class A, class B>
auto map_reverser(const std::unordered_map<A, B>& ab) {
    std::unordered_map<B, A> ba;
    for (auto [k, v] : ab)
        ba.insert({v, k});
    return ba;
}

} // namespace snn