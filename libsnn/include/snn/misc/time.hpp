#pragma once

#include <chrono>

namespace snn {
namespace time {

class duration {
    double duration_m;

public:
    explicit duration(double);
    double seconds();
};

class clock {
    std::chrono::time_point<std::chrono::high_resolution_clock> time_point_m;

public:
    clock();
    std::chrono::time_point<std::chrono::high_resolution_clock> get_time_point() const;

    friend duration operator-(const clock&, const clock&);
};

} // namespace time
} // namespace snn