#include "snn/misc/time.hpp"

namespace snn {
namespace time {

// duration

duration::duration(double seconds) : duration_m(seconds) {}

double duration::seconds() { return duration_m / 1e9; }

// clock

clock::clock() : time_point_m(std::chrono::high_resolution_clock::now()) {}

std::chrono::time_point<std::chrono::high_resolution_clock> clock::get_time_point() const {
    return time_point_m;
}

duration operator-(const clock& end, const clock& start) {
    return duration(std::chrono::duration_cast<std::chrono::nanoseconds>(end.get_time_point() -
                                                                         start.get_time_point())
                        .count());
}

} // namespace time
} // namespace snn