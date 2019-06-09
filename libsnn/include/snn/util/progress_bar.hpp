#pragma once
#include <iostream>
#include <vector>
#include <type_traits>
#include <iomanip>
#include <string>
#include <chrono>

namespace snn {

class progress_bar {
    size_t total_m, current_m, last_m;
    double avg_time_m;
    int places_m;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_m;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_time_m;

public:
    explicit progress_bar(size_t total);
    void progress(size_t steps, const std::string& msg);
};

} // namespace snn