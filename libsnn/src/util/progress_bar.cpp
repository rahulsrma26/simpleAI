#include "snn/util/progress_bar.hpp"

namespace snn {

progress_bar::progress_bar(size_t total)
    : total_m(total), current_m(0), last_m(0), avg_time_m(-1), places_m(1) {
    last_time_m = start_time_m = std::chrono::high_resolution_clock::now();
    for (size_t x = total_m; x; x /= 10)
        places_m++;
}

void display_duration(double duration) {
    duration /= 1e9;
    if (duration > 3600 * 24) {
        int days = duration / (3600 * 24);
        duration -= (double)days * 3600 * 24;
        std::cout << days << ':';
    }
    if (duration > 3600) {
        int hours = duration / 3600;
        duration -= (double)hours * 3600;
        if (hours < 10)
            std::cout << '0';
        std::cout << hours << ':';
    }
    if (duration > 60) {
        int minutes = duration / 60;
        duration -= (double)minutes * 60;
        if (minutes < 10)
            std::cout << '0';
        std::cout << minutes << ':';
    }
    if (duration < 10)
        std::cout << '0';
    std::cout << std::fixed << std::setprecision(2) << duration << "s";
}

void display_bar(size_t length, double complete_ratio) {
    std::cout << ' ' << '[';
    size_t completed = length * complete_ratio;
    size_t i = 0;
    for (; i < completed; i++)
        std::cout << '=';
    if (completed != length) {
        std::cout << '>';
        i++;
    }
    for (; i < length; i++)
        std::cout << '.';
    std::cout << ']' << ' ';
}

void display_speed(double duration, size_t total) {
    std::cout << ' ';
    auto rate = duration / total;
    if (rate < 1000) {
        std::cout << (int)(rate + 0.5) << "ns/rec ";
        return;
    }
    rate /= 1000;
    if (rate < 10000) {
        std::cout << (int)(rate + 0.5) << "us/rec ";
        return;
    }
    rate /= 1000;
    if (rate < 10000) {
        std::cout << (int)(rate + 0.5) << "ms/rec ";
        return;
    }
    rate /= 1000;
    if (rate < 3600) {
        std::cout << (int)(rate + 0.5) << "s/rec ";
        return;
    }
    rate /= 3600;
    std::cout << (int)(rate + 0.5) << "h/rec ";
}

void progress_bar::progress(size_t steps, const std::string& msg) {
    std::ios_base::fmtflags f(std::cout.flags());
    current_m += steps;
    auto current_time = std::chrono::high_resolution_clock::now();
    if (current_m >= total_m) {
        double duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time_m)
                .count();
        std::cout << std::setw(places_m) << current_m << '/' << total_m;
        display_bar(30, (double)current_m / total_m);
        display_duration(duration);
        display_speed(duration, total_m);
        std::cout << msg << ' ';
        std::cout << "    " << std::endl;
        std::cout.flags(f);
        return;
    }

    double duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - last_time_m).count();
    if (duration < 2e8)
        return;

    double cur_time = last_m == current_m ? duration : duration / (current_m - last_m);
    avg_time_m = avg_time_m >= 0 ? 0.6 * avg_time_m + 0.4 * cur_time : cur_time;
    double eta = (total_m - current_m) * avg_time_m;

    std::cout << std::setw(places_m) << current_m << '/' << total_m;
    display_bar(30, (double)current_m / total_m);
    std::cout << "ETA: ";
    display_duration(eta);
    std::cout << ' ' << msg;
    std::cout << "    \r" << std::flush;

    last_time_m = current_time;
    last_m = current_m;
    std::cout.flags(f);
}

} // namespace snn