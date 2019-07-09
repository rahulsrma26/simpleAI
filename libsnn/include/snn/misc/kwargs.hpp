#pragma once

#include <ostream>
#include <string>
#include <sstream>
#include <regex>
#include <vector>
#include <unordered_map>
#include <utility>
#include "snn/misc/string_constant.hpp"

namespace snn {

typedef std::unordered_map<std::string, std::string> kwargs_type;

std::pair<std::string, std::string> split_named_args(const std::string&);

class kwargs {
    kwargs_type args_m;

public:
    kwargs(const char* args);
    kwargs(const std::string& args);

    kwargs(const kwargs& args);
    kwargs(kwargs&& args) noexcept;

    kwargs& operator=(const kwargs&);
    kwargs& operator=(kwargs&&) noexcept;

    kwargs_type& items();
    const kwargs_type& items() const;

    bool has_key(const std::string& key) const;

    std::pair<std::string, kwargs> get_function(const std::string& key) const;

    std::string get_string(const std::string& key) const;

    bool get_bool(const std::string& key) const;

    int get_int(const std::string& key) const;

    double get_double(const std::string& key) const;

    std::vector<std::string> get_vector(const std::string& key) const;

    std::vector<int> get_int_vector(const std::string& key) const;

    std::vector<double> get_double_vector(const std::string& key) const;

    bool get(const std::string& key, bool default_value) const;

    int get(const std::string& key, int default_value) const;

    double get(const std::string& key, double default_value) const;

    std::string get(const std::string& key, const std::string& default_value) const;

    size_t size() const;

    void set(const std::string& key, const std::string& value);

    void set_vector(const std::string& key, const std::vector<std::string>& list);

    void set_int_vector(const std::string& key, const std::vector<int>& list);

    void set_double_vector(const std::string& key, const std::vector<double>& list);

    std::string to_string() const;

    friend std::ostream& operator<<(std::ostream&, const kwargs&);
};

} // namespace snn