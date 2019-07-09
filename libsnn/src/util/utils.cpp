#include "snn/util/utils.hpp"

namespace snn {

bool is_string(const std::string&) { return true; }

bool is_int(const std::string& s) { return std::regex_match(s, REGEX::int_pattern); }

bool is_double(const std::string& s) { return std::regex_match(s, REGEX::double_pattern); }

} // namespace snn