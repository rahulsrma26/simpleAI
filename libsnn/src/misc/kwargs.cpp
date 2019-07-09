#include "snn/misc/kwargs.hpp"

namespace snn {

namespace kwargs_pattern {
std::regex kwargs("('[^']*'|\"[^\"]*\"|\\([^\\)]*\\)|[^,])+");
std::regex named_args("^\\s*([A-Za-z0-9_]*)\\s*\\=\\s*([^\\s]|[^\\s].*[^\\s])\\s*$");
std::regex function_args("^\\s*([A-Za-z0-9_]+)\\s*\\((.*)\\)\\s*$");
std::regex int_value("[-]{0,1}[\\d]*[\\.]{0,1}[\\d]+");
std::regex double_value("[-]{0,1}[\\d]*[\\.]{0,1}[\\d]+([e]{0,1}[-]{0,1}[\\d]+){0,1}");
} // namespace kwargs_pattern

std::pair<std::string, std::string> split_named_args(const std::string& args) {
    std::smatch match;
    if (!std::regex_search(args, match, kwargs_pattern::named_args))
        throw std::runtime_error("Invalid named_args: " + args);
    return {match[1].str(), match[2].str()};
}

kwargs::kwargs(const char* args) {
    std::string str(args);
    std::smatch match;
    if (std::regex_search(str, match, kwargs_pattern::function_args)) {
        args_m.insert({"", str});
    } else {
        for (std::smatch match; std::regex_search(str, match, kwargs_pattern::kwargs);
             str = match.suffix().str()) {
            if (str.find('=') == std::string::npos)
                args_m.insert({"", str});
            else
                args_m.insert(split_named_args(match.str()));
        }
    }
}

kwargs::kwargs(const std::string& args) : args_m(std::move(kwargs(args.c_str()).args_m)) {}

kwargs::kwargs(const kwargs& args) : args_m(args.args_m) {}

kwargs::kwargs(kwargs&& args) noexcept : args_m(std::move(args.args_m)) {}

kwargs& kwargs::operator=(const kwargs& other) {
    args_m = other.args_m;
    return *this;
}
kwargs& kwargs::operator=(kwargs&& other) noexcept {
    args_m = std::move(other.args_m);
    return *this;
}

kwargs_type& kwargs::items() { return args_m; }
const kwargs_type& kwargs::items() const { return args_m; }

bool kwargs::has_key(const std::string& key) const { return args_m.find(key) != args_m.end(); }

std::pair<std::string, kwargs> kwargs::get_function(const std::string& key) const {
    auto r = args_m.find(key);
    if (r == args_m.end())
        throw std::runtime_error("Key not found: " + key);
    std::smatch match;
    if (!std::regex_search(r->second, match, kwargs_pattern::function_args))
        throw std::runtime_error("Invalid function_args: " + r->second);
    return {match[1].str(), kwargs(match[2].str().c_str())};
}

std::string kwargs::get_string(const std::string& key) const { return args_m.at(key); }

bool kwargs::get_bool(const std::string& key) const { return args_m.at(key) == TEXT::TRUE; }

int kwargs::get_int(const std::string& key) const { return std::stoi(args_m.at(key)); }

double kwargs::get_double(const std::string& key) const { return std::stod(args_m.at(key)); }

std::vector<std::string> kwargs::get_vector(const std::string& key) const {
    std::vector<std::string> values;
    std::string str = args_m.at(key);
    for (std::smatch match; std::regex_search(str, match, kwargs_pattern::kwargs);
         str = match.suffix().str())
        values.push_back(match.str());
    return values;
}

std::vector<int> kwargs::get_int_vector(const std::string& key) const {
    std::vector<int> numbers;
    std::string str = args_m.at(key);
    for (std::smatch match; std::regex_search(str, match, kwargs_pattern::int_value);
         str = match.suffix().str())
        numbers.push_back(std::stoi(match.str()));
    return numbers;
}

std::vector<double> kwargs::get_double_vector(const std::string& key) const {
    std::vector<double> numbers;
    std::string str = args_m.at(key);
    for (std::smatch match; std::regex_search(str, match, kwargs_pattern::double_value);
         str = match.suffix().str())
        numbers.push_back(std::stod(match.str()));
    return numbers;
}

std::string kwargs::get(const std::string& key, const std::string& default_value) const {
    auto r = args_m.find(key);
    if (r == args_m.end())
        return default_value;
    return r->second;
}

bool kwargs::get(const std::string& key, bool default_value) const {
    auto r = args_m.find(key);
    if (r == args_m.end())
        return default_value;
    return r->second == TEXT::TRUE;
}

int kwargs::get(const std::string& key, int default_value) const {
    auto r = args_m.find(key);
    if (r == args_m.end())
        return default_value;
    return std::stoi(r->second);
}

double kwargs::get(const std::string& key, double default_value) const {
    auto r = args_m.find(key);
    if (r == args_m.end())
        return default_value;
    return std::stod(r->second);
}

size_t kwargs::size() const { return args_m.size(); }

void kwargs::set(const std::string& key, const std::string& value) { args_m[key] = value; }

template<class T>
std::string vector_to_string(const std::vector<T>& list) {
    std::stringstream ss;
    ss << "(" << list[0];
    for (size_t i = 1; i < list.size(); i++)
        ss << ", " << list[i];
    ss << ")";
    return ss.str();
}

void kwargs::set_vector(const std::string& key, const std::vector<std::string>& list) {
    args_m[key] = vector_to_string<std::string>(list);
}

void kwargs::set_int_vector(const std::string& key, const std::vector<int>& list) {
    args_m[key] = vector_to_string<int>(list);
}

void kwargs::set_double_vector(const std::string& key, const std::vector<double>& list) {
    args_m[key] = vector_to_string<double>(list);
}

std::string kwargs::to_string() const {
    std::string out;
    int c = 0;
    for (const auto& r : args_m) {
        if (c++)
            out.append(", ");
        out.append(r.first);
        if (r.first != "")
            out.append("=");
        out.append(r.second);
    }
    return out;
}

std::ostream& operator<<(std::ostream& os, const kwargs& t) { return os << t.to_string(); }

} // namespace snn