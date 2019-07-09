#include "snn/util/argparse.hpp"

namespace snn {

argparse::argparse(const std::string& name, const std::string& description) {
    args_m.push_back(name);
    args_m.push_back(description);
}

std::string ltrim_dash(const std::string& str) {
    size_t start = 0;
    while (start < str.size() && str[start] == '-')
        start++;
    return str.substr(start);
}

void argparse::add(const std::vector<std::string>& params, const kwargs& args) {
    for (const auto& param : params) {
        auto key = ltrim_dash(param);
        if (index_m.find(key) != index_m.end())
            throw std::runtime_error("Duplicate param assignment for: " + key);
        index_m[key] = args_m.size();
    }
    args_m.push_back(args);
    auto& arg = args_m.back();
    arg.set_vector(TEXT::PARAMS, params);
    if (!arg.has_key(TEXT::TYPE))
        arg.set(TEXT::TYPE, TEXT::STRING);
    if (!arg.has_key(TEXT::USAGE)) {
        if (arg.has_key(TEXT::DEFAULT)) {
            arg.set(TEXT::VALUE, arg.get_string(TEXT::DEFAULT));
            arg.set(TEXT::USAGE, "[" + params.front() + " " + ltrim_dash(params.back()) + "]");
        } else {
            arg.set(TEXT::USAGE, params.front());
            required_m.push_back(args_m.size() - 1);
        }
    }
}

std::string argparse::operator[](const std::string& param) const {
    auto result = index_m.find(param);
    if (result == index_m.end())
        throw std::runtime_error("Key not found. Invalid parameter: " + param);
    return args_m[result->second].get_string(TEXT::VALUE);
}

bool check_valid(const kwargs& arg, const std::string& value) {
    auto type = arg.get_string(TEXT::TYPE);
    if (type == TEXT::INTEGER && !is_int(value)) {
        std::cout << "Invalid Value: '" << value << "'. Expected integer. \n";
        return false;
    } else if (type == TEXT::REAL && !is_double(value)) {
        std::cout << "Invalid Value: '" << value << "'. Expected real value. \n";
        return false;
    }
    return true;
}

bool argparse::parse(int argc, char* argv[]) {
    size_t required = 0;
    for (int i = 1; i < argc; i++) {
        std::string param(argv[i]);
        if (param == "--help" || param == "-h") {
            print_help();
            exit(0);
        }
        if (param[0] == '-') {
            auto result = index_m.find(ltrim_dash(param));
            if (result == index_m.end()) {
                std::cout << "Key not found. Invalid parameter: " << param << '\n';
                print_usage();
                exit(0);
            }
            auto& arg = args_m[result->second];
            auto type = arg.get_string(TEXT::TYPE);
            if (type == TEXT::FLAG) {
                arg.set(TEXT::VALUE, arg.get_string(TEXT::STORE));
            } else {
                if (i == argc - 1) {
                    std::cout << "Value not found for parameter: " << param << '\n';
                    print_usage();
                    exit(0);
                }
                std::string value(argv[++i]);
                if (!check_valid(arg, value)) {
                    print_usage();
                    exit(0);
                }
                arg.set(TEXT::VALUE, value);
            }
        } else if (required == required_m.size()) {
            std::cout << "Too many required parameters. \n";
            print_usage();
            exit(0);
        } else {
            auto& arg = args_m[required_m[required++]];
            if (!check_valid(arg, param)) {
                print_usage();
                exit(0);
            }
            arg.set(TEXT::VALUE, param);
        }
    }
    if (required < required_m.size()) {
        std::cout << "Required parameters missing. \n";
        print_usage();
        exit(0);
    }
    return true;
}

void argparse::print_usage() const {
    std::cout << TEXT::USAGE << ':' << '\n';
    std::cout << args_m[0] << " [-h]";
    for (size_t i = 2; i < args_m.size(); i++) {
        std::cout << ' ' << args_m[i].get_string(TEXT::USAGE);
    }
    std::cout << '\n';
}

void argparse::print_help() const {
    print_usage();
    std::cout << '\n' << args_m[1] << '\n';
    std::vector<bool> shown(args_m.size());
    std::cout << "\npositional arguments: \n";
    for (auto idx : required_m) {
        shown[idx] = true;
        std::cout << args_m[idx].get_string(TEXT::PARAMS);
        if (args_m[idx].has_key(TEXT::DESCRIPTION))
            std::cout << "\n\t\t" << args_m[idx].get_string(TEXT::DESCRIPTION);
        std::cout << '\n';
    }
    std::cout << "\noptional arguments: \n";
    for (size_t idx = 2; idx < args_m.size(); idx++) {
        if (shown[idx])
            continue;
        std::cout << args_m[idx].get_string(TEXT::PARAMS);
        if (args_m[idx].has_key(TEXT::DESCRIPTION))
            std::cout << "\n\t\t" << args_m[idx].get_string(TEXT::DESCRIPTION);
        std::cout << "(" << TEXT::DEFAULT << "=" << args_m[idx].get_string(TEXT::DEFAULT) << ")\n";
    }
    std::cout << "(-h, --help)";
    std::cout << "\n\t\tshows this help\n";
}

std::string argparse::to_string() const {
    std::stringstream ss;
    for (const auto& arg : args_m)
        ss << arg << '\n';
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const argparse& t) { return os << t.to_string(); }

} // namespace snn