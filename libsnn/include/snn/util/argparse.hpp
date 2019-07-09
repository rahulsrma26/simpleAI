#pragma once

#include <string>
#include <vector>
#include "snn/misc/kwargs.hpp"
#include "snn/util/utils.hpp"

namespace snn {

class argparse {
    std::string name_m;
    std::vector<kwargs> args_m;
    std::vector<size_t> required_m;
    std::unordered_map<std::string, size_t> index_m;

public:
    argparse(const std::string& name, const std::string& description = "");
    void add(const std::vector<std::string>&, const kwargs& args = "");
    std::string operator[](const std::string&) const;
    bool parse(int argc, char* argv[]);
    void print_usage() const;
    void print_help() const;
    std::string to_string() const;
};

std::ostream& operator<<(std::ostream& os, const argparse& args);

} // namespace snn