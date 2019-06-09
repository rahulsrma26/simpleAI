#include "snn/loss.hpp"

namespace snn {

void loss::create(const kwargs& args) {
    auto [type, sub_args] = args.get_function("");
    if (type == losses::quadratic::type())
        loss_m = std::make_unique<losses::quadratic>(sub_args);
    else if (type == losses::hillinger::type())
        loss_m = std::make_unique<losses::hillinger>(sub_args);
    else if (type == losses::cross_entropy::type())
        loss_m = std::make_unique<losses::cross_entropy>(sub_args);
    else
        throw std::runtime_error("Invalid loss type: " + type);
}

void loss::load(std::istream& is) {
    uint32_t type_length;
    is.read(reinterpret_cast<char*>(&type_length), sizeof(uint32_t));
    std::string type(type_length, ' ');
    is.read(reinterpret_cast<char*>(&type[0]), type_length);
    if (type == losses::quadratic::type())
        loss_m = std::make_unique<losses::quadratic>(is);
    else if (type == losses::hillinger::type())
        loss_m = std::make_unique<losses::hillinger>(is);
    else if (type == losses::cross_entropy::type())
        loss_m = std::make_unique<losses::cross_entropy>(is);
    else
        throw std::runtime_error("Invalid loss type: " + type);
};

void loss::save(std::ostream& os) const {
    std::string type = loss_m->name();
    uint32_t type_length = type.length();
    os.write(reinterpret_cast<const char*>(&type_length), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&type[0]), type_length);
    loss_m->save(os);
};

std::string loss::name() const { return loss_m->name(); }

real loss::f(const tensor<real>& o, const tensor<real>& l) const { return loss_m->f(o, l); }

tensor<real> loss::df(const tensor<real>& o, const tensor<real>& l) const {
    return loss_m->df(o, l);
}

} // namespace snn