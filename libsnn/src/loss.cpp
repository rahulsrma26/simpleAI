#include "snn/loss.hpp"

namespace snn {

template <class T>
std::unique_ptr<losses::base_loss> get_loss_from_type(const std::string& type, T args) {
    if (type == losses::quadratic::type())
        return std::make_unique<losses::quadratic>(args);
    else if (type == losses::hillinger::type())
        return std::make_unique<losses::hillinger>(args);
    else if (type == losses::cross_entropy::type())
        return std::make_unique<losses::cross_entropy>(args);
    throw std::runtime_error("Invalid loss type: " + type);
}

void loss::create(const kwargs& args) {
    auto [type, sub_args] = args.get_function("");
    loss_m = get_loss_from_type<const kwargs&>(type, sub_args);
}

void loss::load(std::istream& is) {
    uint32_t type_length;
    is.read(reinterpret_cast<char*>(&type_length), sizeof(uint32_t));
    std::string type(type_length, ' ');
    is.read(reinterpret_cast<char*>(&type[0]), type_length);
    loss_m = get_loss_from_type<std::istream&>(type, is);
}

void loss::save(std::ostream& os) const {
    std::string type = loss_m->name();
    uint32_t type_length = type.length();
    os.write(reinterpret_cast<const char*>(&type_length), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&type[0]), type_length);
    loss_m->save(os);
}

std::string loss::name() const { return loss_m->name(); }

real loss::f(const tensor<real>& o, const tensor<real>& l) const {
    if (o.get_shape() != l.get_shape())
        throw std::runtime_error("Output Shape " + vector_to_string(o.get_shape()) +
                                 " != " + vector_to_string(l.get_shape()) +
                                 " Lable Shape. Can not calculate loss.");
    return loss_m->f(o, l);
}

tensor<real> loss::df(const tensor<real>& o, const tensor<real>& l) const {
    if (o.get_shape() != l.get_shape())
        throw std::runtime_error("Output Shape " + vector_to_string(o.get_shape()) +
                                 " != " + vector_to_string(l.get_shape()) +
                                 " Lable Shape. Can not calculate loss derivative.");
    return loss_m->df(o, l);
}

} // namespace snn