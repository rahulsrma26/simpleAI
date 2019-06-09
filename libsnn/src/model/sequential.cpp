#include "snn/model/sequential.hpp"

namespace snn {
namespace models {

void sequential::add(const kwargs& args) {
    auto [type, sub_args] = args.get_function("");
    if (sub_args.has_key(TEXT::INPUT)) {
        layers_m.emplace_back();
        layers_m.back().create(args);
    } else {
        sub_args.set(TEXT::INPUT, std::to_string(layers_m.back().output()));
        layers_m.emplace_back();
        auto nargs = type + "(" + sub_args.to_string() + ")";
        layers_m.back().create(nargs);
    }
}

void sequential::compile(const kwargs& args) {
    auto optimizer = args.get_string(TEXT::OPTIMIZER);
    for (auto& layer : layers_m)
        layer.set_optimizer(optimizer);
    loss_m.create(args.get_string(TEXT::LOSS));
}

void sequential::summary() {
    printf("==========================================\n");
    printf("               Layer    Output  Parameters\n");
    printf("------------------------------------------\n");
    size_t total = 0;
    for (auto& layer : layers_m) {
        printf("%20s %9d %11d \n", layer.name().c_str(), (int)layer.output(), (int)layer.params());
        total += layer.params();
    }
    printf("==========================================\n");
    std::cout << "Total parameters : " << total << std::endl;
}

real get_correct_count(const tensor<real>& output, const tensor<real>& label) {
    real correct = 0;
    const auto dim = output.get_shape();
    if (dim.size() != 2)
        throw std::runtime_error("not a valid tensor for accuracy.");
    const auto data_points = dim.front();
    const auto classes = dim.back();
    if (classes == 1) {
        for (size_t i = 0; i < output.size(); i++)
            correct += (output[i] < 0.5 ? 0 : 1) == label[i];
    } else {
        for (size_t i = 0, k = 0; i < data_points; i++) {
            real max_out = output[k], max_out_j = 0;
            real max_lbl = label[k], max_lbl_j = 0;
            k++;
            for (size_t j = 1; j < classes; j++, k++) {
                if (max_out < output[k]) {
                    max_out = output[k];
                    max_out_j = j;
                }
                if (max_lbl < label[k]) {
                    max_lbl = label[k];
                    max_lbl_j = j;
                }
            }
            correct += max_out_j == max_lbl_j;
        }
    }
    return correct;
}

void sequential::run(const tensor<real>& x, const tensor<real>& y, const kwargs& args) {
    auto batch_size = (size_t)args.get(TEXT::BATCH_SIZE, 1);
    const int verbose = args.get(TEXT::VERBOSE, 2);
    const bool shuffle = args.get(TEXT::SHUFFLE, true);
    const bool train = args.get(TEXT::TRAIN, true);

    const size_t data_points = x.get_shape().front();
    const size_t x_size = x.size() / data_points;
    const size_t y_size = y.size() / data_points;

    auto batch_x_shape = x.get_shape();
    auto batch_y_shape = y.get_shape();
    batch_x_shape[0] = batch_y_shape[0] = batch_size;

    std::vector<size_t> indices(data_points);
    for (size_t idx = 0; idx < data_points; idx++)
        indices[idx] = idx;
    if (shuffle)
        std::shuffle(indices.begin(), indices.end(), variable_random_engine);

    real avg_loss = 0, avg_acc = 0, total = 0;
    tensor<real> bx(batch_x_shape), by(batch_y_shape);

    progress_bar pb(data_points);

    for (size_t idx = 0; idx < data_points; idx += batch_size) {
        if (idx + batch_size >= data_points) {
            batch_size = data_points - idx;
            batch_x_shape[0] = batch_y_shape[0] = batch_size;
            bx = tensor<real>(batch_x_shape);
            by = tensor<real>(batch_y_shape);
        }
        for (size_t i = 0; i < batch_size; i++)
            for (size_t j = 0; j < x_size; j++)
                bx[i * x_size + j] = x[indices[idx + i] * x_size + j];
        for (size_t i = 0; i < batch_size; i++)
            for (size_t j = 0; j < y_size; j++)
                by[i * y_size + j] = y[indices[idx + i] * y_size + j];

        auto z = bx;
        for (size_t i = 0; i < layers_m.size(); i++)
            z = layers_m[i].forward(z);

        auto batch_loss = loss_m.f(z, by);
        avg_loss += batch_loss * batch_size;
        avg_acc += get_correct_count(z, by);
        total += batch_size;

        if (train) {
            auto l = loss_m.df(z, by);
            for (int i = layers_m.size() - 1; i >= 0; i--)
                l = layers_m[i].backward(l);
        }

        if (verbose == 2) {
            std::stringstream ss;
            ss << TEXT::LOSS << ": " << std::setprecision(5) << std::fixed << avg_loss / total
               << ", " << TEXT::ACC << ": " << std::setprecision(3) << std::fixed
               << 100 * avg_acc / total;
            pb.progress(batch_size, ss.str());
        }
    }
    if (verbose == 1) {
        std::cout << TEXT::LOSS << ": " << std::setprecision(5) << std::fixed << avg_loss / total
                  << ", " << TEXT::ACC << ": " << std::setprecision(3) << std::fixed
                  << 100 * avg_acc / total << std::endl;
    }
}

tensor<real> sequential::predict(const tensor<real>& x, const kwargs& args) {
    std::ignore = args;
    auto y = x;
    for (auto& layer : layers_m)
        y = layer.forward(y);
    return y;
}

void sequential::save(std::ostream& os, bool save_gradient) const {
    os << variable_random_engine;
    uint32_t num_layers = layers_m.size();
    os.write(reinterpret_cast<const char*>(&num_layers), sizeof(uint32_t));
    for (auto& layer : layers_m)
        layer.save(os, save_gradient);
    loss_m.save(os);
};

void sequential::load(std::istream& is) {
    is >> variable_random_engine;
    uint32_t num_layers;
    is.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));
    for (uint32_t i = 0; i < num_layers; i++) {
        layers_m.emplace_back();
        layers_m.back().load(is);
    }
    loss_m.load(is);
};

} // namespace models
} // namespace snn
