#include "snn/model/autoencoder.hpp"

namespace snn {
namespace models {

void autoencoder::compile(const kwargs& args) {
    encoder.compile(args);
    decoder.compile(args);
    loss_m.create(args.get_string(TEXT::LOSS));
}

void autoencoder::summary() {
    printf("Encoder: \n");
    encoder.summary();
    printf("Decoder: \n");
    decoder.summary();
}

void autoencoder::run(const tensor<real>& x, const tensor<real>& y, const kwargs& args) {
    auto batch_size = (size_t)args.get(TEXT::BATCH_SIZE, 1);
    const int verbose = args.get(TEXT::VERBOSE, 2);
    const bool shuffle = args.get(TEXT::SHUFFLE, true);

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

    real avg_loss = 0, avg_dist = 0, total = 0;
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

        bx = encoder.forward(bx);
        bx = decoder.forward(bx);

        auto batch_loss = loss_m.f(bx, by);
        avg_loss += batch_loss * batch_size;
        avg_dist += math::l2_norm(bx, by) / batch_size;
        total += batch_size;

        auto dl = loss_m.df(bx, by);
        dl = decoder.backward(dl);
        dl = encoder.backward(dl);

        if (verbose == 2) {
            std::stringstream ss;
            ss << TEXT::LOSS << ": " << std::setprecision(5) << std::fixed << avg_loss / total
               << ", " << TEXT::DIST << ": " << std::setprecision(3) << std::fixed
               << avg_dist / total;
            pb.progress(batch_size, ss.str());
        }
    }
    if (verbose == 1) {
        std::cout << TEXT::LOSS << ": " << std::setprecision(5) << std::fixed << avg_loss / total
                  << ", " << TEXT::DIST << ": " << std::setprecision(3) << std::fixed
                  << avg_dist / total << std::endl;
    }
}

tensor<real> autoencoder::encode(const tensor<real>& x, const kwargs& args) {
    std::ignore = args;
    return encoder.predict(x);
}

tensor<real> autoencoder::decode(const tensor<real>& x, const kwargs& args) {
    std::ignore = args;
    return decoder.predict(x);
}

void autoencoder::save(std::ostream& os, bool save_gradient) const {
    os << variable_random_engine;
    loss_m.save(os);
    encoder.save(os, save_gradient);
    decoder.save(os, save_gradient);
};

void autoencoder::load(std::istream& is) {
    is >> variable_random_engine;
    loss_m.load(is);
    encoder.load(is);
    decoder.load(is);
};

} // namespace models
} // namespace snn
