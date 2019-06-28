#include "snn/dataset/spiral.hpp"

namespace snn {
namespace dataset {
namespace spiral {

dataset4 generate(shapeType samples, bool sin_cos) {
    constexpr real pi = 3.14159265358979f;
    std::vector<real> trainX, trainY, testX, testY;

    auto gen = [&](real deltaT, const real label, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            real r = 5.0f * i / n;
            real t = 1.75f * i / n * 2.0f * pi + deltaT;
            real x = r * sin(t);
            real y = r * cos(t);
            auto& xdata = i & 1 ? testX : trainX;
            auto& ydata = i & 1 ? testY : trainY;
            if (sin_cos)
                xdata.insert(xdata.end(), {x, y, (real)sin(x), (real)cos(y)});
            else
                xdata.insert(xdata.end(), {x, y});
            ydata.push_back(label);
        }
    };

    gen(0, 0, samples / 2);
    gen(pi, 1, samples - samples / 2);

    const shapeType dims = sin_cos ? 4 : 2;
    return {tensor<real>({(shapeType)(trainX.size() / dims), dims}, trainX),
            tensor<real>({(shapeType)(trainY.size()), 1}, trainY),
            tensor<real>({(shapeType)(testX.size() / dims), dims}, testX),
            tensor<real>({(shapeType)(testY.size()), 1}, testY)};
}

tensor<real> generate_grid(int n, bool sin_cos) {
    std::vector<real> inputs;
    for (int y = -n; y <= n; ++y)
        for (int x = -n; x <= n; ++x) {
            real xv = 5.0f * x / n, yv = 5.0f * y / n;
            if (sin_cos)
                inputs.insert(inputs.end(), {xv, yv, sin(xv), cos(xv)});
            else
                inputs.insert(inputs.end(), {xv, yv});
        }

    const shapeType dims = sin_cos ? 4 : 2;
    return tensor<real>({(shapeType)(inputs.size() / dims), dims}, inputs);
}

} // namespace spiral
} // namespace dataset
} // namespace snn