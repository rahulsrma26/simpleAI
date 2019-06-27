#include "snn/util/image.hpp"

namespace snn {

void save_pgm(const std::string& filename, const tensor<real>& matrix, int levels) {
    const auto dim = matrix.get_shape();
    if (dim.size() != 2)
        throw std::runtime_error("Should be a matrix to save as an image.");
    const size_t width = dim[1];
    const size_t height = dim[0];
    const int maxValue = levels - 1;

    std::ofstream fout;
    fout.open(filename);
    fout << "P2\n";
    fout << width << ' ' << height << '\n';
    fout << maxValue << '\n';

    for (size_t y = 0, i = 0; y < height; ++y, fout << '\n')
        for (size_t x = 0; x < width; ++x, ++i)
            fout << std::min(std::max(0, (int)(matrix[i] * maxValue)), maxValue) << ' ';
    fout.close();
}

} // namespace snn