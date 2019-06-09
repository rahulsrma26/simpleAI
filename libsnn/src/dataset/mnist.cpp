#include "snn/dataset/mnist.hpp"

namespace snn {
namespace dataset {
namespace mnist {

dataset2 load(const std::string& path_prefix) {
    auto image_file = path_prefix + "-images-idx3-ubyte";
    auto label_file = path_prefix + "-labels-idx1-ubyte";
    uint32_t _count, _height, _width;
    tensor<real> dataX({1}), dataY({1});
    {
        std::ifstream imgfile(image_file, std::ios::binary);
        if (imgfile.fail())
            throw std::runtime_error("cant load file: " + std::string(image_file));

        uint32_t magic;
        imgfile.read((char*)&magic, sizeof(uint32_t));
        magic = bswap_32(magic);
        if (magic != 2051) {
            imgfile.close();
            throw std::runtime_error("not a valid file: " + std::string(image_file));
        }

        imgfile.read((char*)&_count, sizeof(uint32_t));
        imgfile.read((char*)&_height, sizeof(uint32_t));
        imgfile.read((char*)&_width, sizeof(uint32_t));
        _count = bswap_32(_count);
        _height = bswap_32(_height);
        _width = bswap_32(_width);

        std::vector<uint8_t> imgBuffer(_count * _height * _width);
        imgfile.read((char*)imgBuffer.data(), _count * _height * _width);
        imgfile.close();
        dataX = tensor<real>({(shapeType)_count, (shapeType)_height, (shapeType)_width});
        for (uint32_t i = 0; i < dataX.size(); i++)
            dataX[i] = imgBuffer[i] / 255.0f;
    }
    {
        std::ifstream lblfile(label_file, std::ios::binary);
        if (lblfile.fail())
            throw std::runtime_error("cant load file: " + std::string(label_file));

        uint32_t magic;
        lblfile.read((char*)&magic, sizeof(uint32_t));
        magic = bswap_32(magic);
        if (magic != 2049) {
            lblfile.close();
            throw std::runtime_error("not a valid file: " + std::string(label_file));
        }

        lblfile.read((char*)&magic, sizeof(uint32_t));
        magic = bswap_32(magic);
        if (magic != _count) {
            lblfile.close();
            throw std::runtime_error("count does not match with data in label file: " +
                                     std::string(label_file));
        }

        std::vector<uint8_t> lblBuffer(_count);
        lblfile.read((char*)lblBuffer.data(), _count);
        lblfile.close();
        dataY = tensor<real>({(shapeType)_count, 10});
        for (uint32_t i = 0; i < _count; i++)
            dataY[i * 10 + lblBuffer[i]] = 1;
    }
    return {dataX, dataY};
}

} // namespace mnist
} // namespace dataset
} // namespace snn
