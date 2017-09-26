#ifndef __MNIST_DATASET_H__
#define __MNIST_DATASET_H__

#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <random>
#include <fstream>
#include <intrin.h>
#include <iostream>

namespace dataset {
    using real = float;
    using realVector = std::vector<real>;
    using realMatrix = std::vector<realVector>;

    class MnistDataSet
    {
        uint32_t _count;
        uint32_t _width;
        uint32_t _height;
        std::unique_ptr<uint8_t[]> _imageBuffer;
        std::unique_ptr<uint8_t[]> _labelBuffer;

    public:
        MnistDataSet();
        bool Load(const std::string &imageFile, const std::string &labelFile);
        uint32_t size();
        std::pair<realMatrix, realMatrix> getData();
        uint32_t show(uint32_t idx) const;
        ~MnistDataSet();
        static std::pair<realMatrix, realMatrix> get(const std::string &imageFile, const std::string &labelFile);
    };
}

#endif //!__MNIST_DATASET_H__
