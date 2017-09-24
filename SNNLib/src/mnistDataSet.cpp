#include "stdafx.h"
#include "mnistDataSet.h"

namespace dataset {
    MnistDataSet::MnistDataSet() :
        _count(0), _width(0), _height(0), _imageBuffer(nullptr), _labelBuffer(nullptr) {}


    MnistDataSet::~MnistDataSet() {}


    uint32_t MnistDataSet::size() { return _count; }


    std::pair<realMatrix, realMatrix> MnistDataSet::getData(size_t begin, size_t end){
        if (begin > size())
            return { {},{} };

        if (end > size())
            end = size();

        const auto length = _height*_width;
        realMatrix dataX, dataY;
        for (size_t idx = begin; idx < end; ++idx) {
            realVector image(length);
            for (uint32_t i = 0, j = idx*length; i < length; i++, j++)
                image[i] = _imageBuffer[j] / 255.0;
            dataX.emplace_back(image);

            realVector label(10, 0.0);
            label[_labelBuffer[idx]] = 1.0;
            dataY.emplace_back(label);
        }
        return { dataX,  dataY };
    }


    uint32_t MnistDataSet::show(uint32_t idx) const {
        const auto length = (idx + 1)*_height*_width;

        for (uint32_t i = length - _height*_width; i < length;) {
            std::cout << ((_imageBuffer[i] > 127) ? "#" : ".");
            if (++i % _width == 0)
                std::cout << '\n';
        }

        return _labelBuffer[idx];
    }


    bool MnistDataSet::Load(const std::string &imageFile, const std::string &labelFile) {
        {
            std::ifstream imgfile(imageFile, std::ios::binary);
            if (imgfile.fail())
                return false;

            uint32_t magic;
            imgfile.read((char*)&magic, sizeof(uint32_t));
            magic = _byteswap_ulong(magic);
            if (magic != 2051) {
                imgfile.close();
                return false;
            }

            imgfile.read((char*)&_count, sizeof(uint32_t));
            imgfile.read((char*)&_height, sizeof(uint32_t));
            imgfile.read((char*)&_width, sizeof(uint32_t));
            _count = _byteswap_ulong(_count);
            _height = _byteswap_ulong(_height);
            _width = _byteswap_ulong(_width);

            std::unique_ptr<uint8_t[]> imgBuffer(new uint8_t[_count*_height*_width]);
            _imageBuffer = std::move(imgBuffer);

            imgfile.read((char*)_imageBuffer.get(), _count*_height*_width);
            imgfile.close();
        }
        {
            std::ifstream lblfile(labelFile, std::ios::binary);
            if (lblfile.fail())
                return false;

            uint32_t magic;
            lblfile.read((char*)&magic, sizeof(uint32_t));
            magic = _byteswap_ulong(magic);
            if (magic != 2049) {
                lblfile.close();
                return false;
            }

            lblfile.read((char*)&magic, sizeof(uint32_t));
            magic = _byteswap_ulong(magic);
            if (magic != _count) {
                lblfile.close();
                return false;
            }

            std::unique_ptr<uint8_t[]> lblBuffer(new uint8_t[_count]);
            _labelBuffer = std::move(lblBuffer);

            lblfile.read((char*)_labelBuffer.get(), _count);
            lblfile.close();
        }
    }
}