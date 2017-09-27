#ifndef __SPIRAL_DATASET_H__
#define __SPIRAL_DATASET_H__

#include <fstream>
#include <vector>
#include <string>
#include <tuple>

namespace dataset {

    template<class T>
    auto spiralDataset(size_t samples = 1000, bool addSin = false) {
        constexpr T pi = 3.14159265358979f;
        std::vector<std::vector<T>> trainX, trainY, testX, testY;

        auto gen = [&](T deltaT, const std::vector<T>& label, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                real r = 5.0f * i / n;
                real t = 1.75f * i / n * 2.0f * pi + deltaT;
                real x = r * sin(t);
                real y = r * cos(t);
                auto& xdata = i & 1 ? testX : trainX;
                auto& ydata = i & 1 ? testY : trainY;
                if (addSin)
                    xdata.push_back({ x, y, sin(x), cos(y) });
                else
                    xdata.push_back({ x, y });
                ydata.push_back(label);
            }
        };

        gen(0, { 0 }, samples / 2);
        gen(pi, { 1 }, samples - samples / 2);
        return std::make_tuple(trainX, trainY, testX, testY);
    }

    template<class T>
    auto spiralGrid(int n = 100, bool addSin = false) {
        std::vector<std::vector<T>> inputs;
        for (int y = -n; y <= n; ++y) {
            for (int x = -n; x <= n; ++x) {
                T xv = 5.0f * x / n, yv = 5.0f * y / n;
                if (addSin)
                    inputs.push_back({ xv, yv, sin(xv), cos(xv) });
                else
                    inputs.push_back({ xv, yv });
            }
        }
        return inputs;
    }

    template<class T>
    void saveGrid(const std::string& filename, const std::vector<std::vector<T>>& out, size_t width, int levels=100) {
        using namespace std;
        const size_t n = out.size();
        const size_t height = n / width;
        const int maxValue = levels - 1;

        ofstream fout;
        fout.open(filename);
        fout << "P2" << endl;
        fout << width << ' ' << height << endl;
        fout << maxValue << endl;

        for (size_t y = 0, i = 0; y < height; ++y, fout << endl)
            for (size_t x = 0; x < width; ++x, ++i)
                fout << min(max(0, (int)(out[i][0] * maxValue)), maxValue) << ' ';
        fout.close();
    }
}
#endif //!#define __SPIRAL_DATASET_H__