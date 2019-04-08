#ifndef __MNIST_DATASET_H__
#define __MNIST_DATASET_H__

#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>

#ifdef _MSC_VER

#include <stdlib.h>
#define bswap_32(x) _byteswap_ulong(x)
#define bswap_64(x) _byteswap_uint64(x)

#elif defined(__APPLE__)

// Mac OS X / Darwin features
#include <libkern/OSByteOrder.h>
#define bswap_32(x) OSSwapInt32(x)
#define bswap_64(x) OSSwapInt64(x)

#elif defined(__sun) || defined(sun)

#include <sys/byteorder.h>
#define bswap_32(x) BSWAP_32(x)
#define bswap_64(x) BSWAP_64(x)

#elif defined(__FreeBSD__)

#include <sys/endian.h>
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)

#elif defined(__OpenBSD__)

#include <sys/types.h>
#define bswap_32(x) swap32(x)
#define bswap_64(x) swap64(x)

#elif defined(__NetBSD__)

#include <sys/types.h>
#include <machine/bswap.h>
#if defined(__BSWAP_RENAME) && !defined(__bswap_32)
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)
#endif

#elif defined(__MINGW32__)
#  define bswap_32(x) __builtin_bswap32((x))
#  define bswap_64(x) __builtin_bswap64((x))
#else

#include <byteswap.h>

#endif

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
