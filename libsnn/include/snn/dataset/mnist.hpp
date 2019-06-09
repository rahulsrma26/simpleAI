#pragma once

#include <tuple>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include "snn/nntypes.hpp"
#include "snn/math/tensor.hpp"

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
#define bswap_32(x) __builtin_bswap32((x))
#define bswap_64(x) __builtin_bswap64((x))
#else

#include <byteswap.h>

#endif

namespace snn {
namespace dataset {
namespace mnist {

std::pair<tensor<real>, tensor<real>> load(const std::string& path_prefix);

} // namespace mnist
} // namespace dataset
} // namespace snn
