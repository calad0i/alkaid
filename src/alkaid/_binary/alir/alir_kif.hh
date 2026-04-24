#pragma once

// Scalar ports of cmvm::iceil_log2 / Python minimal_kif, used by the JSON
// ingestion path. The float32 cast in iceil_log2_f32 is load-bearing: its
// precision loss is what produces byte-exact parity with the Python path
// (e.g. 2^37 + 1 rounds to 2^37, giving log2 = 37 not 38).

#include "alir_types.hh"

#include <bit>
#include <cmath>
#include <cstdint>

namespace alir {

    inline int8_t iceil_log2_f32(float x) {
        uint32_t bits = std::bit_cast<uint32_t>(x);
        uint8_t exp = static_cast<uint8_t>((bits >> 23) & 0xFF);
        uint32_t mant = bits & 0x7FFFFF;
        return static_cast<int8_t>(exp - 127 + (mant != 0));
    }

    inline DType minimal_kif(double qmin, double qmax, double qstep) {
        if (qmin == 0.0 && qmax == 0.0)
            return DType{0, 0, 0};
        const int32_t keep_negative = (qmin < 0.0) ? 1 : 0;
        const int32_t fractional = -iceil_log2_f32(static_cast<float>(qstep));
        const int64_t int_min = static_cast<int64_t>(std::llround(qmin / qstep));
        const int64_t int_max = static_cast<int64_t>(std::llround(qmax / qstep));
        int64_t abs_imin = int_min < 0 ? -int_min : int_min;
        int64_t bits_arg = abs_imin;
        const int64_t cand = int_max + 1;
        if (cand > bits_arg)
            bits_arg = cand;
        int32_t bits = 0;
        if (bits_arg > 0)
            bits = iceil_log2_f32(static_cast<float>(bits_arg));
        const int32_t integers = bits - fractional;
        return DType{keep_negative, integers, fractional};
    }

    // Mirror of Python LookupTable._get_pads, pad_left only.
    inline int32_t table_pad_left(double qmin, double qstep, const DType &kif) {
        if (kif.is_signed) {
            return static_cast<int32_t>(std::llround((qmin + std::ldexp(1.0, kif.integers)) / qstep));
        }
        return static_cast<int32_t>(std::llround(qmin / qstep));
    }

} // namespace alir
