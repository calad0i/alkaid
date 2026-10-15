#pragma once

// Scalar ports of cmvm::iceil_log2 / Python minimal_kif, used by the JSON
// ingestion path.

#include "alir_types.hh"

#include <bit>
#include <cmath>
#include <cstdint>

namespace alir {

    inline int16_t iceil_log2_f64(double x) {
        uint64_t bits = std::bit_cast<uint64_t>(x);
        uint16_t exp = static_cast<uint16_t>((bits >> 52) & 0x7FF);
        uint64_t mant = bits & 0xFFFFFFFFFFFFFULL;
        return static_cast<int16_t>(exp - 1023 + (mant != 0));
    }

    inline DType minimal_kif(double qmin, double qmax, double qstep) {
        if (qmin == 0.0 && qmax == 0.0)
            return DType{0, 0, 0};
        const int32_t _signed = (qmin < 0.0) ? 1 : 0;
        const int32_t fractional = -iceil_log2_f64(qstep);
        const double int_min = std::round(qmin / qstep);
        const double int_max = std::round(qmax / qstep);
        double bits_arg = std::abs(int_min);
        const double cand = int_max + 1;
        if (cand > bits_arg)
            bits_arg = cand;
        int32_t bits = 0;
        if (bits_arg > 0)
            bits = iceil_log2_f64(bits_arg);
        const int32_t integers = bits - fractional;
        return DType{_signed, integers, fractional};
    }

    // Mirror of Python LookupTable._get_pads, pad_left only.
    inline int32_t table_pad_left(double qmin, double qstep, const DType &kif) {
        if (kif.is_signed) {
            return static_cast<int32_t>(std::llround((qmin + std::ldexp(1.0, kif.integers)) / qstep));
        }
        return static_cast<int32_t>(std::llround(qmin / qstep));
    }

} // namespace alir
