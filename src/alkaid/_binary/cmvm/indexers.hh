#pragma once

#include "types.hh"

Pair idx_mc(const DAState &state);
Pair idx_mc_dc(const DAState &state, bool absolute = false);
std::tuple<int8_t, int8_t, int8_t>
overlap_counts(const QInterval &q0, const QInterval &q1, const int8_t shift1);
Pair idx_wmc(const DAState &state);
Pair idx_wmc_dc(const DAState &state, bool absolute = false);

inline int8_t iceil_log2(float x) {
    uint32_t bits = std::bit_cast<uint32_t>(x);
    uint8_t exp = static_cast<uint8_t>((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;
    return static_cast<int8_t>(exp - 127 + (mant != 0));
}

// Scalar minimal_kif. Uses llrint (FE_TONEAREST) to match Python's round().
// The float32 cast in iceil_log2 is load-bearing — see iceil_log2 above.
inline void minimal_kif_one(
    double qmin,
    double qmax,
    double qstep,
    int32_t &out_k,
    int32_t &out_i,
    int32_t &out_f
) {
    if (qmin == 0.0 && qmax == 0.0) {
        out_k = 0;
        out_i = 0;
        out_f = 0;
        return;
    }
    const double safe_step = qstep > 0.0 ? qstep : 1.0;
    const int32_t k = qmin < 0.0 ? 1 : 0;
    const int32_t fractional = -iceil_log2(static_cast<float>(safe_step));
    const int64_t int_min = std::llrint(qmin / safe_step);
    const int64_t int_max = std::llrint(qmax / safe_step);
    const int64_t bits_arg = std::max(std::abs(int_min), int_max + 1);
    const int32_t bits = bits_arg > 0 ? iceil_log2(static_cast<float>(bits_arg)) : 0;
    out_k = k;
    out_i = bits - fractional;
    out_f = fractional;
}

// Writes n × 3 int32: [keep_negative, integers, fractionals] per entry.
inline void minimal_kif_batch(
    const double *qmins,
    const double *qmaxs,
    const double *qsteps,
    int32_t *out,
    size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        minimal_kif_one(
            qmins[i], qmaxs[i], qsteps[i], out[3 * i], out[3 * i + 1], out[3 * i + 2]
        );
    }
}
