#include "verilated.h"
#include <cassert>
#include <cstdint>
#include <vector>
template <size_t bw, size_t N_in> std::vector<int32_t> bitpack(const int64_t *values) {
    static_assert(bw > 0 && bw <= 64, "Bit width must be between 1 and 64");

    constexpr size_t total_bits = N_in * bw;
    constexpr size_t result_size = (total_bits + 31) / 32;
    std::vector<int32_t> result(result_size, 0);

    constexpr uint64_t mask = (bw == 64) ? ~uint64_t(0) : ((uint64_t(1) << bw) - 1);

    size_t bit_pos = 0;
    for (size_t i = 0; i < N_in; i++) {
        int64_t val = values[i];
        uint64_t bits = static_cast<uint64_t>(val) & mask;

        size_t result_idx = bit_pos / 32;
        size_t offset = bit_pos % 32;

        // base case
        result[result_idx] |= static_cast<uint32_t>(bits << offset);

        // cross boundary case (one chunk over)
        if (offset + bw > 32 && result_idx + 1 < result.size()) {
            result[result_idx + 1] |= static_cast<uint32_t>(bits >> (32 - offset));
        }

        // cross boundary case (two chunks over) -- only when offset + bw exceeds 64
        if (offset + bw > 64 && result_idx + 2 < result.size()) {
            result[result_idx + 2] |= static_cast<uint32_t>(bits >> (64 - offset));
        }

        bit_pos += bw;
    }

    return result;
}

template <size_t bw, size_t N_out> std::vector<int64_t> bitunpack(const std::vector<int32_t> &packed) {
    static_assert(bw > 0 && bw <= 64, "Bit width must be between 1 and 64");

    constexpr size_t total_bits = N_out * bw;
    constexpr size_t packed_size = (total_bits + 31) / 32;
    assert(packed.size() == packed_size);

    std::vector<int64_t> result(N_out, 0);

    for (size_t i = 0; i < N_out; i++) {
        size_t bit_pos = i * bw;
        size_t packed_idx = bit_pos / 32;
        size_t offset = bit_pos % 32;

        // base case
        size_t bw_v0 = std::min<size_t>(bw, 32 - offset);
        uint32_t mask = bw_v0 == 32 ? 0xFFFFFFFF : ((1U << bw_v0) - 1);
        uint64_t value = (static_cast<uint32_t>(packed[packed_idx]) >> offset) & mask;

        // cross boundary (one chunk over)
        if (offset + bw > 32) {
            assert(packed_idx + 1 < packed.size());
            size_t bw_v1 = std::min<size_t>(32, bw - bw_v0);
            uint32_t mask_v1 = bw_v1 == 32 ? 0xFFFFFFFF : ((1U << bw_v1) - 1);
            uint32_t additional_bits = static_cast<uint32_t>(packed[packed_idx + 1]) & mask_v1;
            value |= (static_cast<uint64_t>(additional_bits) << bw_v0);
        }

        // cross boundary (two chunks over)
        if (offset + bw > 64) {
            assert(packed_idx + 2 < packed.size());
            size_t bw_v2 = offset + bw - 64;
            uint32_t mask_v2 = ((1U << bw_v2) - 1);
            uint32_t additional_bits = static_cast<uint32_t>(packed[packed_idx + 2]) & mask_v2;
            value |= (static_cast<uint64_t>(additional_bits) << (bw_v0 + 32));
        }

        result[i] = static_cast<int64_t>(value);
    }

    return result;
}

template <size_t bits_in, typename inp_buf_t>
std::enable_if_t<std::is_integral_v<inp_buf_t>, void>
_write_input(inp_buf_t &inp_buf, const std::vector<int32_t> &input) {
    assert(input.size() == (bits_in + 31) / 32);
    inp_buf = input[0] & 0xFFFFFFFF;
    if (bits_in > 32) {
        inp_buf |= static_cast<int64_t>(input[1]) << 32;
    }
}

template <size_t bits_in, size_t N_in>
void _write_input(VlWide<N_in> &inp_buf, const std::vector<int32_t> &input) {
    assert(input.size() == (bits_in + 31) / 32);
    for (size_t i = 0; i < input.size(); ++i) {
        inp_buf[i] = input[i];
    }
}

template <size_t bits_out, typename out_buf_t>
std::enable_if_t<std::is_integral_v<out_buf_t>, std::vector<int32_t>> _read_output(out_buf_t &out_buf) {
    std::vector<int32_t> output((bits_out + 31) / 32);
    output[0] = out_buf & 0xFFFFFFFF;
    if (bits_out > 32) {
        output[1] = (out_buf >> 32) & 0xFFFFFFFF;
    }
    return output;
}

template <size_t bits_out, size_t N_out> std::vector<int32_t> _read_output(VlWide<N_out> out_buf) {
    std::vector<int32_t> output((bits_out + 31) / 32);
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = out_buf[i] & 0xFFFFFFFF;
    }
    return output;
}

template <size_t N, size_t max_bw, typename inp_buf_t>
void write_input(inp_buf_t &inp_buf, const int64_t *c_inp) {
    constexpr size_t bits_in = N * max_bw;
    std::vector<int32_t> input = bitpack<max_bw, N>(c_inp);
    _write_input<bits_in>(inp_buf, input);
}

template <size_t N, size_t max_bw, typename out_buf_t> void read_output(out_buf_t out_buf, int64_t *c_out) {
    constexpr size_t bits_out = N * max_bw;
    std::vector<int32_t> packed = _read_output<bits_out>(out_buf);
    std::vector<int64_t> unpacked = bitunpack<max_bw, N>(packed);
    for (size_t i = 0; i < N; ++i) {
        c_out[i] = unpacked[i];
    }
}
