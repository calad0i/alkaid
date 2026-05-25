#pragma once

#include "alir_types.hh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace alir {
    namespace {

        template <int B> inline void op_neg(int64_t *out, const int64_t *in0) {
#pragma omp simd
            for (int s = 0; s < B; ++s)
                out[s] = -in0[s];
        }

        template <int B>
        inline void op_input(
            int64_t *out,
            const double *inputs,
            size_t n_in,
            uint32_t input_idx,
            double scale,
            int64_t mask,
            int64_t sign_bit,
            bool out_signed,
            size_t batch_size
        ) {
            for (size_t s = 0; s < batch_size; ++s) {
                double f = inputs[s * n_in + input_idx] * scale;
                int64_t iv = static_cast<int64_t>(std::floor(f));
                int64_t u = iv & mask;
                if (out_signed)
                    u = (u ^ sign_bit) - sign_bit;
                out[s] = u;
            }
            // reminder when batch_size < B
            for (size_t s = batch_size; s < B; ++s)
                out[s] = 0;
        }

        template <int B, bool SUB>
        inline void op_shift_add(
            int64_t *out,
            const int64_t *in0,
            const int64_t *in1,
            int32_t actual_shift_v2,
            int32_t global_shift
        ) {
            if (actual_shift_v2 >= 0) {
                const int32_t sh = actual_shift_v2;
#pragma omp simd
                for (int s = 0; s < B; ++s) {
                    int64_t r = SUB ? (in0[s] - (in1[s] << sh)) : (in0[s] + (in1[s] << sh));
                    r >>= global_shift;
                    out[s] = r;
                }
            }
            else {
                const int32_t sh = -actual_shift_v2;
#pragma omp simd
                for (int s = 0; s < B; ++s) {
                    int64_t r = SUB ? ((in0[s] << sh) - in1[s]) : ((in0[s] << sh) + in1[s]);
                    r >>= global_shift;
                    out[s] = r;
                }
            }
        }

        template <int B>
        inline void op_sum(
            int64_t *out,
            const int64_t *buffer,
            const Op_SumTerm *terms,
            uint32_t n_terms,
            int32_t global_shift
        ) {
            for (int s = 0; s < B; ++s) {
                int64_t acc = 0;
                for (uint32_t i = 0; i < n_terms; ++i) {
                    const Op_SumTerm &term = terms[i];
                    int64_t v = buffer[(size_t)term.addr * B + s];
                    if (term.shift >= 0)
                        v <<= term.shift;
                    else
                        v >>= -term.shift;
                    acc = term.sign > 0 ? acc + v : acc - v;
                }
                out[s] = global_shift >= 0 ? (acc >> global_shift) : (acc << -global_shift);
            }
        }

        inline int64_t
        reduce_lane(int64_t v, int32_t shift, int64_t mask, int64_t sign_bit, bool out_signed) {
            v >>= shift;
            v &= mask;
            if (out_signed)
                v = (v ^ sign_bit) - sign_bit;
            return v;
        }

        template <int B>
        inline void op_relu(
            int64_t *out,
            const int64_t *in0,
            int32_t reduce_shift,
            int64_t mask,
            int64_t sign_bit,
            bool out_signed
        ) {
#pragma omp simd
            for (int s = 0; s < B; ++s) {
                int64_t v = in0[s];
                v = (v < 0) ? 0 : v;
                out[s] = reduce_lane(v, reduce_shift, mask, sign_bit, out_signed);
            }
        }

        template <int B>
        inline void op_quantize(
            int64_t *out,
            const int64_t *in0,
            int32_t reduce_shift,
            int64_t mask,
            int64_t sign_bit,
            bool out_signed
        ) {
#pragma omp simd
            for (int s = 0; s < B; ++s) {
                out[s] = reduce_lane(in0[s], reduce_shift, mask, sign_bit, out_signed);
            }
        }

        template <int B>
        inline void op_const_add(
            int64_t *out,
            const int64_t *in0,
            int32_t actual_shift,
            int32_t global_shift,
            int64_t v2_payload
        ) {
            if (actual_shift >= 0) {
#pragma omp simd
                for (int s = 0; s < B; ++s) {
                    int64_t r = in0[s] + v2_payload;
                    r >>= global_shift;
                    out[s] = r;
                }
            }
            else {
                const int32_t sh = -actual_shift;
#pragma omp simd
                for (int s = 0; s < B; ++s) {
                    int64_t r = (in0[s] << sh) + v2_payload;
                    r >>= global_shift;
                    out[s] = r;
                }
            }
        }

        template <int B> inline void op_const(int64_t *out, int64_t constv) {
#pragma omp simd
            for (int s = 0; s < B; ++s)
                out[s] = constv;
        }

        template <int B>
        inline void op_msb_mux(
            int64_t *out,
            const int64_t *in0,
            const int64_t *in1,
            const int64_t *inc,
            int32_t shift0,
            int32_t shift1,
            int32_t W_cond_minus_1,
            int64_t mask_out,
            int64_t sign_out,
            bool out_signed
        ) {
            for (int s = 0; s < B; ++s) {
                int64_t cond = (inc[s] >> W_cond_minus_1) & 1;
                int64_t pa = in0[s];
                pa <<= shift0;
                int64_t pb = in1[s];
                pb <<= shift1;
                int64_t v = cond ? pa : pb;
                v &= mask_out;
                if (out_signed)
                    v = (v ^ sign_out) - sign_out;
                out[s] = v;
            }
        }

        template <int B> inline void op_mul(int64_t *out, const int64_t *in0, const int64_t *in1) {
#pragma omp simd
            for (int s = 0; s < B; ++s)
                out[s] = in0[s] * in1[s];
        }

        template <int B>
        inline void op_lookup(
            int64_t *out,
            const int64_t *in0,
            const int32_t *table,
            int64_t table_size,
            int64_t off,
            bool allow_oob_lookup
        ) {
            for (int s = 0; s < B; ++s) {
                int64_t index = in0[s] - off;
                if (index < 0 || index >= table_size) {
                    if (allow_oob_lookup) {
                        static thread_local uint32_t lcg = 0x114514u;
                        lcg = lcg * 1664525u + 1013904223u;
                        out[s] = static_cast<int64_t>(static_cast<int32_t>(lcg));
                        continue;
                    }
                    throw std::runtime_error(
                        "Logic lookup index out of bounds: " + std::to_string(index) +
                        " vs table_size=" + std::to_string(table_size)
                    );
                }
                out[s] = static_cast<int64_t>(table[index]);
            }
        }

        template <int B>
        inline void
        op_bit_unary(int64_t *out, const int64_t *in0, int32_t sub_op, int64_t mask_in, bool out_signed) {
            switch (sub_op) {
            case 0: // NOT
                if (out_signed) {
#pragma omp simd
                    for (int s = 0; s < B; ++s)
                        out[s] = ~in0[s];
                }
                else {
#pragma omp simd
                    for (int s = 0; s < B; ++s)
                        out[s] = (~in0[s]) & mask_in;
                }
                break;
            case 1: // REDUCE_OR
#pragma omp simd
                for (int s = 0; s < B; ++s)
                    out[s] = (in0[s] != 0);
                break;
            case 2: // REDUCE_AND
#pragma omp simd
                for (int s = 0; s < B; ++s)
                    out[s] = ((in0[s] & mask_in) == mask_in);
                break;
            default: throw std::runtime_error("Unknown bit unary sub_op=" + std::to_string(sub_op));
            }
        }

        template <int B>
        inline void op_bit_binary(
            int64_t *out,
            const int64_t *in0,
            const int64_t *in1,
            int32_t sh_left_0,
            int32_t sh_left_1,
            int32_t bit_op
        ) {
            switch (bit_op) {
            case 0:
#pragma omp simd
                for (int s = 0; s < B; ++s)
                    out[s] = (in0[s] << sh_left_0) & (in1[s] << sh_left_1);
                break;
            case 1:
#pragma omp simd
                for (int s = 0; s < B; ++s)
                    out[s] = (in0[s] << sh_left_0) | (in1[s] << sh_left_1);
                break;
            case 2:
#pragma omp simd
                for (int s = 0; s < B; ++s)
                    out[s] = (in0[s] << sh_left_0) ^ (in1[s] << sh_left_1);
                break;
            default: throw std::runtime_error("Unknown bit binary bit_op=" + std::to_string(bit_op));
            }
        }

    } // anonymous namespace
} // namespace alir
