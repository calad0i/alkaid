#pragma once

#include <cstdint>
#include <stdalign.h>
#include <type_traits>

namespace alir {

    struct DType {
        int32_t is_signed;
        int32_t integers;
        int32_t fractionals;

        int32_t width() const { return integers + fractionals + (is_signed ? 1 : 0); }
        int64_t int_max() const { return (1ll << (width() - (is_signed ? 1 : 0))) - 1; }
        int64_t int_min() const { return is_signed ? -(1ll << (width() - 1)) : 0; }

        DType operator<<(int32_t shift) const {
            return DType{is_signed, integers + shift, fractionals - shift};
        }
        DType with_fractionals(int32_t new_fractionals) const {
            return DType{is_signed, integers + (fractionals - new_fractionals), new_fractionals};
        }
    };

    struct alignas(4) Op {
        int32_t opcode;
        int32_t id0;
        int32_t id1;
        int32_t data_low;
        int32_t data_high;
        DType dtype;
    };

    // flags bit 0: signedness (out for -1/2/3/6/9, input for 8).
    // w_out / w_in are 1..64 when used, 0 otherwise.
    struct OpHeader {
        int8_t opcode;
        uint8_t flags;
        uint8_t w_out;
        uint8_t w_in;
        int32_t out_addr;
    };
    static_assert(sizeof(OpHeader) == 8, "OpHeader must be 8 bytes");
    static_assert(std::is_standard_layout_v<OpHeader>);
    static_assert(std::is_trivial_v<OpHeader>);

    struct alignas(8) Op_Neg {
        OpHeader h;
        int32_t a0;
        int32_t _pad[3];
    };

    struct alignas(8) Op_Input {
        OpHeader h;
        int32_t input_idx;
        int32_t _pad[3];
    };

    struct alignas(8) Op_AddSub {
        OpHeader h;
        int32_t a0;
        int32_t a1;
        int8_t actual_shift_v2;
        int8_t global_shift;
        int16_t _pad0;
        int32_t _pad1;
    };

    struct alignas(8) Op_ReluQuant {
        OpHeader h;
        int32_t a0;
        int8_t reduce_shift;
        uint8_t _pad[11];
    };

    // v2_payload holds `v2 << actual_shift` when actual_shift >= 0, else raw v2.
    struct Op_ConstAdd {
        OpHeader h;
        int32_t a0;
        int8_t actual_shift;
        int8_t global_shift;
        int16_t _pad;
        int64_t v2_payload;
    };

    struct Op_Const {
        OpHeader h;
        int32_t _pad;
        int64_t const_val;
    };

    // Load-time precondition: shift0 == 0 || shift1 == 0.
    struct alignas(8) Op_MsbMux {
        OpHeader h;
        int32_t a0;
        int32_t a1;
        int32_t cond;
        int8_t shift0;
        int8_t shift1;
        int16_t _pad;
    };

    struct alignas(8) Op_Mul {
        OpHeader h;
        int32_t a0;
        int32_t a1;
        int32_t _pad[2];
    };

    struct alignas(8) Op_Lookup {
        OpHeader h;
        int32_t a0;
        int32_t table_idx;
        int32_t data_high;
        int32_t _pad;
    };

    // sub_op: 0=NOT, 1=REDUCE_OR, 2=REDUCE_AND.
    struct alignas(8) Op_BitUnary {
        OpHeader h;
        int32_t a0;
        int8_t sub_op;
        uint8_t _pad[11];
    };

    // bit_op: 0=AND, 1=OR, 2=XOR.
    struct alignas(8) Op_BitBinary {
        OpHeader h;
        int32_t a0;
        int32_t a1;
        int8_t shl_a;
        int8_t shl_b;
        int8_t bit_op;
        uint8_t _pad[5];
    };

    static_assert(sizeof(Op_Neg) == 24 && alignof(Op_Neg) == 8);
    static_assert(sizeof(Op_Input) == 24 && alignof(Op_Input) == 8);
    static_assert(sizeof(Op_AddSub) == 24 && alignof(Op_AddSub) == 8);
    static_assert(sizeof(Op_ReluQuant) == 24 && alignof(Op_ReluQuant) == 8);
    static_assert(sizeof(Op_ConstAdd) == 24 && alignof(Op_ConstAdd) == 8);
    static_assert(sizeof(Op_Const) == 24 && alignof(Op_Const) == 8);
    static_assert(sizeof(Op_MsbMux) == 24 && alignof(Op_MsbMux) == 8);
    static_assert(sizeof(Op_Mul) == 24 && alignof(Op_Mul) == 8);
    static_assert(sizeof(Op_Lookup) == 24 && alignof(Op_Lookup) == 8);
    static_assert(sizeof(Op_BitUnary) == 24 && alignof(Op_BitUnary) == 8);
    static_assert(sizeof(Op_BitBinary) == 24 && alignof(Op_BitBinary) == 8);
    static_assert(std::is_trivial_v<Op_AddSub>);
    static_assert(std::is_trivial_v<Op_MsbMux>);

    union OpExec {
        OpHeader h;
        Op_Neg neg;
        Op_Input input;
        Op_AddSub add_sub;
        Op_ReluQuant reduce;
        Op_ConstAdd const_add;
        Op_Const constant;
        Op_MsbMux mux;
        Op_Mul mul;
        Op_Lookup lookup;
        Op_BitUnary bit_un;
        Op_BitBinary bit_bin;
    };
    static_assert(sizeof(OpExec) == 24, "OpExec must be 24 bytes");
    static_assert(alignof(OpExec) == 8);
    static_assert(std::is_trivial_v<OpExec>);

    // W >= 1 guaranteed by validate(); W == 64 handled by the ternary.
    inline int64_t mask_from(uint8_t W) { return (W == 64) ? ~int64_t(0) : ((int64_t(1) << W) - 1); }
    inline int64_t sign_from(uint8_t W) { return int64_t(1) << (W - 1); }

} // namespace alir
