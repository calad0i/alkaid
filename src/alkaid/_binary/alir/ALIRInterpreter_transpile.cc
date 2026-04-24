// Linear-scan allocation + OpExec materialization. Two passes: scan
// last-use per op, then forward-sweep releasing/allocating slots and
// writing per-opcode union variants.

#include "ALIRInterpreter.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace alir {

    void ALIRInterpreter::build_exec_program(const std::vector<Op> &ops) {
        std::vector<int32_t> last_use(n_ops, -1);
        for (size_t i = 0; i < n_ops; ++i) {
            const Op &op = ops[i];
            switch (op.opcode) {
            case -1:
            case 5: break;
            case -2:
            case 2:
            case 3:
            case 4:
            case 8:
            case 9: last_use[op.id0] = (int32_t)i; break;
            case 0:
            case 1:
            case 7:
            case 10:
                last_use[op.id0] = (int32_t)i;
                last_use[op.id1] = (int32_t)i;
                break;
            case 6:
                last_use[op.id0] = (int32_t)i;
                last_use[op.id1] = (int32_t)i;
                last_use[op.data_low] = (int32_t)i;
                break;
            default:
                throw std::runtime_error("build_exec_program: unknown opcode " + std::to_string(op.opcode));
            }
        }
        const int32_t PIN = (int32_t)n_ops;
        for (int32_t oi : out_idxs) {
            if (oi >= 0)
                last_use[oi] = PIN;
        }

        // WAR reuse: write to own input slot is allowed
        std::vector<std::vector<int32_t>> release_at(n_ops);
        for (size_t j = 0; j < n_ops; ++j) {
            int32_t lu = last_use[j];
            if (lu < PIN)
                release_at[lu].push_back((int32_t)j);
        }

        op_out_addr.assign(n_ops, -1);
        std::vector<int32_t> free_stack;
        free_stack.reserve(1024);
        int32_t next_slot = 0;
        ops_exec.assign(n_ops, OpExec{});
        input_scales.assign(n_ops, 0.0);

        for (size_t i = 0; i < n_ops; ++i) {
            for (int32_t j : release_at[i]) {
                if (op_out_addr[j] >= 0)
                    free_stack.push_back(op_out_addr[j]);
            }
            int32_t addr;
            if (!free_stack.empty()) {
                addr = free_stack.back();
                free_stack.pop_back();
            }
            else {
                addr = next_slot++;
            }
            op_out_addr[i] = addr;

            const Op &op = ops[i];
            OpExec &ex = ops_exec[i];
            const DType &d_out = op.dtype;
            const uint8_t W_out = static_cast<uint8_t>(d_out.width());
            const bool out_signed = d_out.is_signed != 0;
            const uint8_t flag_signed = out_signed ? 0x1 : 0x0;

            switch (op.opcode) {
            case -2: {
                ex.neg = Op_Neg{};
                ex.neg.h.opcode = -2;
                ex.neg.h.out_addr = addr;
                ex.neg.a0 = op_out_addr[op.id0];
                break;
            }
            case -1: {
                ex.input = Op_Input{};
                ex.input.h.opcode = -1;
                ex.input.h.flags = flag_signed;
                ex.input.h.w_out = W_out;
                ex.input.h.out_addr = addr;
                ex.input.input_idx = op.id0;
                input_scales[i] = std::ldexp(1.0, inp_shifts[op.id0] + d_out.fractionals);
                break;
            }
            case 0:
            case 1: {
                const DType &d0 = ops[op.id0].dtype;
                const DType &d1 = ops[op.id1].dtype;
                const int32_t actual = op.data_low + d0.fractionals - d1.fractionals;
                const int32_t global =
                    std::max(d0.fractionals, d1.fractionals - op.data_low) - d_out.fractionals;
                ex.add_sub = Op_AddSub{};
                ex.add_sub.h.opcode = static_cast<int8_t>(op.opcode);
                ex.add_sub.h.out_addr = addr;
                ex.add_sub.a0 = op_out_addr[op.id0];
                ex.add_sub.a1 = op_out_addr[op.id1];
                ex.add_sub.actual_shift_v2 = static_cast<int8_t>(actual);
                ex.add_sub.global_shift = static_cast<int8_t>(global);
                break;
            }
            case 2:
            case 3: {
                const DType &d0 = ops[op.id0].dtype;
                const int32_t reduce = d0.fractionals - d_out.fractionals;
                ex.reduce = Op_ReluQuant{};
                ex.reduce.h.opcode = static_cast<int8_t>(op.opcode);
                ex.reduce.h.flags = flag_signed;
                ex.reduce.h.w_out = W_out;
                ex.reduce.h.out_addr = addr;
                ex.reduce.a0 = op_out_addr[op.id0];
                ex.reduce.reduce_shift = static_cast<int8_t>(reduce);
                break;
            }
            case 4: {
                const DType &d0 = ops[op.id0].dtype;
                const int32_t actual = -op.data_high + d0.fractionals;
                const int32_t global = std::max(d0.fractionals, op.data_high) - d_out.fractionals;
                const int64_t v2 = op.data_low;
                const int64_t payload = (actual >= 0) ? (v2 << actual) : v2;
                ex.const_add = Op_ConstAdd{};
                ex.const_add.h.opcode = 4;
                ex.const_add.h.out_addr = addr;
                ex.const_add.a0 = op_out_addr[op.id0];
                ex.const_add.actual_shift = static_cast<int8_t>(actual);
                ex.const_add.global_shift = static_cast<int8_t>(global);
                ex.const_add.v2_payload = payload;
                break;
            }
            case 5: {
                ex.constant = Op_Const{};
                ex.constant.h.opcode = 5;
                ex.constant.h.out_addr = addr;
                ex.constant.const_val =
                    (static_cast<int64_t>(op.data_high) << 32) | static_cast<uint32_t>(op.data_low);
                break;
            }
            case 6: {
                const DType &d0 = ops[op.id0].dtype;
                const DType &d1 = ops[op.id1].dtype;
                const DType &dc = ops[op.data_low].dtype;
                const int32_t shift0 = d_out.fractionals - d0.fractionals;
                const int32_t shift1 = d_out.fractionals - d1.fractionals + op.data_high;
                if (shift0 != 0 && shift1 != 0) {
                    throw std::runtime_error(
                        "Unsupported msb_mux shift configuration at op " + std::to_string(i) +
                        ": shift0=" + std::to_string(shift0) + ", shift1=" + std::to_string(shift1)
                    );
                }
                ex.mux = Op_MsbMux{};
                ex.mux.h.opcode = 6;
                ex.mux.h.flags = flag_signed;
                ex.mux.h.w_out = W_out;
                ex.mux.h.w_in = static_cast<uint8_t>(dc.width());
                ex.mux.h.out_addr = addr;
                ex.mux.a0 = op_out_addr[op.id0];
                ex.mux.a1 = op_out_addr[op.id1];
                ex.mux.cond = op_out_addr[op.data_low];
                ex.mux.shift0 = static_cast<int8_t>(shift0);
                ex.mux.shift1 = static_cast<int8_t>(shift1);
                break;
            }
            case 7: {
                ex.mul = Op_Mul{};
                ex.mul.h.opcode = 7;
                ex.mul.h.out_addr = addr;
                ex.mul.a0 = op_out_addr[op.id0];
                ex.mul.a1 = op_out_addr[op.id1];
                break;
            }
            case 8: {
                const DType &d0 = ops[op.id0].dtype;
                ex.lookup = Op_Lookup{};
                ex.lookup.h.opcode = 8;
                ex.lookup.h.flags = (d0.is_signed != 0) ? 0x1 : 0x0;
                ex.lookup.h.w_in = static_cast<uint8_t>(d0.width());
                ex.lookup.h.out_addr = addr;
                ex.lookup.a0 = op_out_addr[op.id0];
                ex.lookup.table_idx = op.data_low;
                ex.lookup.data_high = op.data_high;
                break;
            }
            case 9: {
                const DType &d0 = ops[op.id0].dtype;
                ex.bit_un = Op_BitUnary{};
                ex.bit_un.h.opcode = 9;
                ex.bit_un.h.flags = flag_signed;
                ex.bit_un.h.w_in = static_cast<uint8_t>(d0.width());
                ex.bit_un.h.out_addr = addr;
                ex.bit_un.a0 = op_out_addr[op.id0];
                ex.bit_un.sub_op = static_cast<int8_t>(op.data_low);
                break;
            }
            case 10: {
                const DType &d0 = ops[op.id0].dtype;
                const DType &d1 = ops[op.id1].dtype;
                const int32_t actual = op.data_low + d0.fractionals - d1.fractionals;
                const int32_t shl_a = (actual < 0) ? -actual : 0;
                const int32_t shl_b = (actual > 0) ? actual : 0;
                ex.bit_bin = Op_BitBinary{};
                ex.bit_bin.h.opcode = 10;
                ex.bit_bin.h.out_addr = addr;
                ex.bit_bin.a0 = op_out_addr[op.id0];
                ex.bit_bin.a1 = op_out_addr[op.id1];
                ex.bit_bin.shl_a = static_cast<int8_t>(shl_a);
                ex.bit_bin.shl_b = static_cast<int8_t>(shl_b);
                ex.bit_bin.bit_op = static_cast<int8_t>(op.data_high >> 24);
                break;
            }
            }
        }
        n_slots = (size_t)next_slot;

        out_idxs_slot.assign(n_out, -1);
        output_scales.assign(n_out, 0.0);
        for (size_t j = 0; j < n_out; ++j) {
            int32_t oi = out_idxs[j];
            if (oi >= 0) {
                out_idxs_slot[j] = op_out_addr[oi];
                output_scales[j] = std::ldexp(1.0, out_shifts[j] - ops[oi].dtype.fractionals);
            }
        }

        op_dump_scales.assign(n_ops, 0.0);
        for (size_t i = 0; i < n_ops; ++i) {
            op_dump_scales[i] = std::ldexp(1.0, -ops[i].dtype.fractionals);
        }
    }

} // namespace alir
