// Linear-scan allocation + in-place Op materialization. Loaders provide one
// compact Op array plus contiguous addr/data pools; this pass rewrites the same
// Op objects into the execution form.

#include "ALIRInterpreter.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace alir {

    namespace {

        uint32_t
        op_addr(const OpLoad &op, const std::vector<uint32_t> &addr_pool, size_t op_idx, size_t addr_idx) {
            if (addr_idx >= op.n_addr)
                throw std::runtime_error(
                    "Operation " + std::to_string(op_idx) + " opcode " + std::to_string(op.opcode) +
                    " is missing address " + std::to_string(addr_idx)
                );
            return addr_pool[op.addr_offset + addr_idx];
        }

        int64_t
        op_data(const OpLoad &op, const std::vector<int64_t> &data_pool, size_t op_idx, size_t data_idx) {
            if (data_idx >= op.n_data)
                throw std::runtime_error(
                    "Operation " + std::to_string(op_idx) + " opcode " + std::to_string(op.opcode) +
                    " is missing data " + std::to_string(data_idx)
                );
            return data_pool[op.data_offset + data_idx];
        }

        int32_t i64_to_i32(int64_t value, size_t op_idx, const char *field) {
            if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
                throw std::runtime_error(
                    "Operation " + std::to_string(op_idx) + " has out-of-range " + field + " value " +
                    std::to_string(value)
                );
            return static_cast<int32_t>(value);
        }

        uint32_t i64_to_u32(int64_t value, size_t op_idx, const char *field) {
            if (value < 0 || value > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error(
                    "Operation " + std::to_string(op_idx) + " has out-of-range " + field + " value " +
                    std::to_string(value)
                );
            return static_cast<uint32_t>(value);
        }

    } // anonymous namespace

    void ALIRInterpreter::build_exec_program(
        std::vector<Op> program,
        const std::vector<uint32_t> &addr_pool,
        const std::vector<int64_t> &data_pool,
        const std::vector<DType> &dtypes
    ) {
        std::vector<int32_t> last_use(n_ops, -1);
        for (size_t i = 0; i < n_ops; ++i) {
            const OpLoad &op = program[i].load;
            for (uint16_t j = 0; j < op.n_addr; ++j)
                last_use[addr_pool[op.addr_offset + j]] = static_cast<int32_t>(i);
        }
        const int32_t PIN = static_cast<int32_t>(n_ops);
        for (int32_t oi : out_idxs) {
            if (oi >= 0)
                last_use[oi] = PIN;
        }

        // WAR reuse: write to own input slot is allowed.
        std::vector<std::vector<uint32_t>> release_at(n_ops);
        for (size_t j = 0; j < n_ops; ++j) {
            int32_t lu = last_use[j];
            if (lu >= 0 && lu < PIN)
                release_at[lu].push_back(static_cast<uint32_t>(j));
        }

        std::vector<uint32_t> free_stack;
        free_stack.reserve(1024);
        uint32_t next_slot = 0;
        sum_terms.clear();
        sum_terms.reserve(addr_pool.size());
        input_scales.assign(n_ops, 0.0);

        for (size_t i = 0; i < n_ops; ++i) {
            for (uint32_t j : release_at[i])
                free_stack.push_back(program[j].h.addr_out);

            uint32_t addr_out;
            if (!free_stack.empty()) {
                addr_out = free_stack.back();
                free_stack.pop_back();
            }
            else {
                addr_out = next_slot++;
            }

            Op &op = program[i];
            const OpLoad raw = op.load;
            const DType &d_out = dtypes[i];
            const uint8_t W_out = static_cast<uint8_t>(d_out.width());
            const bool out_signed = d_out.is_signed != 0;
            const uint8_t flag_signed = out_signed ? 0x1 : 0x0;

            switch (raw.opcode) {
            case -2: {
                const uint32_t x = op_addr(raw, addr_pool, i, 0);
                op.neg = Op_Neg{};
                op.neg.h.opcode = -2;
                op.neg.h.addr_out = addr_out;
                op.neg.a0 = program[x].h.addr_out;
                break;
            }
            case -1: {
                const uint32_t input_idx = i64_to_u32(op_data(raw, data_pool, i, 0), i, "input_idx");
                if (input_idx >= n_in)
                    throw std::runtime_error(
                        "Operation " + std::to_string(i) + " has out-of-range input index"
                    );
                op.input = Op_Input{};
                op.input.h.opcode = -1;
                op.input.h.flags = flag_signed;
                op.input.h.w_out = W_out;
                op.input.h.addr_out = addr_out;
                op.input.input_idx = input_idx;
                input_scales[i] = std::ldexp(1.0, inp_shifts[input_idx] + d_out.fractionals);
                break;
            }
            case 0:
            case 1: {
                const uint32_t a = op_addr(raw, addr_pool, i, 0);
                const uint32_t b = op_addr(raw, addr_pool, i, 1);
                const int32_t shift = i64_to_i32(op_data(raw, data_pool, i, 0), i, "shift");
                const DType &d0 = dtypes[a];
                const DType &d1 = dtypes[b];
                const int32_t actual = shift + d0.fractionals - d1.fractionals;
                const int32_t global = std::max(d0.fractionals, d1.fractionals - shift) - d_out.fractionals;
                op.add_sub = Op_AddSub{};
                op.add_sub.h.opcode = static_cast<int8_t>(raw.opcode);
                op.add_sub.h.addr_out = addr_out;
                op.add_sub.a0 = program[a].h.addr_out;
                op.add_sub.a1 = program[b].h.addr_out;
                op.add_sub.actual_shift_v2 = static_cast<int8_t>(actual);
                op.add_sub.global_shift = static_cast<int8_t>(global);
                break;
            }
            case 11: {
                assert(raw.n_addr >= 2);
                int32_t common_fractionals = std::numeric_limits<int32_t>::min();
                for (uint16_t j = 0; j < raw.n_addr; ++j) {
                    const uint32_t a = op_addr(raw, addr_pool, i, j);
                    const int32_t shift = i64_to_i32(op_data(raw, data_pool, i, 2 * j + 1), i, "shift");
                    common_fractionals = std::max(common_fractionals, dtypes[a].fractionals - shift);
                }
                const uint32_t term_offset = static_cast<uint32_t>(sum_terms.size());
                for (uint16_t j = 0; j < raw.n_addr; ++j) {
                    const uint32_t a = op_addr(raw, addr_pool, i, j);
                    const bool plus = op_data(raw, data_pool, i, 2 * j) != 0;
                    const int32_t shift = i64_to_i32(op_data(raw, data_pool, i, 2 * j + 1), i, "shift");
                    sum_terms.push_back(
                        Op_SumTerm{
                            program[a].h.addr_out,
                            common_fractionals - (dtypes[a].fractionals - shift),
                            static_cast<int8_t>(plus ? 1 : -1),
                            {0, 0, 0},
                        }
                    );
                }
                op.sum = Op_Sum{};
                op.sum.h.opcode = 11;
                op.sum.h.addr_out = addr_out;
                op.sum.n_terms = raw.n_addr;
                op.sum.global_shift = common_fractionals - d_out.fractionals;
                op.sum.terms = sum_terms.data() + term_offset;
                break;
            }
            case 2:
            case 3: {
                const uint32_t x = op_addr(raw, addr_pool, i, 0);
                const DType &d0 = dtypes[x];
                const int8_t relative_shift =
                    raw.opcode == 3 ? static_cast<int8_t>(op_data(raw, data_pool, i, 0)) : 0;
                const int32_t reduce = d0.fractionals - relative_shift - d_out.fractionals;
                op.reduce = Op_ReluQuant{};
                op.reduce.h.opcode = static_cast<int8_t>(raw.opcode);
                op.reduce.h.flags = flag_signed;
                op.reduce.h.w_out = W_out;
                op.reduce.h.addr_out = addr_out;
                op.reduce.a0 = program[x].h.addr_out;
                op.reduce.reduce_shift = static_cast<int8_t>(reduce);
                break;
            }
            case 4: {
                const uint32_t x = op_addr(raw, addr_pool, i, 0);
                const int32_t const_shift = i64_to_i32(op_data(raw, data_pool, i, 1), i, "const_shift");
                const DType &d0 = dtypes[x];
                const int32_t actual = -const_shift + d0.fractionals;
                const int32_t global = std::max(d0.fractionals, const_shift) - d_out.fractionals;
                const int64_t v2 = op_data(raw, data_pool, i, 0);
                const int64_t payload = (actual >= 0) ? (v2 << actual) : v2;
                op.const_add = Op_ConstAdd{};
                op.const_add.h.opcode = 4;
                op.const_add.h.addr_out = addr_out;
                op.const_add.a0 = program[x].h.addr_out;
                op.const_add.actual_shift = static_cast<int8_t>(actual);
                op.const_add.global_shift = static_cast<int8_t>(global);
                op.const_add.v2_payload = payload;
                break;
            }
            case 5: {
                op.constant = Op_Const{};
                op.constant.h.opcode = 5;
                op.constant.h.addr_out = addr_out;
                op.constant.const_val = op_data(raw, data_pool, i, 0);
                break;
            }
            case 6: {
                const uint32_t true_idx = op_addr(raw, addr_pool, i, 0);
                const uint32_t false_idx = op_addr(raw, addr_pool, i, 1);
                const uint32_t cond_idx = op_addr(raw, addr_pool, i, 2);
                const int32_t false_shift = i64_to_i32(op_data(raw, data_pool, i, 0), i, "false_shift");
                const DType &d0 = dtypes[true_idx];
                const DType &d1 = dtypes[false_idx];
                const DType &dc = dtypes[cond_idx];
                const int32_t shift0 = d_out.fractionals - d0.fractionals;
                const int32_t shift1 = d_out.fractionals - d1.fractionals + false_shift;
                if (shift0 != 0 && shift1 != 0) {
                    throw std::runtime_error(
                        "Unsupported msb_mux shift configuration at op " + std::to_string(i) +
                        ": shift0=" + std::to_string(shift0) + ", shift1=" + std::to_string(shift1)
                    );
                }
                op.mux = Op_MsbMux{};
                op.mux.h.opcode = 6;
                op.mux.h.flags = flag_signed;
                op.mux.h.w_out = W_out;
                op.mux.h.w_in = static_cast<uint8_t>(dc.width());
                op.mux.h.addr_out = addr_out;
                op.mux.a0 = program[true_idx].h.addr_out;
                op.mux.a1 = program[false_idx].h.addr_out;
                op.mux.cond = program[cond_idx].h.addr_out;
                op.mux.shift0 = static_cast<int8_t>(shift0);
                op.mux.shift1 = static_cast<int8_t>(shift1);
                break;
            }
            case 7: {
                const uint32_t a = op_addr(raw, addr_pool, i, 0);
                const uint32_t b = op_addr(raw, addr_pool, i, 1);
                op.mul = Op_Mul{};
                op.mul.h.opcode = 7;
                op.mul.h.addr_out = addr_out;
                op.mul.a0 = program[a].h.addr_out;
                op.mul.a1 = program[b].h.addr_out;
                break;
            }
            case 8: {
                const uint32_t x = op_addr(raw, addr_pool, i, 0);
                const int64_t table_idx64 = op_data(raw, data_pool, i, 0);
                if (table_idx64 < 0 || table_idx64 >= static_cast<int64_t>(n_tables))
                    throw std::runtime_error(
                        "Operation " + std::to_string(i) + " has out-of-range lookup table index"
                    );
                const DType &d0 = dtypes[x];
                op.lookup = Op_Lookup{};
                op.lookup.h.opcode = 8;
                op.lookup.h.flags = (d0.is_signed != 0) ? 0x1 : 0x0;
                op.lookup.h.w_in = static_cast<uint8_t>(d0.width());
                op.lookup.h.addr_out = addr_out;
                op.lookup.a0 = program[x].h.addr_out;
                op.lookup.table_idx = static_cast<uint32_t>(table_idx64);
                op.lookup.data_high = i64_to_i32(op_data(raw, data_pool, i, 1), i, "lookup_pad");
                break;
            }
            case 9: {
                const uint32_t x = op_addr(raw, addr_pool, i, 0);
                const DType &d0 = dtypes[x];
                op.bit_un = Op_BitUnary{};
                op.bit_un.h.opcode = 9;
                op.bit_un.h.flags = flag_signed;
                op.bit_un.h.w_in = static_cast<uint8_t>(d0.width());
                op.bit_un.h.addr_out = addr_out;
                op.bit_un.a0 = program[x].h.addr_out;
                op.bit_un.sub_op = static_cast<int8_t>(op_data(raw, data_pool, i, 0));
                break;
            }
            case 10: {
                const uint32_t a = op_addr(raw, addr_pool, i, 0);
                const uint32_t b = op_addr(raw, addr_pool, i, 1);
                const int32_t shift = i64_to_i32(op_data(raw, data_pool, i, 0), i, "shift");
                const DType &d0 = dtypes[a];
                const DType &d1 = dtypes[b];
                const int32_t actual = shift + d0.fractionals - d1.fractionals;
                const int32_t shl_a = (actual < 0) ? -actual : 0;
                const int32_t shl_b = (actual > 0) ? actual : 0;
                op.bit_bin = Op_BitBinary{};
                op.bit_bin.h.opcode = 10;
                op.bit_bin.h.addr_out = addr_out;
                op.bit_bin.a0 = program[a].h.addr_out;
                op.bit_bin.a1 = program[b].h.addr_out;
                op.bit_bin.shl_a = static_cast<int8_t>(shl_a);
                op.bit_bin.shl_b = static_cast<int8_t>(shl_b);
                op.bit_bin.bit_op = static_cast<int8_t>(op_data(raw, data_pool, i, 1));
                break;
            }
            default:
                throw std::runtime_error("build_exec_program: unknown opcode " + std::to_string(raw.opcode));
            }
        }
        n_slots = next_slot;

        out_idxs_slot.assign(n_out, -1);
        output_scales.assign(n_out, 0.0);
        for (size_t j = 0; j < n_out; ++j) {
            int32_t oi = out_idxs[j];
            if (oi >= 0) {
                out_idxs_slot[j] = static_cast<int32_t>(program[oi].h.addr_out);
                output_scales[j] = std::ldexp(1.0, out_shifts[j] - dtypes[oi].fractionals);
            }
        }

        op_dump_scales.assign(n_ops, 0.0);
        for (size_t i = 0; i < n_ops; ++i)
            op_dump_scales[i] = std::ldexp(1.0, -dtypes[i].fractionals);

        ops = std::move(program);
    }

} // namespace alir
