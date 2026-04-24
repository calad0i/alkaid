#include "ALIRInterpreter.hh"
#include "alir_kernels.hh"

namespace alir {

    template <int B>
    void ALIRInterpreter::exec_batch_core(const double *inputs, size_t batch_size, int64_t *buffer) const {
        const OpExec *ops_ptr = ops_exec.data();
        const double *scales_ptr = input_scales.data();
        const size_t nin = n_in;

        for (size_t i = 0; i < n_ops; ++i) {
            const OpExec &op = ops_ptr[i];

            switch (op.h.opcode) {
            case -2: {
                const auto &_op = op.neg;
                op_neg<B>(buffer + (size_t)_op.h.out_addr * B, buffer + (size_t)_op.a0 * B);
                break;
            }
            case -1: {
                const auto &_op = op.input;
                op_input<B>(
                    buffer + (size_t)_op.h.out_addr * B,
                    inputs,
                    nin,
                    _op.input_idx,
                    scales_ptr[i],
                    mask_from(_op.h.w_out),
                    sign_from(_op.h.w_out),
                    (_op.h.flags & 0x1) != 0,
                    batch_size
                );
                break;
            }
            case 0: {
                const auto &_op = op.add_sub;
                op_shift_add<B, false>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    buffer + (size_t)_op.a1 * B,
                    _op.actual_shift_v2,
                    _op.global_shift
                );
                break;
            }
            case 1: {
                const auto &_op = op.add_sub;
                op_shift_add<B, true>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    buffer + (size_t)_op.a1 * B,
                    _op.actual_shift_v2,
                    _op.global_shift
                );
                break;
            }
            case 2: {
                const auto &_op = op.reduce;
                op_relu<B>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    _op.reduce_shift,
                    mask_from(_op.h.w_out),
                    sign_from(_op.h.w_out),
                    (_op.h.flags & 0x1) != 0
                );
                break;
            }
            case 3: {
                const auto &_op = op.reduce;
                op_quantize<B>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    _op.reduce_shift,
                    mask_from(_op.h.w_out),
                    sign_from(_op.h.w_out),
                    (_op.h.flags & 0x1) != 0
                );
                break;
            }
            case 4: {
                const auto &_op = op.const_add;
                op_const_add<B>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    _op.actual_shift,
                    _op.global_shift,
                    _op.v2_payload
                );
                break;
            }
            case 5: {
                const auto &_op = op.constant;
                op_const<B>(buffer + (size_t)_op.h.out_addr * B, _op.const_val);
                break;
            }
            case 6: {
                const auto &_op = op.mux;
                op_msb_mux<B>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    buffer + (size_t)_op.a1 * B,
                    buffer + (size_t)_op.cond * B,
                    _op.shift0,
                    _op.shift1,
                    /*W_cond_minus_1=*/_op.h.w_in - 1,
                    mask_from(_op.h.w_out),
                    sign_from(_op.h.w_out),
                    (_op.h.flags & 0x1) != 0
                );
                break;
            }
            case 7: {
                const auto &_op = op.mul;
                op_mul<B>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    buffer + (size_t)_op.a1 * B
                );
                break;
            }
            case 8: {
                const auto &_op = op.lookup;
                // `zero = -input_signed * 2^(W_in-1)`; fold with data_high
                // into `off`. Cheap — a few ALU ops per op, not per sample.
                const int64_t sign = ((_op.h.flags & 0x1) != 0) ? (int64_t(1) << (_op.h.w_in - 1)) : 0;
                const int64_t off = -sign + _op.data_high;
                const auto &table = lookup_tables[_op.table_idx];
                op_lookup<B>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    table.data(),
                    (int64_t)table.size(),
                    off
                );
                break;
            }
            case 9: {
                const auto &_op = op.bit_un;
                op_bit_unary<B>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    _op.sub_op,
                    mask_from(_op.h.w_in),
                    (_op.h.flags & 0x1) != 0
                );
                break;
            }
            case 10: {
                const auto &_op = op.bit_bin;
                op_bit_binary<B>(
                    buffer + (size_t)_op.h.out_addr * B,
                    buffer + (size_t)_op.a0 * B,
                    buffer + (size_t)_op.a1 * B,
                    _op.shl_a,
                    _op.shl_b,
                    _op.bit_op
                );
                break;
            }
            default:
                throw std::runtime_error(
                    "exec_batch_core: unknown opcode " + std::to_string(op.h.opcode) + " at index " +
                    std::to_string(i)
                );
            }
        }
    }

    template <int B>
    void ALIRInterpreter::exec_batch(
        const double *inputs,
        double *outputs,
        size_t batch_size,
        int64_t *buffer
    ) const {
        exec_batch_core<B>(inputs, batch_size, buffer);

        const size_t nout = n_out;
        for (size_t j = 0; j < nout; ++j) {
            const int32_t os = out_idxs_slot[j];
            if (os < 0) {
                for (size_t s = 0; s < batch_size; ++s)
                    outputs[s * nout + j] = 0.0;
                continue;
            }
            const int64_t *in_slot = buffer + (size_t)os * B;
            const double scale = output_scales[j];
            const bool neg = out_negs[j] != 0;
            if (neg) {
                for (size_t s = 0; s < batch_size; ++s)
                    outputs[s * nout + j] = static_cast<double>(-in_slot[s]) * scale;
            }
            else {
                for (size_t s = 0; s < batch_size; ++s)
                    outputs[s * nout + j] = static_cast<double>(in_slot[s]) * scale;
            }
        }
    }

    template <int B>
    void ALIRInterpreter::dump_batch(
        const double *inputs,
        double *dump_outputs,
        size_t batch_size,
        int64_t *buffer
    ) const {
        exec_batch_core<B>(inputs, batch_size, buffer);

        const size_t nops = n_ops;
        for (size_t i = 0; i < nops; ++i) {
            const int64_t *slot = buffer + (size_t)op_out_addr[i] * B;
            const double scale = op_dump_scales[i];
            for (size_t s = 0; s < batch_size; ++s) {
                dump_outputs[s * nops + i] = static_cast<double>(slot[s]) * scale;
            }
        }
    }

    // Explicit instantiations — B choices picked by bindings.cc via
    // pick_auto_config.
    template void ALIRInterpreter::exec_batch<4>(const double *, double *, size_t, int64_t *) const;
    template void ALIRInterpreter::exec_batch<8>(const double *, double *, size_t, int64_t *) const;
    template void ALIRInterpreter::exec_batch<16>(const double *, double *, size_t, int64_t *) const;
    template void ALIRInterpreter::dump_batch<4>(const double *, double *, size_t, int64_t *) const;
    template void ALIRInterpreter::dump_batch<8>(const double *, double *, size_t, int64_t *) const;
    template void ALIRInterpreter::dump_batch<16>(const double *, double *, size_t, int64_t *) const;

} // namespace alir
