#include "ALIRInterpreter.hh"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace alir {

    void ALIRInterpreter::load_from_bytecode(const std::span<const int32_t> &binary_data) {
        if (binary_data.size() < 6) {
            throw std::runtime_error("Binary data too small to contain valid ALIR model file");
        }
        if (binary_data[0] != alir_version) {
            throw std::runtime_error(
                "ALIR version mismatch: expected version " + std::to_string(alir_version) + ", got version " +
                std::to_string(binary_data[0])
            );
        }

        n_in = binary_data[2];
        n_out = binary_data[3];
        n_ops = binary_data[4];
        n_tables = binary_data[5];

        const size_t fixed_offset = 6;
        const size_t table_offset = fixed_offset + n_in + 3 * n_out + 8 * n_ops;

        size_t expect_length = table_offset;
        for (size_t i = 0; i < n_tables; ++i) {
            expect_length += 1 + binary_data[table_offset + i];
        }

        constexpr size_t d_size = sizeof(int32_t);
        if (binary_data.size() != expect_length) {
            throw std::runtime_error(
                "Binary data size mismatch: expected " + std::to_string(expect_length * d_size) +
                " bytes, got " + std::to_string(binary_data.size() * d_size) + " bytes"
            );
        }

        inp_shifts.resize(n_in);
        out_idxs.resize(n_out);
        out_shifts.resize(n_out);
        out_negs.resize(n_out);
        std::vector<Op> ops(n_ops);

        std::memcpy(inp_shifts.data(), &binary_data[fixed_offset], n_in * d_size);
        std::memcpy(out_idxs.data(), &binary_data[fixed_offset + n_in], n_out * d_size);
        std::memcpy(out_shifts.data(), &binary_data[fixed_offset + n_in + n_out], n_out * d_size);
        std::memcpy(out_negs.data(), &binary_data[fixed_offset + n_in + 2 * n_out], n_out * d_size);
        std::memcpy(ops.data(), &binary_data[fixed_offset + n_in + 3 * n_out], n_ops * 8 * d_size);

        size_t curr_table_offset = table_offset + n_tables;
        lookup_tables.clear();
        lookup_tables.reserve(n_tables);
        for (size_t i = 0; i < n_tables; ++i) {
            int32_t table_size = binary_data[table_offset + i];
            std::vector<int32_t> table_data(table_size);
            std::memcpy(table_data.data(), &binary_data[curr_table_offset], table_size * d_size);
            lookup_tables.emplace_back(std::move(table_data));
            curr_table_offset += table_size;
        }

        max_ops_width = 0;
        max_inp_width = 0;
        max_out_width = 0;
        bits_in = 0;
        bits_out = 0;
        for (const Op &op : ops) {
            int32_t width = op.dtype.width();
            if (op.opcode == -1) {
                max_inp_width = std::max(max_inp_width, width);
                bits_in += width;
            }
            max_ops_width = std::max(max_ops_width, width);
        }
        for (int32_t idx : out_idxs) {
            if (idx >= 0) {
                int32_t width = ops[idx].dtype.width();
                max_out_width = std::max(max_out_width, width);
                bits_out += width;
            }
        }

        for (size_t i = 0; i < n_ops; ++i) {
            const Op &op = ops[i];
            if (op.opcode != -1 && op.id0 >= (int32_t)i)
                throw std::runtime_error(
                    "Operation " + std::to_string(i) + " has id0=" + std::to_string(op.id0) +
                    " violating causality"
                );
            if (op.id1 >= (int32_t)i)
                throw std::runtime_error(
                    "Operation " + std::to_string(i) + " has id1=" + std::to_string(op.id1) +
                    " violating causality"
                );
            if (op.opcode == 6 && (size_t)op.data_low >= i)
                throw std::runtime_error(
                    "Operation " + std::to_string(i) + " has cond_idx=" + std::to_string(op.data_low) +
                    " violating causality"
                );
        }
        if (max_ops_width > 64) {
            throw std::runtime_error(
                "ALIR op width " + std::to_string(max_ops_width) +
                " > 64 bits is not representable in the int64 interpreter"
            );
        }

        build_exec_program(ops);
    }

    void ALIRInterpreter::print_program_info() const {
        std::cout << "ALIR Sequence:\n"
                  << n_in << " (" << bits_in << " bits) -> " << n_out << " (" << bits_out << " bits)\n"
                  << "# operations: " << n_ops << "\n"
                  << "# live slots: " << n_slots << "\n"
                  << "Maximum intermediate width: " << max_ops_width << " bits\n";
    }

} // namespace alir
