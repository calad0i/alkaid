#include "ALIRInterpreter.hh"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace alir {

    namespace {

        uint64_t read_u_le(std::span<const uint8_t> bytes, size_t &pos, size_t n) {
            if (bytes.size() - pos < n)
                throw std::runtime_error("Unexpected EOF while parsing ALIR bytecode");
            uint64_t v = 0;
            for (size_t i = 0; i < n; ++i)
                v |= uint64_t(bytes[pos++]) << (8 * i);
            return v;
        }

        uint8_t read_u8(std::span<const uint8_t> bytes, size_t &pos) {
            return static_cast<uint8_t>(read_u_le(bytes, pos, 1));
        }
        int8_t read_i8(std::span<const uint8_t> bytes, size_t &pos) {
            return static_cast<int8_t>(read_u8(bytes, pos));
        }
        uint16_t read_u16(std::span<const uint8_t> bytes, size_t &pos) {
            return static_cast<uint16_t>(read_u_le(bytes, pos, 2));
        }
        uint32_t read_u32(std::span<const uint8_t> bytes, size_t &pos) {
            return static_cast<uint32_t>(read_u_le(bytes, pos, 4));
        }
        int32_t read_i32(std::span<const uint8_t> bytes, size_t &pos) {
            return static_cast<int32_t>(read_u32(bytes, pos));
        }
        int64_t read_i64(std::span<const uint8_t> bytes, size_t &pos) {
            return static_cast<int64_t>(read_u_le(bytes, pos, 8));
        }

        void expect_magic(std::span<const uint8_t> bytes, size_t &pos) {
            if (bytes.size() - pos < 4)
                throw std::runtime_error("Binary data too small to contain ALIR magic");
            if (bytes[pos] != 'A' || bytes[pos + 1] != 'L' || bytes[pos + 2] != 'I' || bytes[pos + 3] != 'R')
                throw std::runtime_error("Invalid ALIR bytecode magic");
            pos += 4;
        }

        void expect_eof(std::span<const uint8_t> bytes, size_t pos) {
            if (pos != bytes.size())
                throw std::runtime_error(
                    "Trailing bytes after ALIR bytecode: " + std::to_string(bytes.size() - pos) + " bytes"
                );
        }

    } // anonymous namespace

    void ALIRInterpreter::load_from_bytecode(const std::span<const uint8_t> &binary_data) {
        ops.clear();
        size_t pos = 0;
        expect_magic(binary_data, pos);
        const uint32_t version = read_u32(binary_data, pos);
        if (version != alir_version) {
            throw std::runtime_error(
                "ALIR version mismatch: expected version " + std::to_string(alir_version) + ", got version " +
                std::to_string(version) +
                ". Run `alkaid convert INPUT OUTPUT --flavor json` for older JSON files."
            );
        }

        n_in = read_u32(binary_data, pos);
        n_out = read_u32(binary_data, pos);
        n_ops = read_u32(binary_data, pos);
        n_tables = read_u32(binary_data, pos);

        inp_shifts.resize(n_in);
        out_idxs.resize(n_out);
        out_shifts.resize(n_out);
        out_negs.resize(n_out);
        std::vector<Op> program(n_ops);
        std::vector<DType> dtypes(n_ops);
        std::vector<uint32_t> addr_pool;
        std::vector<int64_t> data_pool;

        for (size_t i = 0; i < n_in; ++i)
            inp_shifts[i] = read_i32(binary_data, pos);
        for (size_t i = 0; i < n_out; ++i)
            out_idxs[i] = read_i32(binary_data, pos);
        for (size_t i = 0; i < n_out; ++i)
            out_shifts[i] = read_i32(binary_data, pos);
        for (size_t i = 0; i < n_out; ++i)
            out_negs[i] = read_u8(binary_data, pos);

        for (size_t i = 0; i < n_ops; ++i) {
            OpLoad &op = program[i].load;
            op = OpLoad{};
            op.opcode = read_i8(binary_data, pos);
            op.addr_out = static_cast<uint32_t>(i);
            dtypes[i].is_signed = read_u8(binary_data, pos);
            dtypes[i].integers = read_i8(binary_data, pos);
            dtypes[i].fractionals = read_i8(binary_data, pos);
            const uint16_t n_addr = read_u16(binary_data, pos);
            const uint16_t n_data = read_u16(binary_data, pos);
            op.n_addr = n_addr;
            op.n_data = n_data;
            op.addr_offset = static_cast<uint32_t>(addr_pool.size());
            op.data_offset = static_cast<uint32_t>(data_pool.size());

            for (uint16_t j = 0; j < n_addr; ++j)
                addr_pool.push_back(read_u32(binary_data, pos));
            for (uint16_t j = 0; j < n_data; ++j)
                data_pool.push_back(read_i64(binary_data, pos));
        }

        lookup_tables.clear();
        lookup_tables.reserve(n_tables);
        for (size_t i = 0; i < n_tables; ++i) {
            const uint32_t table_size = read_u32(binary_data, pos);
            std::vector<int32_t> table_data(table_size);
            for (uint32_t j = 0; j < table_size; ++j)
                table_data[j] = read_i32(binary_data, pos);
            lookup_tables.emplace_back(std::move(table_data));
        }
        expect_eof(binary_data, pos);

        max_ops_width = 0;
        max_inp_width = 0;
        max_out_width = 0;
        bits_in = 0;
        bits_out = 0;
        for (size_t i = 0; i < n_ops; ++i) {
            int32_t width = dtypes[i].width();
            if (program[i].load.opcode == -1) {
                max_inp_width = std::max(max_inp_width, width);
                bits_in += width;
            }
            max_ops_width = std::max(max_ops_width, width);
        }
        for (int32_t idx : out_idxs) {
            if (idx >= 0) {
                int32_t width = dtypes[idx].width();
                max_out_width = std::max(max_out_width, width);
                bits_out += width;
            }
        }

        for (size_t i = 0; i < n_ops; ++i) {
            const OpLoad &op = program[i].load;
            for (uint16_t j = 0; j < op.n_addr; ++j) {
                const uint32_t addr = addr_pool[op.addr_offset + j];
                if (addr >= i)
                    throw std::runtime_error(
                        "Operation " + std::to_string(i) + " has address " + std::to_string(addr) +
                        " violating causality"
                    );
            }
            if (op.opcode == 8 && (op.n_data == 0 || data_pool[op.data_offset] < 0 ||
                                   data_pool[op.data_offset] >= static_cast<int64_t>(n_tables)))
                throw std::runtime_error(
                    "Operation " + std::to_string(i) + " has out-of-range lookup table index"
                );
        }
        if (max_ops_width > 64) {
            throw std::runtime_error(
                "ALIR op width " + std::to_string(max_ops_width) +
                " > 64 bits is not representable in the int64 interpreter"
            );
        }

        build_exec_program(std::move(program), addr_pool, data_pool, dtypes);
    }

    void ALIRInterpreter::print_program_info() const {
        std::cout << "ALIR Sequence:\n"
                  << n_in << " (" << bits_in << " bits) -> " << n_out << " (" << bits_out << " bits)\n"
                  << "# operations: " << n_ops << "\n"
                  << "# live slots: " << n_slots << "\n"
                  << "Maximum intermediate width: " << max_ops_width << " bits\n";
    }

} // namespace alir
