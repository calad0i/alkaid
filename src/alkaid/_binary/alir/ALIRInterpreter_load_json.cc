// Parse CombLogic JSON into the same std::vector<Op> that load_from_binary
// produces, then hand off to build_exec_program. Accepts gzipped input.

#include "ALIRInterpreter.hh"
#include "alir_gzip.hh"
#include "alir_kif.hh"

#include <nlohmann/json.hpp>

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace alir {

    namespace {

        using json = nlohmann::json;

        int64_t read_i64_json(const json &v) {
            if (v.is_number_unsigned()) {
                const uint64_t u = v.get<uint64_t>();
                if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                    throw std::runtime_error("op.data is outside signed int64 range");
                return static_cast<int64_t>(u);
            }
            if (v.is_number_integer())
                return v.get<int64_t>();
            throw std::runtime_error("op.data is not an integer");
        }

        uint32_t read_addr_json(const json &v) {
            if (v.is_number_unsigned()) {
                const uint64_t addr = v.get<uint64_t>();
                if (addr > std::numeric_limits<uint32_t>::max())
                    throw std::runtime_error("op.addr is outside uint32 range");
                return static_cast<uint32_t>(addr);
            }
            if (v.is_number_integer()) {
                const int64_t addr = v.get<int64_t>();
                if (addr < 0 || addr > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
                    throw std::runtime_error("op.addr is outside uint32 range");
                return static_cast<uint32_t>(addr);
            }
            throw std::runtime_error("op.addr is not an integer");
        }

    } // anonymous namespace

    void ALIRInterpreter::load_from_json_string(std::string_view s) {
        ops.clear();
        json doc = json::parse(s);
        if (!doc.contains("spec_version") || doc["spec_version"].get<int>() != alir_version) {
            throw std::runtime_error(
                "ALIR JSON spec version mismatch: expected " + std::to_string(alir_version) + ", got " +
                (doc.contains("spec_version") ? std::to_string(doc["spec_version"].get<int>())
                                              : std::string("<missing>")) +
                ". Run `alkaid convert INPUT OUTPUT --flavor json` to convert older JSON first."
            );
        }
        if (!doc.contains("meta") || doc["meta"].get<std::string>() != "ALIRModel") {
            throw std::runtime_error(
                "ALIR JSON meta mismatch: expected 'ALIRModel', got '" +
                (doc.contains("meta") ? doc["meta"].get<std::string>() : std::string("<missing>")) + "'"
            );
        }
        const json &m = doc.at("model");
        if (!m.is_array() || (m.size() != 8 && m.size() != 9)) {
            throw std::runtime_error(
                "ALIR JSON model is not a length-8-or-9 array (got " + std::to_string(m.size()) + ")"
            );
        }

        const json &shape = m[0];
        n_in = shape[0].get<size_t>();
        n_out = shape[1].get<size_t>();

        inp_shifts.resize(n_in);
        for (size_t i = 0; i < n_in; ++i)
            inp_shifts[i] = m[1][i].get<int32_t>();

        out_idxs.resize(n_out);
        for (size_t j = 0; j < n_out; ++j)
            out_idxs[j] = m[2][j].get<int32_t>();

        out_shifts.resize(n_out);
        for (size_t j = 0; j < n_out; ++j)
            out_shifts[j] = m[3][j].get<int32_t>();

        out_negs.resize(n_out);
        for (size_t j = 0; j < n_out; ++j) {
            const json &v = m[4][j];
            out_negs[j] = v.is_boolean() ? (v.get<bool>() ? 1 : 0) : v.get<int32_t>();
        }

        const json &jops = m[5];
        n_ops = jops.size();
        std::vector<Op> program(n_ops);
        std::vector<uint32_t> addr_pool;
        std::vector<int64_t> data_pool;

        // qmin/qstep saved for opcode-8 pad_left fixup below (needs producer qint).
        std::vector<double> qmin(n_ops), qstep(n_ops);
        std::vector<DType> dtypes(n_ops);

        for (size_t i = 0; i < n_ops; ++i) {
            const json &jo = jops[i];
            if (!jo.is_array() || jo.size() != 6)
                throw std::runtime_error(
                    "ALIR JSON op " + std::to_string(i) + " is not a v4 length-6 record"
                );
            const json &addr = jo[0];
            const int32_t opcode = jo[1].get<int32_t>();
            const json &payload = jo[2];
            if (!addr.is_array() || !payload.is_array())
                throw std::runtime_error("ALIR JSON op addr/data fields must be arrays");
            const json &jq = jo[3];
            const double qm_min = jq[0].get<double>();
            const double qm_max = jq[1].get<double>();
            const double qm_step = jq[2].get<double>();

            const DType dtype = minimal_kif(qm_min, qm_max, qm_step);
            dtypes[i] = dtype;
            qmin[i] = qm_min;
            qstep[i] = qm_step;

            if (addr.size() > std::numeric_limits<uint16_t>::max() ||
                payload.size() > std::numeric_limits<uint16_t>::max())
                throw std::runtime_error("ALIR JSON op has too many addr/data entries");

            OpLoad &op = program[i].load;
            op = OpLoad{};
            op.opcode = opcode;
            op.addr_out = static_cast<uint32_t>(i);
            op.addr_offset = static_cast<uint32_t>(addr_pool.size());
            op.data_offset = static_cast<uint32_t>(data_pool.size());
            op.n_addr = static_cast<uint16_t>(addr.size());
            op.n_data = static_cast<uint16_t>(payload.size());
            for (const json &v : addr)
                addr_pool.push_back(read_addr_json(v));
            for (const json &v : payload)
                data_pool.push_back(read_i64_json(v));
        }

        lookup_tables.clear();
        if (m.size() == 9) {
            const json &jtabs = m[8];
            lookup_tables.reserve(jtabs.size());
            for (const auto &jt : jtabs) {
                const json &arr = jt.at("table");
                std::vector<int32_t> table(arr.size());
                for (size_t k = 0; k < arr.size(); ++k)
                    table[k] = arr[k].get<int32_t>();
                lookup_tables.emplace_back(std::move(table));
            }
        }

        // Opcode 8 JSON is semantic; derive the internal lookup pad entry from the input op's qint.
        for (size_t i = 0; i < n_ops; ++i) {
            OpLoad &op = program[i].load;
            if (op.opcode != 8)
                continue;
            if (op.n_addr == 0 || addr_pool[op.addr_offset] >= n_ops)
                throw std::runtime_error("op " + std::to_string(i) + " (opcode 8) has invalid input address");
            if (op.n_data == 0)
                throw std::runtime_error("op " + std::to_string(i) + " (opcode 8) has no lookup table index");
            if (op.n_data == std::numeric_limits<uint16_t>::max())
                throw std::runtime_error("op " + std::to_string(i) + " (opcode 8) has too many data entries");
            const uint32_t old_offset = op.data_offset;
            const uint16_t old_n = op.n_data;
            const uint32_t producer = addr_pool[op.addr_offset];
            op.data_offset = static_cast<uint32_t>(data_pool.size());
            op.n_data = old_n + 1;
            data_pool.push_back(data_pool[old_offset]);
            data_pool.push_back(table_pad_left(qmin[producer], qstep[producer], dtypes[producer]));
            for (uint16_t j = 1; j < old_n; ++j)
                data_pool.push_back(data_pool[old_offset + j]);
        }

        n_tables = lookup_tables.size();

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

    void ALIRInterpreter::load_from_json_file(const std::string &path) {
        // binary mode: don't let line-ending conversion touch gzip streams.
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("Failed to open JSON file: " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string s = ss.str();
        std::string decompressed;
        if (is_gzip_magic(s.data(), s.size())) {
            decompressed = gzip_inflate(s.data(), s.size());
        }
        else {
            decompressed = std::move(s);
        }
        load_from_json_string(decompressed);
    }

} // namespace alir
