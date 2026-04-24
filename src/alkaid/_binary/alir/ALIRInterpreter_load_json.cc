// Parse CombLogic JSON into the same std::vector<Op> that load_from_binary
// produces, then hand off to build_exec_program. Accepts gzipped input.

#include "ALIRInterpreter.hh"
#include "alir_gzip.hh"
#include "alir_kif.hh"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace alir {

    namespace {

        using json = nlohmann::json;

        // Op 5 can encode full 64-bit patterns, so accept uint64 too.
        int64_t read_data_u64(const json &v) {
            if (v.is_number_unsigned())
                return static_cast<int64_t>(v.get<uint64_t>());
            if (v.is_number_integer())
                return v.get<int64_t>();
            throw std::runtime_error("op.data is not an integer");
        }

    } // anonymous namespace

    void ALIRInterpreter::load_from_json_string(std::string_view s) {
        std::string decompressed;
        if (is_gzip_magic(s.data(), s.size())) {
            decompressed = gzip_inflate(s.data(), s.size());
            s = decompressed;
        }
        json doc = json::parse(s);
        if (!doc.contains("spec_version") || doc["spec_version"].get<int>() != alir_version) {
            throw std::runtime_error(
                "ALIR JSON spec version mismatch: expected " + std::to_string(alir_version) + ", got " +
                (doc.contains("spec_version") ? std::to_string(doc["spec_version"].get<int>())
                                              : std::string("<missing>"))
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
        std::vector<Op> ops(n_ops);

        // qmin/qstep saved for opcode-8 pad_left fixup below (needs producer qint).
        std::vector<double> qmin(n_ops), qstep(n_ops);
        std::vector<DType> dtypes(n_ops);

        for (size_t i = 0; i < n_ops; ++i) {
            const json &jo = jops[i];
            const int32_t id0 = jo[0].get<int32_t>();
            const int32_t id1 = jo[1].get<int32_t>();
            const int32_t opcode = jo[2].get<int32_t>();
            const int64_t data = read_data_u64(jo[3]);
            const json &jq = jo[4];
            const double qm_min = jq[0].get<double>();
            const double qm_max = jq[1].get<double>();
            const double qm_step = jq[2].get<double>();

            const DType dtype = minimal_kif(qm_min, qm_max, qm_step);
            dtypes[i] = dtype;
            qmin[i] = qm_min;
            qstep[i] = qm_step;

            Op &op = ops[i];
            op.opcode = opcode;
            op.id0 = id0;
            op.id1 = id1;
            op.data_low = static_cast<int32_t>(data & 0xFFFFFFFFll);
            op.data_high = static_cast<int32_t>((data >> 32) & 0xFFFFFFFFll);
            op.dtype = dtype;
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

        // Opcode 8 packs pad_left (derived from the input op's qint) into
        // data_high. Python to_binary does the same.
        for (size_t i = 0; i < n_ops; ++i) {
            Op &op = ops[i];
            if (op.opcode != 8)
                continue;
            if (op.id0 < 0 || (size_t)op.id0 >= n_ops)
                throw std::runtime_error("op " + std::to_string(i) + " (opcode 8) has invalid id0");
            op.data_high = table_pad_left(qmin[op.id0], qstep[op.id0], dtypes[op.id0]);
        }

        n_tables = lookup_tables.size();

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

    void ALIRInterpreter::load_from_json_file(const std::string &path) {
        // binary mode: don't let line-ending conversion touch gzip streams.
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("Failed to open JSON file: " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        load_from_json_string(ss.str());
    }

} // namespace alir
