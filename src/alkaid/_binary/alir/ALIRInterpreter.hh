#pragma once

#include "alir_types.hh"

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

namespace alir {

    class ALIRInterpreter {
      private:
        size_t n_in = 0, n_out = 0, n_ops = 0, n_slots = 0, n_tables = 0;
        int32_t max_ops_width = 0, max_inp_width = 0, max_out_width = 0;
        size_t bits_in = 0, bits_out = 0;

        std::vector<int32_t> inp_shifts;
        std::vector<int32_t> out_idxs;      // op-indexed (from on-disk format)
        std::vector<int32_t> out_idxs_slot; // slot-indexed (after transpile)
        std::vector<int32_t> out_shifts;
        std::vector<int32_t> out_negs;
        std::vector<std::vector<int32_t>> lookup_tables;

        std::vector<OpExec> ops_exec;

        std::vector<double> input_scales;   // per-op (-1 only)
        std::vector<double> output_scales;  // per output
        std::vector<double> op_dump_scales; // per op

        std::vector<int32_t> op_out_addr; // op -> slot, kept for dump path

        void build_exec_program(const std::vector<Op> &ops);

        template <int B> void exec_batch_core(const double *inputs, size_t batch_size, int64_t *buffer) const;

      public:
        static const int alir_version = 2;

        void load_from_file(const std::string &filename);
        void load_from_binary(const std::span<const int32_t> &binary_data);

        // Accept CombLogic JSON directly; handles gzip and plain.
        void load_from_json_file(const std::string &path);
        void load_from_json_string(std::string_view json_text);

        void print_program_info() const;

        // buffer must be n_slots * B int64s.
        template <int B>
        void exec_batch(const double *inputs, double *outputs, size_t batch_size, int64_t *buffer) const;

        // (batch_size, n_ops) per-op scaled decimals, for debug cross-check.
        template <int B>
        void dump_batch(const double *inputs, double *dump_outputs, size_t batch_size, int64_t *buffer) const;

        size_t get_n_in() const { return n_in; }
        size_t get_n_out() const { return n_out; }
        size_t get_n_ops() const { return n_ops; }
        size_t get_n_slots() const { return n_slots; }
    };

} // namespace alir
