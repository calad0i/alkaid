#pragma once

#include "ioutil.hh"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

struct fsm_schedule_config_t {
    uint8_t enabled;
    size_t period;
    size_t bias;
    const uint8_t *valid_mask;
};

template <size_t N, size_t BW, typename field_t> void fsm_set_raw(field_t &field, const int64_t *values) {
    write_input<N, BW>(field, values);
}

template <size_t N, size_t BW, typename field_t> void fsm_get_raw(const field_t &field, int64_t *values) {
    read_output<N, BW>(field, values);
}

template <size_t N, size_t BW, int FRAC, typename field_t>
void fsm_set_float(field_t &field, const double *values) {
    int64_t raw[N];
    for (size_t i = 0; i < N; ++i) {
        raw[i] = fp_to_int<FRAC>(values[i]);
    }
    write_input<N, BW>(field, raw);
}

template <size_t N, size_t BW, bool SGN, int FRAC, typename field_t>
void fsm_get_float(const field_t &field, double *values) {
    int64_t raw[N];
    read_output<N, BW>(field, raw);
    for (size_t i = 0; i < N; ++i) {
        values[i] = int_to_fp<SGN, FRAC, static_cast<int>(BW)>(raw[i]);
    }
}

inline bool fsm_schedule_check(const fsm_schedule_config_t &schedule, size_t t) {
    if (!schedule.enabled) {
        return true;
    }
    if (t < schedule.bias) {
        return false;
    }
    assert(schedule.period > 0);
    return schedule.valid_mask[(t - schedule.bias) % schedule.period] != 0;
}

// Number of valid cycles in [0, t_end) for a schedule.
inline size_t fsm_valid_count(const fsm_schedule_config_t &schedule, size_t n_period) {
    if (!schedule.enabled) {
        return n_period * schedule.period;
    }
    size_t per_period = 0;
    for (size_t i = 0; i < schedule.period; ++i) {
        per_period += schedule.valid_mask[i] != 0;
    }
    return per_period * n_period;
}

template <typename config_t> inline size_t get_period() {
    static_assert(
        config_t::n_inputs + config_t::n_outputs > 0,
        "At least one signal is required to determine the period."
    );
    auto schedule = config_t::signal_schedules[0];
    return schedule.period;
}

// inline size_t get_period(const

template <typename T, typename = void> struct fsm_has_clk : std::false_type {};
template <typename T>
struct fsm_has_clk<T, std::void_t<decltype(std::declval<T &>().clk)>> : std::true_type {};

template <typename config_t>
void config_set_signal(typename config_t::dut_t *dut, size_t signal_id, const int64_t *values) {
    config_t::visit_signal(dut, signal_id, [&](auto &member, auto N, auto BW, auto, auto) {
        fsm_set_raw<decltype(N)::value, decltype(BW)::value>(member, values);
    });
}

template <typename config_t>
void config_get_signal(typename config_t::dut_t *dut, size_t signal_id, int64_t *values) {
    config_t::visit_signal(dut, signal_id, [&](const auto &member, auto N, auto BW, auto, auto) {
        fsm_get_raw<decltype(N)::value, decltype(BW)::value>(member, values);
    });
}

template <typename config_t>
void config_set_float_signal(typename config_t::dut_t *dut, size_t signal_id, const double *values) {
    config_t::visit_signal(dut, signal_id, [&](auto &member, auto N, auto BW, auto, auto FRAC) {
        fsm_set_float<decltype(N)::value, decltype(BW)::value, decltype(FRAC)::value>(member, values);
    });
}

template <typename config_t>
void config_get_float_signal(typename config_t::dut_t *dut, size_t signal_id, double *values) {
    config_t::visit_signal(dut, signal_id, [&](const auto &member, auto N, auto BW, auto SGN, auto FRAC) {
        fsm_get_float<decltype(N)::value, decltype(BW)::value, decltype(SGN)::value, decltype(FRAC)::value>(
            member, values
        );
    });
}

// Drive every reset-control port to `value` (1 = assert, 0 = deassert; active-high).
template <typename config_t> void config_drive_reset(typename config_t::dut_t *dut, int64_t value) {
    for (size_t k = 0; k < config_t::n_reset_signals; ++k) {
        config_set_signal<config_t>(dut, config_t::reset_signal_ids[k], &value);
    }
}

template <typename config_t> class FSMWrapper {
  private:
    using dut_t = typename config_t::dut_t;
    static constexpr bool has_clk = fsm_has_clk<dut_t>::value;

    std::unique_ptr<dut_t> dut_;
    size_t t_ = 0;

  public:
    FSMWrapper() {
        dut_ = std::make_unique<dut_t>();
        if constexpr (has_clk) {
            dut_->clk = false;
        }
        dut_->eval();
    }

    ~FSMWrapper() {
        if (dut_) {
            dut_->final();
        }
    }

    void soft_reset() {
        if constexpr (config_t::n_reset_signals > 0) {
            config_drive_reset<config_t>(dut_.get(), 1);
            dut_->eval();
            tick();
            config_drive_reset<config_t>(dut_.get(), 0);
        }
        if constexpr (has_clk) {
            dut_->clk = false;
        }
        dut_->eval();
        t_ = 0;
    }

    void eval() { dut_->eval(); }

    void tick() {
        if constexpr (has_clk) {
            dut_->clk = true;
            dut_->eval();
            dut_->clk = false;
        }
        ++t_;
    }

    size_t time() const { return t_; }

    void set_signal(size_t signal_id, const int64_t *values) {
        config_set_signal<config_t>(dut_.get(), signal_id, values);
    }

    void get_signal(size_t signal_id, int64_t *values) const {
        config_get_signal<config_t>(dut_.get(), signal_id, values);
    }

    void set_float_signal(size_t signal_id, const double *values) {
        config_set_float_signal<config_t>(dut_.get(), signal_id, values);
    }

    void get_float_signal(size_t signal_id, double *values) const {
        config_get_float_signal<config_t>(dut_.get(), signal_id, values);
    }

    void run(
        const double *const *input_data,
        const size_t *input_n_samples,
        double *const *output_data,
        size_t steps,
        bool scheduled,
        size_t _period_start = 0 // _period_start > 0 implies launching worker in parallel exec
    ) {
        std::vector<size_t> input_counts(config_t::n_inputs, 0);
        std::vector<size_t> output_counts(config_t::n_outputs, 0);

        if (_period_start > 0) {
            for (size_t i = 0; i < config_t::n_inputs; ++i) {
                input_counts[i] = fsm_valid_count(config_t::signal_schedules[i], _period_start);
            }
            for (size_t j = 0; j < config_t::n_outputs; ++j) {
                output_counts[j] =
                    fsm_valid_count(config_t::signal_schedules[config_t::n_inputs + j], _period_start);
            }
        }

        for (size_t step = 0; step < steps; ++step) {
            const size_t t = t_;
            for (size_t i = 0; i < config_t::n_inputs; ++i) {
                if (scheduled && !fsm_schedule_check(config_t::signal_schedules[i], t)) {
                    continue;
                }
                const size_t sample_idx = input_counts[i]++;
                if (sample_idx < input_n_samples[i]) {
                    set_float_signal(i, input_data[i] + sample_idx * config_t::signal_sizes[i]);
                }
            }

            eval();
            tick();

            for (size_t j = 0; j < config_t::n_outputs; ++j) {
                const size_t sid = config_t::n_inputs + j;
                if (scheduled && !fsm_schedule_check(config_t::signal_schedules[sid], t_)) {
                    continue;
                }
                const size_t sample_idx = output_counts[j]++;
                get_float_signal(sid, output_data[j] + sample_idx * config_t::signal_sizes[sid]);
            }
        }
    }
};
