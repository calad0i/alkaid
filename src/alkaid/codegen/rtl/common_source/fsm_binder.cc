#include "fsm_config.hh"
#include "fsm_wrapper.hh"

#include <cstddef>
#include <cstdint>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

using wrapper_t = FSMWrapper<fsm_config_t>;

extern "C"
{
    void *fsm_create() { return new wrapper_t(); }

    void fsm_destroy(void *handle) { delete static_cast<wrapper_t *>(handle); }

    void fsm_soft_reset(void *handle) { static_cast<wrapper_t *>(handle)->soft_reset(); }

    void fsm_eval(void *handle) { static_cast<wrapper_t *>(handle)->eval(); }

    void fsm_tick(void *handle) { static_cast<wrapper_t *>(handle)->tick(); }

    size_t fsm_time(void *handle) { return static_cast<wrapper_t *>(handle)->time(); }

    void fsm_set_signal(void *handle, size_t signal_id, const int64_t *values) {
        static_cast<wrapper_t *>(handle)->set_signal(signal_id, values);
    }

    void fsm_get_signal(void *handle, size_t signal_id, int64_t *values) {
        static_cast<wrapper_t *>(handle)->get_signal(signal_id, values);
    }

    void fsm_set_signal_f64(void *handle, size_t signal_id, const double *values) {
        static_cast<wrapper_t *>(handle)->set_float_signal(signal_id, values);
    }

    void fsm_get_signal_f64(void *handle, size_t signal_id, double *values) {
        static_cast<wrapper_t *>(handle)->get_float_signal(signal_id, values);
    }

    int openmp_enabled() {
#ifdef _OPENMP
        return 1;
#else
        return 0;
#endif
    }

    void fsm_run(
        void *handle,
        const double *const *input_data,
        const size_t *input_n_samples,
        double *const *output_data,
        size_t steps,
        size_t extra_steps,
        uint8_t scheduled,
        size_t n_thread
    ) {
        if (n_thread == 1 || !openmp_enabled() || !scheduled) {
            static_cast<wrapper_t *>(handle)->run(
                input_data, input_n_samples, output_data, steps + extra_steps, scheduled != 0, 0
            );
        }
#ifdef _OPENMP
        else {
            size_t period = get_period<fsm_config_t>();
            if (steps % period != 0) {
                std::cerr << "Error: steps must be a multiple of " << period << " when n_thread > 1"
                          << std::endl;
                return;
            }
            size_t n_periods = steps / period;
            if (n_thread == 0) {
                n_thread = omp_get_max_threads();
            }
            n_thread = std::min(n_thread, n_periods);
            size_t n_periods_per_thread = (n_periods + n_thread - 1) / n_thread;

#pragma omp parallel for num_threads(n_thread) schedule(static)

            for (size_t i = 0; i < n_thread; ++i) {
                wrapper_t _dut;
                _dut.soft_reset();
                size_t start_period = i * n_periods_per_thread;
                size_t steps = std::min(n_periods_per_thread, n_periods - start_period) * period;
                _dut.run(input_data, input_n_samples, output_data, steps + extra_steps, true, start_period);
            }
        }
#endif
    }
}
