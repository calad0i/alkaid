#include "fsm_config.hh"
#include "fsm_wrapper.hh"

#include <cstddef>
#include <cstdint>

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

    void fsm_run(
        void *handle,
        const double *const *input_data,
        const size_t *input_n_samples,
        double *const *output_data,
        size_t steps,
        uint8_t scheduled
    ) {
        static_cast<wrapper_t *>(handle)->run(
            input_data, input_n_samples, output_data, steps, scheduled != 0
        );
    }
}
