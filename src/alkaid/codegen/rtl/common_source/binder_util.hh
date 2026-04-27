#include "ioutil.hh"
#include <cmath>
#include <verilated.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
constexpr bool _openmp = true;
#else
constexpr bool _openmp = false;
#endif

inline int64_t _sign_ext(uint64_t v, int bw) {
    uint64_t sign_bit = uint64_t(1) << (bw - 1);
    uint64_t ext_mask = ~((uint64_t(1) << bw) - 1);
    if (v & sign_bit) {
        v |= ext_mask;
    }
    return static_cast<int64_t>(v);
}

template <typename CONFIG_T, typename T> static inline int64_t _fp_to_int(T v) {
    static const double scale = std::ldexp(1.0, CONFIG_T::f_in);
    return static_cast<int64_t>(std::floor(static_cast<double>(v) * scale));
}

template <typename CONFIG_T, typename T> static inline T _int_to_fp(int64_t v) {
    static const double scale_inv = std::ldexp(1.0, -CONFIG_T::f_out);
    constexpr int bw_out = CONFIG_T::k_out + CONFIG_T::i_out + CONFIG_T::f_out;
    double dv;
    if constexpr (CONFIG_T::k_out) {
        dv = static_cast<double>(_sign_ext(v, bw_out));
    }
    else {
        dv = static_cast<double>(static_cast<uint64_t>(v));
    }
    return static_cast<T>(dv * scale_inv);
}

template <typename CONFIG_T, typename T>
static inline void fp_to_int_vec(const T *in_fp, std::vector<int64_t> &in_int) {
    constexpr int bw_in = CONFIG_T::k_in + CONFIG_T::i_in + CONFIG_T::f_in;
#ifdef _OPENMP
#pragma omp simd
#endif
    for (size_t j = 0; j < CONFIG_T::N_inp; ++j) {
        in_int[j] = _fp_to_int<CONFIG_T, T>(in_fp[j]);
    }
}

template <typename CONFIG_T, typename T>
static inline void int_vec_to_fp(const std::vector<int64_t> &out_int, T *c_out) {
    constexpr int bw_out = CONFIG_T::k_out + CONFIG_T::i_out + CONFIG_T::f_out;
#ifdef _OPENMP
#pragma omp simd
#endif
    for (size_t j = 0; j < CONFIG_T::N_out; ++j) {
        c_out[j] = _int_to_fp<CONFIG_T, T>(out_int[j]);
    }
}

template <typename CONFIG_T, typename T>
std::enable_if_t<CONFIG_T::II != 0> _inference(T *c_inp, T *c_out, size_t n_samples) {
    constexpr int bw_in = CONFIG_T::k_in + CONFIG_T::i_in + CONFIG_T::f_in;
    constexpr int bw_out = CONFIG_T::k_out + CONFIG_T::i_out + CONFIG_T::f_out;
    auto dut = std::make_unique<typename CONFIG_T::dut_t>();

    size_t clk_req = n_samples * CONFIG_T::II + (CONFIG_T::latency - CONFIG_T::II) + 1;
    std::vector<int64_t> in_int(CONFIG_T::N_inp);
    std::vector<int64_t> out_int(CONFIG_T::N_out);

    for (size_t t_inp = 0; t_inp < clk_req; ++t_inp) {
        size_t t_out = t_inp - CONFIG_T::latency;

        if (t_inp < n_samples * CONFIG_T::II && t_inp % CONFIG_T::II == 0) {
            size_t off_in = t_inp / CONFIG_T::II * CONFIG_T::N_inp;
            fp_to_int_vec<CONFIG_T, T>(&c_inp[off_in], in_int);
            write_input<CONFIG_T::N_inp, bw_in>(dut->model_inp, in_int.data());
        }

        dut->clk = 0;
        dut->eval();
        dut->clk = 1;
        dut->eval();

        if (t_inp >= CONFIG_T::latency && t_out % CONFIG_T::II == 0) {
            read_output<CONFIG_T::N_out, bw_out>(dut->model_out, out_int.data());
            size_t off_out = t_out / CONFIG_T::II * CONFIG_T::N_out;
            int_vec_to_fp<CONFIG_T, T>(out_int, &c_out[off_out]);
        }
    }

    dut->final();
}

template <typename CONFIG_T, typename T>
std::enable_if_t<CONFIG_T::II == 0> _inference(T *c_inp, T *c_out, size_t n_samples) {
    constexpr int bw_in = CONFIG_T::k_in + CONFIG_T::i_in + CONFIG_T::f_in;
    constexpr int bw_out = CONFIG_T::k_out + CONFIG_T::i_out + CONFIG_T::f_out;
    auto dut = std::make_unique<typename CONFIG_T::dut_t>();
    std::vector<int64_t> in_int(CONFIG_T::N_inp);
    std::vector<int64_t> out_int(CONFIG_T::N_out);

    for (size_t i = 0; i < n_samples; ++i) {
        fp_to_int_vec<CONFIG_T, T>(&c_inp[i * CONFIG_T::N_inp], in_int);
        write_input<CONFIG_T::N_inp, bw_in>(dut->model_inp, in_int.data());
        dut->eval();
        read_output<CONFIG_T::N_out, bw_out>(dut->model_out, out_int.data());
        int_vec_to_fp<CONFIG_T, T>(out_int, &c_out[i * CONFIG_T::N_out]);
    }

    dut->final();
}

template <typename CONFIG_T, typename T>
void batch_inference(T *c_inp, T *c_out, size_t n_samples, size_t n_threads) {
    if (n_threads > 1 || n_threads == 0) {
#ifdef _OPENMP
        size_t min_samples_per_thread;
        size_t n_max_threads;
        if (n_threads == 0) {
            min_samples_per_thread = 1;
            n_max_threads = omp_get_max_threads();
        }
        else {
            min_samples_per_thread = std::max<size_t>(1, n_samples / n_threads);
            n_max_threads = n_threads;
        }
        size_t n_sample_per_thread = n_samples / n_max_threads + (n_samples % n_max_threads ? 1 : 0);
        n_sample_per_thread = std::max<size_t>(n_sample_per_thread, min_samples_per_thread);
        size_t n_thread = n_samples / n_sample_per_thread;
        n_thread += (n_samples % n_sample_per_thread) ? 1 : 0;

#pragma omp parallel for num_threads(n_thread) schedule(static)
        for (size_t i = 0; i < n_thread; ++i) {
            size_t start = i * n_sample_per_thread;
            size_t end = std::min<size_t>(start + n_sample_per_thread, n_samples);
            size_t n_samples_this_thread = end - start;
            size_t offset_in = start * CONFIG_T::N_inp;
            size_t offset_out = start * CONFIG_T::N_out;
            _inference<CONFIG_T, T>(&c_inp[offset_in], &c_out[offset_out], n_samples_this_thread);
        }
#else
        _inference<CONFIG_T, T>(c_inp, c_out, n_samples);
#endif
    }
    else {
        _inference<CONFIG_T, T>(c_inp, c_out, n_samples);
    }
}
