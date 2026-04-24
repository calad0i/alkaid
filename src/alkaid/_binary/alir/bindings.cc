#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "ALIRInterpreter.hh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>
#include <omp.h>

#if defined(__linux__)
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;

template <class F> static void dispatch_batch_B(int batch_B, F &&f) {
    if (batch_B == 16)
        f(std::integral_constant<int, 16>{});
    else if (batch_B == 4)
        f(std::integral_constant<int, 4>{});
    else
        f(std::integral_constant<int, 8>{});
}

static void _run_predict(
    const alir::ALIRInterpreter &interp,
    const std::span<const double> &inputs,
    std::span<double> &outputs,
    size_t n_samples,
    int batch_B
) {
    const size_t n_in = interp.get_n_in();
    const size_t n_out = interp.get_n_out();
    const size_t n_slots = interp.get_n_slots();
    dispatch_batch_B(batch_B, [&](auto B_const) {
        constexpr int B = decltype(B_const)::value;
        std::vector<int64_t> buffer(n_slots * B);
        const size_t n_full = n_samples / B;
        const size_t rem = n_samples % B;
        for (size_t batch = 0; batch < n_full; ++batch) {
            interp.exec_batch<B>(&inputs[batch * B * n_in], &outputs[batch * B * n_out], B, buffer.data());
        }
        if (rem) {
            interp.exec_batch<B>(
                &inputs[n_full * B * n_in], &outputs[n_full * B * n_out], rem, buffer.data()
            );
        }
    });
}

static void _run_dump(
    const alir::ALIRInterpreter &interp,
    const std::span<const double> &inputs,
    std::span<double> &outputs,
    size_t n_samples,
    int batch_B
) {
    const size_t n_in = interp.get_n_in();
    const size_t n_ops = interp.get_n_ops();
    const size_t n_slots = interp.get_n_slots();
    dispatch_batch_B(batch_B, [&](auto B_const) {
        constexpr int B = decltype(B_const)::value;
        std::vector<int64_t> buffer(n_slots * B);
        const size_t n_full = n_samples / B;
        const size_t rem = n_samples % B;
        for (size_t batch = 0; batch < n_full; ++batch) {
            interp.dump_batch<B>(&inputs[batch * B * n_in], &outputs[batch * B * n_ops], B, buffer.data());
        }
        if (rem) {
            interp.dump_batch<B>(
                &inputs[n_full * B * n_in], &outputs[n_full * B * n_ops], rem, buffer.data()
            );
        }
    });
}

// Return cache size in bytes, or 0 if unknown (non-Linux/macOS,
// aarch64 Linux, musl, etc.).
static size_t probe_l2_cache_per_core() {
#if defined(__linux__) && defined(_SC_LEVEL2_CACHE_SIZE)
    long r = sysconf(_SC_LEVEL2_CACHE_SIZE);
    return r > 0 ? (size_t)r : 0;
#elif defined(__APPLE__)
    // Apple Silicon reports per-perflevel L2; prefer the perf cluster.
    uint64_t val = 0;
    size_t sz = sizeof(val);
    if (sysctlbyname("hw.perflevel0.l2cachesize", &val, &sz, nullptr, 0) == 0 && val > 0)
        return (size_t)val;
    sz = sizeof(val);
    if (sysctlbyname("hw.l2cachesize", &val, &sz, nullptr, 0) == 0 && val > 0)
        return (size_t)val;
    return 0;
#else
    return 0;
#endif
}

static size_t probe_l3_cache() {
#if defined(__linux__) && defined(_SC_LEVEL3_CACHE_SIZE)
    long r = sysconf(_SC_LEVEL3_CACHE_SIZE);
    return r > 0 ? (size_t)r : 0;
#elif defined(__APPLE__)
    uint64_t val = 0;
    size_t sz = sizeof(val);
    if (sysctlbyname("hw.l3cachesize", &val, &sz, nullptr, 0) == 0 && val > 0)
        return (size_t)val;
    return 0;
#else
    return 0;
#endif
}

// Pick (B, n_threads) from the probed cache sizes. B is the largest of
// {16, 8, 4} whose per-thread buffer fits 80% of L2; defaults to 8 when
// unknown. n_threads is capped so the aggregate fits 80% of L3. Override
// via ALIR_BATCH_B / ALIR_NUM_THREADS.
struct AutoConfig {
    int64_t n_threads;
    int B;
};
static AutoConfig pick_auto_config(size_t n_samples, size_t n_slots, int64_t requested_threads) {
    const size_t l2 = probe_l2_cache_per_core();
    const size_t l3 = probe_l3_cache();
    const size_t slot_bytes = n_slots * sizeof(int64_t);

    int B = 8;
    if (l2 > 0 && slot_bytes > 0) {
        const size_t budget = l2 * 8 / 10;
        for (int candidate : {16, 8, 4}) {
            if ((size_t)candidate * slot_bytes <= budget) {
                B = candidate;
                break;
            }
        }
    }

    int64_t hw_max = omp_get_max_threads();
    int64_t T = (requested_threads > 0) ? std::min<int64_t>(requested_threads, hw_max) : hw_max;
    T = std::min<int64_t>(T, std::max<int64_t>(1, (int64_t)(n_samples / 32)));
    T = std::max<int64_t>(1, T);

    if (requested_threads > 0)
        return {T, B};

    const size_t per_thread_bytes = (size_t)B * slot_bytes;
    if (l2 > 0 && per_thread_bytes <= (l2 * 8 / 10))
        return {T, B};

    if (l3 > 0 && per_thread_bytes > 0) {
        int64_t cap = (int64_t)((l3 * 8 / 10) / per_thread_bytes);
        if (cap < 1)
            cap = 1;
        if (cap < T)
            T = cap;
    }
    return {T, B};
}

static void run_interp_loaded(
    const alir::ALIRInterpreter &interp,
    const std::span<const double> &input,
    std::span<double> &output,
    int64_t n_threads,
    bool dump
) {
    const size_t n_in = interp.get_n_in();
    const size_t n_out_per_sample = dump ? interp.get_n_ops() : interp.get_n_out();
    const size_t n_samples = input.size() / n_in;
    const size_t n_slots = interp.get_n_slots();

    AutoConfig cfg = pick_auto_config(n_samples, n_slots, n_threads);

    const char *b_env = std::getenv("ALIR_BATCH_B");
    if (b_env) {
        int bv = std::atoi(b_env);
        if (bv == 4 || bv == 8 || bv == 16)
            cfg.B = bv;
    }
    const char *t_env = std::getenv("ALIR_NUM_THREADS");
    if (t_env) {
        int64_t tv = std::atoll(t_env);
        if (tv > 0)
            cfg.n_threads = std::min<int64_t>(tv, omp_get_max_threads());
    }

    size_t n_samples_per_thread = std::max<size_t>(n_samples / std::max<int64_t>(1, cfg.n_threads), 32);
    size_t n_thread = n_samples / n_samples_per_thread;
    n_thread += (n_samples % n_samples_per_thread) ? 1 : 0;

    std::exception_ptr eptr = nullptr;

#pragma omp parallel for num_threads(n_thread) schedule(static)
    for (size_t i = 0; i < n_thread; ++i) {
        size_t start = i * n_samples_per_thread;
        size_t end = std::min<size_t>(start + n_samples_per_thread, n_samples);
        size_t n_samples_this_thread = end - start;
        const std::span<const double> inp_span(&input[start * n_in], n_samples_this_thread * n_in);
        std::span<double> out_span(
            &output[start * n_out_per_sample], n_samples_this_thread * n_out_per_sample
        );
        try {
            if (dump)
                _run_dump(interp, inp_span, out_span, n_samples_this_thread, cfg.B);
            else
                _run_predict(interp, inp_span, out_span, n_samples_this_thread, cfg.B);
        }
        catch (...) {
#pragma omp critical
            {
                if (!eptr)
                    eptr = std::current_exception();
            }
        }
    }

    if (eptr)
        std::rethrow_exception(eptr);
}

static nb::ndarray<nb::numpy, double> run_interp_numpy(
    const nb::ndarray<int32_t> &bin_logic,
    const nb::ndarray<nb::numpy, double> &input,
    int64_t n_threads,
    bool dump
) {
    const int32_t *bin_logic_ptr = bin_logic.data();
    if (bin_logic.size() < 5)
        throw std::runtime_error("Invalid binary logic data");
    if (bin_logic_ptr[0] != alir::ALIRInterpreter::alir_version) {
        throw std::runtime_error(
            "ALIR version mismatch: expected version " + std::to_string(alir::ALIRInterpreter::alir_version) +
            ", got version " + std::to_string(bin_logic_ptr[0])
        );
    }

    alir::ALIRInterpreter interp;
    interp.load_from_binary(std::span<const int32_t>(bin_logic_ptr, bin_logic.size()));

    const size_t n_samples = input.size() / interp.get_n_in();
    const size_t n_out = dump ? interp.get_n_ops() : interp.get_n_out();

    double *output_ptr = new double[n_samples * n_out];
    std::span<double> out_span(output_ptr, n_samples * n_out);
    const std::span<const double> inp_span(input.data(), input.size());

    run_interp_loaded(interp, inp_span, out_span, n_threads, dump);

    nb::capsule owner(output_ptr, [](void *p) noexcept { delete[] (double *)p; });
    return nb::ndarray<nb::numpy, double>(output_ptr, {n_samples, n_out}, owner);
}

static nb::ndarray<nb::numpy, double> run_interp_json_numpy(
    const std::string &json_text,
    const nb::ndarray<nb::numpy, double> &input,
    int64_t n_threads,
    bool dump
) {
    alir::ALIRInterpreter interp;
    interp.load_from_json_string(json_text);

    const size_t n_samples = input.size() / interp.get_n_in();
    const size_t n_out = dump ? interp.get_n_ops() : interp.get_n_out();

    double *output_ptr = new double[n_samples * n_out];
    std::span<double> out_span(output_ptr, n_samples * n_out);
    const std::span<const double> inp_span(input.data(), input.size());

    run_interp_loaded(interp, inp_span, out_span, n_threads, dump);

    nb::capsule owner(output_ptr, [](void *p) noexcept { delete[] (double *)p; });
    return nb::ndarray<nb::numpy, double>(output_ptr, {n_samples, n_out}, owner);
}

static nb::ndarray<nb::numpy, double> run_interp_json_file_numpy(
    const std::string &path,
    const nb::ndarray<nb::numpy, double> &input,
    int64_t n_threads,
    bool dump
) {
    alir::ALIRInterpreter interp;
    interp.load_from_json_file(path);

    const size_t n_samples = input.size() / interp.get_n_in();
    const size_t n_out = dump ? interp.get_n_ops() : interp.get_n_out();

    double *output_ptr = new double[n_samples * n_out];
    std::span<double> out_span(output_ptr, n_samples * n_out);
    const std::span<const double> inp_span(input.data(), input.size());

    run_interp_loaded(interp, inp_span, out_span, n_threads, dump);

    nb::capsule owner(output_ptr, [](void *p) noexcept { delete[] (double *)p; });
    return nb::ndarray<nb::numpy, double>(output_ptr, {n_samples, n_out}, owner);
}

NB_MODULE(alir_bin, m) {
    m.def("run_interp", &run_interp_numpy, "bin_logic"_a, "data"_a, "n_threads"_a = 1, "dump"_a = false);
    m.def(
        "run_interp_json", &run_interp_json_numpy, "json_text"_a, "data"_a, "n_threads"_a = 1, "dump"_a = false
    );
    m.def(
        "run_interp_json_file",
        &run_interp_json_file_numpy,
        "path"_a,
        "data"_a,
        "n_threads"_a = 1,
        "dump"_a = false
    );
}
