# Alkaid: Distributed Arithmetic for Machine Learning

[![Tests](https://img.shields.io/github/actions/workflow/status/calad0i/alkaid/unit-test.yml?label=test)](https://github.com/calad0i/alkaid/actions/workflows/unit-test.yml)
[![Documentation](https://img.shields.io/github/actions/workflow/status/calad0i/alkaid/sphinx-build.yml?label=doc)](https://calad0i.github.io/alkaid/)
[![Cov](https://img.shields.io/codecov/c/github/calad0i/alkaid)](https://codecov.io/gh/calad0i/alkaid)

`alkaid` is a lightweight compiler for generating low-latency static-dataflow kernels for FPGAs. It traces quantized arithmetic into ALIR, applies distributed-arithmetic optimization through CMVM where useful, and emits RTL or HLS projects.

The supported code generation targets are:

- RTL: Verilog and VHDL 2008, with optional pipelining.
- HLS: Vitis HLS, HLSlib, and oneAPI-style C++.
- XLS: experimental Verilog generation and JIT execution through `xls-python`.

Models can be described with the NumPy-like symbolic tracing API, loaded from serialized ALIR JSON/JSON.GZ, or converted through plugins. The package includes top-level `keras`, `torch`, and `alkaid` example tracer plugins; Keras and Torch support can also be extended with second-level `alkaid_keras` and `alkaid_torch` entry points.

## Installation

```bash
pip install alkaid
```

Binary wheels are published for Linux x86_64 and macOS ARM64. Building from source requires Python 3.10 or newer, NumPy, `meson-python`, and a C++20 compiler with OpenMP support.

```bash
pip install --no-build-isolation -e '.[tests]'
```

Use `pip install 'alkaid[docs]'` for documentation dependencies. Install `xls-python` when using the XLS backend outside the test extra.

## CLI

```bash
alkaid convert model.keras /tmp/rtl_output --flavor verilog --latency-cutoff 5
alkaid convert model.json.gz /tmp/hls_output --flavor vitis
alkaid report /tmp/rtl_output --sort-by comb_metric
```

See the [documentation](https://calad0i.github.io/alkaid/) for tracing, plugin, ALIR, and backend details.

## License

LGPLv3. See the [LICENSE](LICENSE) file for details.
