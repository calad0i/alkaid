# Alkaid: A Library for Kernel Accelerator Implemented on Device

[![Tests](https://img.shields.io/github/actions/workflow/status/calad0i/alkaid/unit-test.yml?label=test)](https://github.com/calad0i/alkaid/actions/workflows/unit-test.yml)
[![Documentation](https://img.shields.io/github/actions/workflow/status/calad0i/alkaid/sphinx-build.yml?label=doc)](https://calad0i.github.io/alkaid/)
[![PyPI version](https://img.shields.io/pypi/v/alkaid)](https://pypi.org/project/alkaid/)
[![Cov](https://img.shields.io/codecov/c/github/calad0i/alkaid)](https://codecov.io/gh/calad0i/alkaid)

alkaid is a light-weight high-level synthesis (HLS) compiler for generating low-latency, static-dataflow kernels for FPGAs. The main motivation of alkaid is to provide a simple and efficient way for machine learning practitioners requiring ultra-low latency to deploy their models on FPGAs quickly and easily, similar to hls4ml but with a much simpler design and better performance, both for the generated kernels and for the compilation process.

As a static dataflow compiler, alkaid is specialized for kernels that are equivalent to a combinational or fully pipelined logic circuit, which means that the kernel has no loops or has only fully unrolled loops. There is no specific limitation on the types of operations that can be used in the kernel. For resource sharing and time-multiplexing, the users are expected to use the generated kernels as building blocks and manually assemble them into a larger design. In the future, we may employ a XLS-like design to automate the communication and buffer instantiation between kernels, but for now we will keep it simple and let the users have full control over the design.

Installation
------------

```bash
pip install alkaid
```

Note: alkaid is now released as binary wheels on PyPI for Linux X86_64 and MacOS ARM64 platforms. For other platforms, please install from source. C++20 compliant compiler with OpenMP support is required to build alkaid from source. Windows is not officially supported, but you may try building it with MSVC or MinGW.

## License

LGPLv3. See the [LICENSE](LICENSE) file for details.
