# alkaid: HLS Compiler for Low-latency, Static-dataflow Kernels on FPGAs

[![Tests](https://img.shields.io/github/actions/workflow/status/calad0i/alkaid/unit-test.yml?label=test)](https://github.com/calad0i/alkaid/actions/workflows/unit-test.yml)
[![Documentation](https://img.shields.io/github/actions/workflow/status/calad0i/alkaid/sphinx-build.yml?label=doc)](https://calad0i.github.io/alkaid/)
[![PyPI version](https://img.shields.io/pypi/v/alkaid)](https://pypi.org/project/alkaid/)
[![ArXiv](https://img.shields.io/badge/arXiv-2507.04535-b31b1b.svg)](https://arxiv.org/abs/2507.04535)
[![Cov](https://img.shields.io/codecov/c/github/calad0i/alkaid)](https://codecov.io/gh/calad0i/alkaid)

alkaid is a light-weight high-level synthesis (HLS) compiler for generating low-latency, static-dataflow kernels for FPGAs. The main motivation of alkaid is to provide a simple and efficient way for machine learning practitioners requiring ultra-low latency to deploy their models on FPGAs quickly and easily, similar to hls4ml but with a much simpler design and better performance, both for the generated kernels and for the compilation process.

As a static dataflow compiler, alkaid is specialized for kernels that are equivalent to a combinational or fully pipelined logic circuit, which means that the kernel has no loops or has only fully unrolled loops. There is no specific limitation on the types of operations that can be used in the kernel. For resource sharing and time-multiplexing, the users are expected to use the generated kernels as building blocks and manually assemble them into a larger design. In the future, we may employ a XLS-like design to automate the communication and buffer instantiation between kernels, but for now we will keep it simple and let the users have full control over the design.

With DA in its name, alkaid do perform distributed arithmetic (DA) optimization to generate efficient kernels for linear DSP operations. The algorithm used is an efficient hybrid algorithm described in our [TRETS'25 paper](https://doi.org/10.1145/3777387). With DA optimization, any linear DSP operation can be implemented efficiently with only adders (i.e., fast accum and LUTs on FPGAs) without any hardened multipliers. If the user wishes, one can also control what multiplication pairs shall be excluded from DA optimization.


Installation
------------

```bash
pip install alkaid
```

Note: alkaid is now released as binary wheels on PyPI for Linux X86_64 and MacOS ARM64 platforms. For other platforms, please install from source. C++20 compliant compiler with OpenMP support is required to build alkaid from source. Windows is not officially supported, but you may try building it with MSVC or MinGW.

Getting Started
---------------

- See the [Getting Started](https://calad0i.github.io/alkaid/getting_started.html) guide for a quick introduction to using alkaid.
- See [JEDI-linear](https://github.com/calad0i/JEDI-linear) project which is based on alkaid

## License

LGPLv3. See the [LICENSE](LICENSE) file for details.

## Citation

If you use alkaid in a publication, please cite our [TRETS'25 paper](https://doi.org/10.1145/3777387) with the following bibtex entry:

```bibtex
@article{sun2025alkaid,
    author = {Sun, Chang and Que, Zhiqiang and Loncar, Vladimir and Luk, Wayne and Spiropulu, Maria},
    title = {alkaid: Distributed Arithmetic for Real-time Neural Networks on FPGAs},
    year = {2025},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issn = {1936-7406},
    url = {https://doi.org/10.1145/3777387},
    doi = {10.1145/3777387},
    journal = {ACM Trans. Reconfigurable Technol. Syst.},
    month = nov,
}
```
