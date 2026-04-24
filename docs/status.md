# Project Status

alkaid is under active development. The core ALIR format, CMVM optimization path, and RTL/HLS code generation flow are stable, but tracing support for individual framework operations may change as supported frontends expand.

```{note}
It is advised to always verify the results produced from the generated code.
```

## Supported Operations

Most common high-level operations that can be represented in [ALIR](alir.md) are supported, including:
 - Dense/Convolutional/EinsumDense layers
 - ReLU
 - max/minimum of two tensors; max/min pooling
 - element-wise addition/subtraction/multiplication
 - rearrangement of tensors (reshape, transpose, slicing, etc.)
 - fixed-point quantization
 - Arbitrary unary mapping through logic lookup tables (not to be confused with LUT primitives)
 - Sorting operations (sorting networks via bitonic/odd-even merge sort)
 - Bitwise operations (AND, OR, XOR, NOT, reduce-any, reduce-all)

```{note}
An experimental [XLS](https://google.github.io/xls/) backend is available for JIT execution and Verilog generation through `xls-python`. See the [getting started guide](getting_started.md) for details.
```
