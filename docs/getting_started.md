# Getting Started with alkaid

alkaid can be used through the NumPy-like symbolic tracing API, through built-in framework tracers, or from serialized ALIR JSON/JSON.GZ. For standalone code generation, the functional API is the most direct path. For framework models, use the Keras or Torch tracer plugins when the operations in the model are supported.

## functional API:

The most flexible way to use alkaid is through its functional API/Explicit symbolic tracing. This allows users to define arbitrary operations using numpy-like syntax, and then trace the operations to generate synthesizable HDL or HLS code.

```python
# alkaid standalone example
import numpy as np

from alkaid.trace import FVArrayInput, trace
from alkaid.trace.ops import einsum, quantize, relu
from alkaid.codegen import HLSModel, RTLModel

w = np.random.randint(-2**7, 2**7, (4, 5, 6)) / 2**7

def operation(inp):
   inp = quantize(inp, 1, 7, 0) # Input must be quantized before any non-trivial operation
   out1 = relu(inp) # Only activation supported for now; can attach quantization at the same time

   # many native numpy operations are supported
   out2 = inp[:, 1:3].transpose()
   out2 = quantize(np.sin(out2), 1, 0, 7, 'SAT', 'RND')
   out2 = np.repeat(out2, 2, axis=0) * 3 + 4
   out2 = np.amax(np.stack([out2, -out2 * 2], axis=0), axis=0)

   out3 = quantize(out2 @ out1, 1, 10, 2) # can also be einsum here
   out = einsum('ijk,ij->ik', w, out3) # CMVM optimization is performed for all
   return out

# Replay the operation on symbolic tensor
inp = FVArrayInput((4, 5))
out = operation(inp)

# Generate pipelined Verilog code form the traced operation
# flavor can be 'verilog' or 'vhdl'. VHDL code generated will be in 2008 standard.
comb = trace(inp, out)
rtl_model = RTLModel(comb, '/tmp/rtl', flavor='verilog', latency_cutoff=5)
rtl_model.write()
# rtl_model.compile() # compile the generated Verilog code with verilator (with GHDL, if using vhdl)
# rtl_model.predict(data_inp) # run inference with the compiled model; bit-accurate

# Run bit-exact (all int64 arithmetic) inference with the combinational logic model
# Backed by C++-based ALIR interpreter for speed
# comb.predict(data_inp)
```

## Using framework plugins:

alkaid discovers top-level model tracers from the `alir_tracer.plugins` entry-point group. The package registers built-in tracers for `keras`, `torch`, and the `alkaid` example model. The Keras and Torch tracers can also load operation or layer handlers from second-level entry-point groups named `alkaid_keras` and `alkaid_torch`.

`trace_model()` chooses the top-level tracer from `type(model).__module__.split('.', 1)[0]` unless the `framework` argument is provided. See [Conversion Plugin](plugin.md) for the plugin contract and extension points.


## HGQ2/Keras3 integration:

For models defined in [HGQ2](https://github.com/calad0i/HGQ2) (Keras 3 based), alkaid uses its built-in Keras tracer and can load HGQ-specific layer handlers through the `alkaid_keras` second-level plugin group. When the model uses supported layers and operations, it can be converted to HDL or HLS code in the same flow as a plain Keras model.

> **Note**: HGQ2 is a separate package. Installing it provides the additional handlers needed for HGQ layers; the top-level `keras` tracer itself is provided by alkaid.

```python
# alkaid with HGQ2
import numpy as np
import keras
from hgq.layers import QEinsumDenseBatchnorm, QMaxPool1D
from alkaid.codegen import HLSModel, RTLModel
from alkaid.converter import trace_model
from alkaid.trace import trace

inp = keras.Input((4, 5))
out = QEinsumDenseBatchnorm('bij,jk->bik', (4,6), bias_axes='k', activation='relu')(inp)
out1 = QMaxPool1D(pool_size=2)(out)
out = keras.ops.concatenate([out, out1], axis=1)
out1, out2 = out[:, :3], out[:, 3:]
out = keras.ops.einsum('bik,bjk->bij', out1, out1 - out2[:,:1])
model = keras.Model(inp, out)

# Automatically replay the model operation on symbolic tensors
inp, out = trace_model(model)

comb = trace(inp, out)

... # The rest is the same as above
```

## RTL/HLS backends:

### RTL (Verilog/VHDL)

`RTLModel` generates synthesizable RTL and wraps a compiled simulation emulator for bit-accurate inference:

```python
from alkaid.codegen import RTLModel

# flavor='verilog' uses Verilator for simulation;
# 'vhdl' uses GHDL and Verilator chained (GHDL for VHDL to Verilog conversion, then Verilator for simulation)
rtl_model = RTLModel(comb, '/tmp/rtl', flavor='verilog', latency_cutoff=5, clock_period=5.0)
rtl_model.write()        # write RTL project to disk
rtl_model.compile()      # compile simulation emulator (requires Verilator or GHDL)
y = rtl_model.predict(x) # bit-accurate inference via compiled emulator
```

The generated project includes TCL build scripts for Vivado (`build_vivado_prj.tcl`) and Quartus (`build_quartus_prj.tcl`). Both `CombLogic` and `Pipeline` are supported.

### HLS (Vitis / HLSlib / oneAPI)

`HLSModel` generates HLS C++ code and wraps a compiled emulator:

```python
from alkaid.codegen import HLSModel

# flavor='vitis' (ap_types), 'hlslib' (ac_types/Intel), or 'oneapi'
hls_model = HLSModel(comb, '/tmp/hls', flavor='vitis', clock_period=5.0)
hls_model.write()        # write HLS project to disk
hls_model.compile()      # compile C++ emulator
y = hls_model.predict(x) # inference via compiled emulator
```

Note: only `CombLogic` is supported for HLS backends; `Pipeline` is not.


## XLS backend (experimental):


```{note}
`xls-python`, a Python binding for XLS, is required for the XLS backend. It can be installed from PyPI with `pip install xls-python`; binary wheel availability is platform dependent, so other platforms may need a source build.
```

For generating Verilog through [XLS](https://google.github.io/xls/), an experimental backend is available.

```python
from alkaid.codegen.xls import XLSModel

xls_model = XLSModel(comb) # Converts ALIR to XLS IR.
_ = xls_model.jit() # JIT-compile the XLS IR
y = xls_model.predict(data_inp) # Batched inference; bit-exact. No multithreading support for now.
verilog_text = xls_model.compile('/tmp/xls_output.v')
```

## CLI usage:

alkaid provides a command-line interface for common workflows:

```bash
# Convert a Keras/HGQ2 model to an RTL project
alkaid convert model.keras /tmp/rtl_output --flavor verilog --latency-cutoff 5

# Convert a serialized ALIR model to an RTL or HLS project
alkaid convert model.json /tmp/rtl_output --flavor vhdl --n-stages 3
alkaid convert model.json.gz /tmp/hls_output --flavor vitis

# Generate a resource/timing report from an existing RTL project
alkaid report /tmp/rtl_output --sort-by comb_metric
```

Use `alkaid convert --help` and `alkaid report --help` for full option details.

## hls4ml integration:

For existing uses of [hls4ml](https://github.com/fastmachinelearning/hls4ml), alkaid can be used as a strategy provider to enable the `distributed_arithmetic` strategy for supported layers (e.g., Dense, Conv, EinsumDense). This leverages the HLS codegen backend in alkaid to generate only the CMVM part of the design, while still using hls4ml for the rest of the design and integration. For any design aiming for `II>1` (i.e., not-fully unrolled), this is the recommended way to use alkaid.

```python
# alkaid with hls4ml
from hls4ml.converters import convert_from_keras_model

model_hls = convert_from_keras_model(
   model,
   hls_config={'Model': {'Strategy': 'distributed_arithmetic', ...}, ...},
   ...
)

model_hls.write()
```
