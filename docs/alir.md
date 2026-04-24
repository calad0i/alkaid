Alkaid Low-Level Intermediate Representation (ALIR)
===================================================

ALIR is alkaid's low-level static-dataflow representation. A `CombLogic` program is a single SSA-style combinational block: each operation writes one buffer slot, later operations may read earlier slots, and outputs are selected from the final buffer.

The serialized JSON form written by `CombLogic.save()` is:

- `meta`: the string `ALIRModel`.
- `spec_version`: the ALIR spec version. The current version is `2`.
- `model`: the `CombLogic` payload described below.

## `CombLogic` Payload

The `model` payload is stored as an array in the same order as the `CombLogic` fields:

1. `shape`: `[n_inputs, n_outputs]`.
2. `inp_shifts`: input scale shifts.
3. `out_idxs`: buffer indices used as outputs. `-1` means a zero output.
4. `out_shifts`: output scale shifts.
5. `out_negs`: output sign flags.
6. `ops`: operation records.
7. `carry_size`: CMVM cost/latency configuration.
8. `adder_size`: CMVM cost/latency configuration.
9. `lookup_tables`: optional lookup table records, present only when lookup operations are used.

Each operation record is stored in the same order as the `Op` fields:

1. `id0`: first operand or input index.
2. `id1`: second operand, or `-1` when unused.
3. `opcode`: operation code. See the operation code table below.
4. `data`: signed 64-bit integer payload whose meaning depends on the operation.
5. `qint`: output quantization interval as `[min, max, step]`.
6. `latency`: estimated availability time.
7. `cost`: estimated operation cost.

Unused `id0` or `id1` fields must be `-1`. For non-input operations, operand indices must refer only to earlier operations. For opcode `6`, the condition index stored in `data_low` must also refer to an earlier operation.

## Operation Codes

- `-2`: Explicit negation.
  - `buf[i] = -buf[id0]`
- `-1`: Copy from the external input buffer and quantize.
  - `buf[i] = input[id0]`
- `0`: Addition.
  - `buf[i] = buf[id0] + buf[id1] * 2^data`
- `1`: Subtraction.
  - `buf[i] = buf[id0] - buf[id1] * 2^data`
- `2`: ReLU with output quantization.
  - `buf[i] = quantize(relu(buf[id0]))`
- `3`: Output quantization.
  - `buf[i] = quantize(buf[id0])`
- `4`: Add a constant.
  - `data_low` is a signed integer payload, `data_high` is a signed shift, and the constant is `data_low * 2^-data_high`.
- `5`: Define a constant.
  - `buf[i] = data * qint.step`
- `6`: Mux by the most-significant bit of a condition value.
  - `data_low` is the condition buffer index.
  - `data_high` is the shift applied to `id1`.
  - `buf[i] = MSB(buf[data_low]) ? buf[id0] : buf[id1] * 2^data_high`, then quantized to `qint`.
- `7`: Multiplication.
  - `buf[i] = buf[id0] * buf[id1]`
- `8`: Logic lookup table.
  - `data_low` is the lookup table index.
  - In bytecode, `data_high` stores the table pad offset derived from the producer quantization interval.
- `9`: Unary bitwise operation.
  - `data = 0`: bitwise NOT.
  - `data = 1`: reduce-any.
  - `data = 2`: reduce-all.
- `10`: Binary bitwise operation.
  - `data[31:0]` is the signed shift aligning operand 1 to operand 0.
  - `data[55:32]` is reserved.
  - `data[63:56]` is the sub-operation: `0` = AND, `1` = OR, `2` = XOR.

Quantizing operations use direct fixed-point bit drop semantics: wrap for overflow and truncate for rounding.

## External Bytecode Representation

`CombLogic.to_bytecode()` produces the int32 array consumed by the C++ ALIR interpreter. This is an in-memory interpreter format for python -> C++ communication, not a stable on-disk format. The bytecode is further converted to another internal bytecode format in the C++ interpreter for faster dispatch, which is not described here.

The int32 array layout is:

1. Header: `[spec_version, firmware_version, n_inputs, n_outputs, n_ops, n_tables]`.
2. `inp_shifts`: `int32[n_inputs]`.
3. `out_idxs`: `int32[n_outputs]`.
4. `out_shifts`: `int32[n_outputs]`.
5. `out_negs`: `int32[n_outputs]`.
6. `ops`: `int32[n_ops, 8]`.
7. `table_sizes`: `int32[n_tables]`.
8. `table_data`: concatenated `int32` lookup table contents.

Each bytecode operation row is:

1. `opcode`: `int32`.
2. `id0`: `int32`.
3. `id1`: `int32`.
4. `data_low`: low 32 bits of `data`.
5. `data_high`: high 32 bits of `data`.
6. `signed`: output signedness.
7. `integers`: integer bits excluding the sign bit.
8. `fractionals`: fractional bits.

Lookup table data is stored in increasing lookup-index order. The bytecode loader validates the ALIR spec version, bytecode length, causality, and the interpreter's current 64-bit intermediate-width limit.

The JSON loader in the C++ interpreter accepts plain JSON and gzip-compressed JSON with the same `ALIRModel` wrapper used by `CombLogic.save()`.
