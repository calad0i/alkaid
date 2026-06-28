Alkaid Low-Level Intermediate Representation (ALIR)
===================================================

ALIR is alkaid's low-level static-dataflow representation. A `CombLogic` program is a single SSA-style combinational block: each operation writes one buffer slot, later operations may read earlier slots, and outputs are selected from the final buffer.

The serialized JSON form written by `CombLogic.save()` is:

- `meta`: the string `ALIRModel`.
- `spec_version`: the ALIR spec version. The current version is `4`.
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

1. `addr`: buffer dependency indices.
2. `opcode`: operation code. See the operation code table below.
3. `data`: signed 64-bit integer payload tuple whose meaning depends on the operation.
4. `qint`: output quantization interval as `[min, max, step]`.
5. `latency`: estimated availability time.
6. `cost`: estimated operation cost.

For non-input operations, every `addr` index must refer only to an earlier operation.

## Operation Codes

- `-2`: Explicit negation.
  - `addr = (x,)`, `data = ()`.
  - `buf[i] = -buf[x]`
- `-1`: Copy from the external input buffer and quantize.
  - `addr = ()`, `data = (input_idx,)`.
  - `buf[i] = input[input_idx]`
- `0`: Addition.
  - `addr = (a, b)`, `data = (shift,)`.
  - `buf[i] = buf[a] + buf[b] * 2^shift`
- `1`: Subtraction.
  - `addr = (a, b)`, `data = (shift,)`.
  - `buf[i] = buf[a] - buf[b] * 2^shift`
- `2`: ReLU with output quantization.
  - `addr = (x,)`, `data = ()`.
  - `buf[i] = quantize(relu(buf[x]))`
- `3`: Output quantization.
  - `addr = (x,)`, `data = (relative_shift,)`.
  - `buf[i] = quantize(buf[x] * 2^relative_shift)`.
  - `relative_shift` records the source-to-wrap-buffer scale shift needed when tracing-level wrap resets its implementation `_factor`.
- `4`: Add a constant.
  - `addr = (x,)`, `data = (value, shift)`.
  - The constant is `value * 2^-shift`.
- `5`: Define a constant.
  - `addr = ()`, `data = (value,)`.
  - `buf[i] = value * qint.step`
- `6`: Mux by the most-significant bit of a condition value.
  - `addr = (true_idx, false_idx, cond_idx)`, `data = (false_shift,)`.
  - `buf[i] = MSB(buf[cond_idx]) ? buf[true_idx] : buf[false_idx] * 2^false_shift`, then quantized to `qint`.
- `7`: Multiplication.
  - `addr = (a, b)`, `data = ()`.
  - `buf[i] = buf[a] * buf[b]`
- `8`: Logic lookup table.
  - `addr = (x,)`, `data = (table_idx,)`.
  - Bytecode appends an internal second data entry for the lookup pad offset derived from `x`'s quantization interval.
- `9`: Unary bitwise operation.
  - `addr = (x,)`, `data = (subop,)`.
  - `subop = 0`: bitwise NOT.
  - `subop = 1`: reduce-any.
  - `subop = 2`: reduce-all.
- `10`: Binary bitwise operation.
  - `addr = (a, b)`, `data = (shift, subop)`.
  - `subop = 0`: AND.
  - `subop = 1`: OR.
  - `subop = 2`: XOR.
- `11`: Variadic signed shifted sum.
  - `addr = (x0, x1, ..., xN)`, `data = (sign0, shift0, sign1, shift1, ..., signN, shiftN)`.
  - `N >= 1`; each `signK` is `1` for `+` and `0` for `-`.
  - `buf[i] = sum((+1 if signK else -1) * buf[xK] * 2^shiftK for K in range(N + 1))`

Quantizing operations use direct fixed-point bit drop semantics: wrap for overflow and truncate for rounding.

## External Bytecode Representation

`CombLogic.to_bytecode()` produces the raw byte string consumed by the C++ ALIR interpreter. This is an in-memory interpreter format for Python -> C++ communication, not a stable on-disk format. The bytecode is further converted to another internal bytecode format in the C++ interpreter for faster dispatch, which is not described here.

The bytecode is little-endian and laid out sequentially:

1. Header: `magic`, `spec_version`, `n_inputs`, `n_outputs`, `n_ops`, `n_tables`, encoded as `<4sIIIII>`. `magic` is `ALIR`.
2. `inp_shifts`: `i32[n_inputs]`.
3. `out_idxs`: `i32[n_outputs]`.
4. `out_shifts`: `i32[n_outputs]`.
5. `out_negs`: `u8[n_outputs]`.
6. Per-operation records, each followed by its variable-length address and data arrays.
7. Tables: for each table, `u32 table_size` followed by `i32[table_size]`.

Each bytecode operation record starts with:

1. `opcode`: `i8`.
2. `signed`: `u8`.
3. `integers`: signed one-byte integer-bit count, excluding the sign bit.
4. `fractionals`: signed one-byte fractional-bit count.
5. `n_addr`: `u16`.
6. `n_data`: `u16`.
7. `addr`: `u32[n_addr]`.
8. `data`: `i64[n_data]`.

For opcode `8`, bytecode contains the semantic lookup table index plus an internal second data entry containing the derived lookup pad offset. JSON remains semantic and stores only `(table_idx,)`.

Lookup table data is stored in increasing lookup-index order. The bytecode loader validates the magic, ALIR spec version, bytecode length, generic address causality, ranges, EOF, and the interpreter's current 64-bit intermediate-width limit.

The JSON loader in the C++ interpreter accepts v4 plain JSON and gzip-compressed JSON with the same `ALIRModel` wrapper used by `CombLogic.save()`. v2 and v3 files can be loaded in Python, but the C++ JSON loader only accepts v4. Use `alkaid convert old.json[.gz] new.json[.gz] --flavor json` to convert on disk.
