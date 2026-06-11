import ctypes
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from ...stateful import FSM, Signal
from ...types import Precision
from .rtl_model import at_path, canon_name, run_make_build, verilator_warn_suppression
from .verilog.comb import table_mem_gen
from .verilog.fsm import fsm_logic_gen, generate_io_wrapper

P_I64 = ctypes.POINTER(ctypes.c_int64)
P_F64 = ctypes.POINTER(ctypes.c_double)
PP_F64 = ctypes.POINTER(P_F64)
P_SIZE = ctypes.POINTER(ctypes.c_size_t)


def _verilator_ident(name: str) -> str:
    return name.replace('__', '___05F')


def _padded_precision(sig: Signal) -> Precision:
    kif = np.max(sig.precisions, axis=0)
    return Precision(bool(kif[0]), int(kif[1]), int(kif[2]))


def _normalize_dtype(dtype: Any | None, arr: np.ndarray | None = None) -> np.dtype:
    if dtype is None:
        if arr is not None and arr.dtype == np.dtype(np.int64):
            return np.dtype(np.int64)
        return np.dtype(np.float64)
    dt = np.dtype(dtype)
    assert dt in (np.dtype(np.float64), np.dtype(np.int64)), f'Unsupported dtype {dt}; expected np.float64 or np.int64'
    return dt


def _as_1d_values(sig: Signal, values: Any, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    assert arr.size == sig.size, f'Signal {sig.name} expects {sig.size} values, got {arr.size}'
    return np.ascontiguousarray(arr)


def _as_port_matrix(sig: Signal, values: np.ndarray) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1 and sig.size == 1:
        values = values.reshape(-1, 1)
    else:
        values = values.reshape(values.shape[0], sig.size)
    return np.ascontiguousarray(values, dtype=np.float64)


def _ptr(arr: np.ndarray, pointer_t):
    return arr.ctypes.data_as(pointer_t)


def fsm_config_gen(fsm: FSM, module_name: str) -> str:
    signals = fsm.inp_signals + fsm.out_signals
    n_inputs = len(fsm.inp_signals)
    n_outputs = len(fsm.out_signals)
    sizes_str = ', '.join(str(sig.size) for sig in signals)
    top_module = f'{module_name}_wrapper'
    inp_names = {port.name for port in fsm.inp_signals}
    name_to_id = {sig.name: i for i, sig in enumerate(signals)}

    # reset-control input ports (active-high).
    reset_ids: list[int] = []
    seen_resets: set[str] = set()
    for sig in fsm.signals.values():
        if not sig.reg or sig.rst_if is None:
            continue
        rst = sig.rst_if
        assert rst.size == 1 and rst.width == 1, f'Reset control {rst.name} must be a single bit'
        assert rst.name in inp_names, f'Reset control {rst.name} must be an exposed input port'
        if rst.name not in seen_resets:
            seen_resets.add(rst.name)
            reset_ids.append(name_to_id[rst.name])

    # Lambda caller for read/write of each signal. The caller gives a function that accepts (member, size, bw, signed, fractional).
    visit_cases = []
    for signal_id, sig in enumerate(signals):
        pp = _padded_precision(sig)
        bw = sum(pp)
        assert bw <= 64, f'Signal {sig.name} has a {bw}-bit element; int64 get/set supports at most 64 bits per element'
        member = f'dut->{_verilator_ident(sig.name)}'
        visit_cases.append(
            f'        case {signal_id}:\n'
            f'            fn({member}, '
            f'std::integral_constant<size_t, {sig.size}>{{}}, '
            f'std::integral_constant<size_t, {bw}>{{}}, '
            f'std::integral_constant<bool, {str(bool(pp.signed)).lower()}>{{}}, '
            f'std::integral_constant<int, {pp.fractional}>{{}});\n'
            f'            return;'
        )
    visit_cases_str = '\n'.join(visit_cases)

    reset_ids_decl = ''
    if reset_ids:
        reset_ids_decl = f'\n    static constexpr size_t reset_signal_ids[] = {{{", ".join(str(r) for r in reset_ids)}}};'

    sched_decls: list[str] = []
    sched_entries: list[str] = []
    for signal_id, sig in enumerate(signals):
        if sig.schedule is None:
            sched_entries.append('        {0, 1, 0, nullptr}')
            continue
        mask = ', '.join('1' if b else '0' for b in sig.schedule.valid_mask)
        sched_decls.append(f'    static constexpr uint8_t sched_mask_{signal_id}[] = {{{mask}}};')
        sched_entries.append(f'        {{1, {sig.schedule.period}, {sig.schedule.bias}, sched_mask_{signal_id}}}')
    sched_decls_str = ('\n'.join(sched_decls) + '\n') if sched_decls else ''
    sched_entries_str = ',\n'.join(sched_entries)

    return f"""#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "V{top_module}.h"
#include "fsm_wrapper.hh"

namespace {{
struct fsm_config {{
    using dut_t = V{top_module};

    // Port structure (signal ids are 0..n_inputs-1 for inputs, then n_inputs.. for outputs).
    static constexpr size_t n_inputs = {n_inputs};
    static constexpr size_t n_outputs = {n_outputs};
    static constexpr size_t signal_sizes[] = {{{sizes_str}}};

    static constexpr size_t n_reset_signals = {len(reset_ids)};{reset_ids_decl}

{sched_decls_str}    static constexpr fsm_schedule_config_t signal_schedules[] = {{
{sched_entries_str}
    }};

    template <typename Fn> static void visit_signal(dut_t *dut, size_t signal_id, Fn &&fn) {{
        switch (signal_id) {{
{visit_cases_str}
        default:
            assert(false && "Unknown FSM signal id");
        }}
    }}
}};

using fsm_config_t = fsm_config;
}}  // namespace
"""


class FSMProject:
    def __init__(
        self,
        fsm: FSM,
        path: str | Path,
        prj_name: str | None = None,
        print_latency: bool = False,
    ):
        self.fsm = fsm
        self._path = Path(path).resolve()
        self._prj_name = prj_name or canon_name(self._path.stem)
        self._print_latency = print_latency
        self.__src_root = Path(__file__).parent
        self._signals = self.fsm.inp_signals + self.fsm.out_signals
        self._signal_ids = {sig.name: i for i, sig in enumerate(self._signals)}
        self._uuid = None

    def write(self, metadata: None | dict[str, Any] = None, no_shreg: bool = False):
        (self._path / 'src/static').mkdir(parents=True, exist_ok=True)
        (self._path / 'src/memfiles').mkdir(parents=True, exist_ok=True)
        (self._path / 'sim').mkdir(parents=True, exist_ok=True)
        (self._path / 'model').mkdir(parents=True, exist_ok=True)

        codes = fsm_logic_gen(
            self.fsm,
            self._prj_name,
            print_latency=self._print_latency,
            timescale='`timescale 1 ns / 1 ps',
            no_shreg=no_shreg,
        )
        codes[f'{self._prj_name}_wrapper'] = generate_io_wrapper(self.fsm, self._prj_name)
        for name, code in codes.items():
            with open(self._path / f'src/{name}.v', 'w') as f:
                f.write(code)

        memfiles: dict[str, str] = {}
        for comb in self.fsm.logic.values():
            memfiles.update(table_mem_gen(comb))
        for name, mem in memfiles.items():
            with open(self._path / 'src/memfiles' / name, 'w') as f:
                f.write(mem)

        for path in self.__src_root.glob('verilog/source/*.v'):
            shutil.copy(path, self._path / 'src/static')

        with open(self._path / 'sim/fsm_config.hh', 'w') as f:
            f.write(fsm_config_gen(self.fsm, self._prj_name))

        shutil.copy(self.__src_root / 'common_source/build_fsm_binder.mk', self._path / 'sim')
        shutil.copy(self.__src_root / 'common_source/fsm_wrapper.hh', self._path / 'sim')
        shutil.copy(self.__src_root / 'common_source/fsm_binder.cc', self._path / 'sim')
        shutil.copy(self.__src_root / 'common_source/ioutil.hh', self._path / 'sim')

        self.fsm.save(self._path / 'model/fsm.json.gz')

        _metadata = {
            'flavor': 'verilog',
            'top_module': self._prj_name,
            'signal_count': len(self._signals),
        }
        if metadata is not None:
            _metadata.update({k: v for k, v in metadata.items() if k not in _metadata})
        with open(self._path / 'metadata.json', 'w') as f:
            json.dump(_metadata, f)

    def _compile(self, verbose=False, nproc: int | None = None, o3: bool = False, clean: bool = True):
        self._uuid = str(uuid4())
        env = os.environ.copy()
        env['VM_PREFIX'] = self._prj_name
        env['TOP_MODULE'] = f'{self._prj_name}_wrapper'
        env['STAMP'] = self._uuid
        env['VERILATOR_FLAGS'] = f'-Wall {verilator_warn_suppression()}'.strip()
        if nproc is not None:
            env['N_JOBS'] = str(nproc)

        stale = (
            re.compile(
                rf'^lib{re.escape(self._prj_name)}_fsm_[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}\.so$'
            )
            if clean
            else None
        )
        run_make_build(self._path / 'sim', 'build_fsm_binder.mk', env, fast=o3, clean=clean, verbose=verbose, stale_lib_re=stale)
        self._load_lib(self._uuid)

    def compile(
        self,
        verbose=False,
        nproc: int | None = None,
        o3: bool = False,
        clean: bool = True,
        metadata: None | dict[str, Any] = None,
        no_shreg: bool = False,
    ):
        self.write(metadata=metadata, no_shreg=no_shreg)
        self._compile(verbose=verbose, nproc=nproc, o3=o3, clean=clean)

    def _destroy(self):
        if not self._is_loaded():
            return
        self._lib.fsm_destroy(self._handle)
        del self._handle
        del self._lib
        self._uuid = None

    def _configure_lib(self):
        assert self._lib is not None
        specs = {
            'fsm_create': (ctypes.c_void_p, []),
            'fsm_destroy': (None, [ctypes.c_void_p]),
            'fsm_soft_reset': (None, [ctypes.c_void_p]),
            'fsm_eval': (None, [ctypes.c_void_p]),
            'fsm_tick': (None, [ctypes.c_void_p]),
            'fsm_time': (ctypes.c_size_t, [ctypes.c_void_p]),
            'fsm_set_signal': (None, [ctypes.c_void_p, ctypes.c_size_t, P_I64]),
            'fsm_get_signal': (None, [ctypes.c_void_p, ctypes.c_size_t, P_I64]),
            'fsm_set_signal_f64': (None, [ctypes.c_void_p, ctypes.c_size_t, P_F64]),
            'fsm_get_signal_f64': (None, [ctypes.c_void_p, ctypes.c_size_t, P_F64]),
            'fsm_run': (None, [ctypes.c_void_p, PP_F64, P_SIZE, PP_F64, ctypes.c_size_t, ctypes.c_uint8]),
        }
        for name, (restype, argtypes) in specs.items():
            fn = getattr(self._lib, name)
            fn.restype = restype
            fn.argtypes = argtypes

    def _load_lib(self, uuid: str | None = None):
        uuid = uuid if uuid is not None else self._uuid
        if uuid is None:
            libs = list((self._path / 'sim').glob(f'lib{self._prj_name}_fsm_*.so'))
            if len(libs) != 1:
                raise RuntimeError(f'Cannot load FSM library, found {len(libs)} libraries in {self._path / "sim"}')
            uuid = libs[0].name.removeprefix(f'lib{self._prj_name}_fsm_').removesuffix('.so')
        self._uuid = uuid
        lib_path = self._path / f'sim/lib{self._prj_name}_fsm_{uuid}.so'
        if not lib_path.exists():
            raise RuntimeError(f'Library {lib_path} does not exist')

        self._destroy()
        self._lib = ctypes.CDLL(str(lib_path))
        self._configure_lib()
        with at_path(self._path / 'src/memfiles'):
            self._handle = ctypes.c_void_p(self._lib.fsm_create())

    def _is_loaded(self):
        return hasattr(self, '_lib') and hasattr(self, '_handle')

    def _assert_loaded(self):
        assert self._is_loaded(), 'FSM library is not loaded; call compile() first'

    @property
    def t(self) -> int:
        self._assert_loaded()
        return int(self._lib.fsm_time(self._handle))

    def _signal(self, name: str) -> tuple[int, Signal]:
        signal_id = self._signal_ids[name]
        return signal_id, self._signals[signal_id]

    def eval(self):
        self._assert_loaded()
        self._lib.fsm_eval(self._handle)

    def tick(self):
        self._assert_loaded()
        self._lib.fsm_tick(self._handle)

    def soft_reset(self):
        self._assert_loaded()
        self._lib.fsm_soft_reset(self._handle)

    def set_port(self, name: str, values: Any, dtype: Any | None = None):
        self._assert_loaded()
        signal_id, sig = self._signal(name)
        dtype = _normalize_dtype(dtype, np.asarray(values))
        arr = _as_1d_values(sig, values, dtype)
        if dtype == np.dtype(np.float64):
            self._lib.fsm_set_signal_f64(self._handle, signal_id, _ptr(arr, P_F64))
        else:
            raw = np.ascontiguousarray(arr.astype(np.int64, copy=False))
            self._lib.fsm_set_signal(self._handle, signal_id, _ptr(raw, P_I64))

    def get_port(self, name: str, dtype: Any = np.float64, scalar: bool = False):
        self._assert_loaded()
        signal_id, sig = self._signal(name)
        dtype = _normalize_dtype(dtype)
        if dtype == np.dtype(np.float64):
            out = np.empty(sig.size, dtype=np.float64)
            self._lib.fsm_get_signal_f64(self._handle, signal_id, _ptr(out, P_F64))
        else:
            out = np.empty(sig.size, dtype=np.int64)
            self._lib.fsm_get_signal(self._handle, signal_id, _ptr(out, P_I64))
        if scalar:
            assert sig.size == 1, f'Signal {sig.name} has size {sig.size}; scalar=True is only accepted for size-1 signals'
            return out[0].item()
        return out

    def canonicalize_inp_data(
        self, data: Mapping[str, np.ndarray] | Sequence[np.ndarray] | Sequence[Mapping[str, np.ndarray]] | np.ndarray
    ) -> dict[str, np.ndarray]:
        datamap: dict[str, np.ndarray]
        if isinstance(data, np.ndarray):
            assert len(self.fsm.inp_signals) == 1, 'Data array provided for multiple input signals'
            datamap = {self.fsm.inp_signals[0].name: data}
        elif isinstance(data, Sequence) and not isinstance(data, Mapping):
            assert len(data) > 0, 'Data sequence cannot be empty'
            _data = data[0]
            if isinstance(_data, Mapping):
                datamap = {k: np.concatenate([d[k] for d in data]) for k in _data.keys()}
            else:
                assert isinstance(_data, np.ndarray)
                assert len(data) == len(self.fsm.inp_signals)
                datamap = {port.name: data[i] for i, port in enumerate(self.fsm.inp_signals)}  # type: ignore
        else:
            assert isinstance(data, Mapping)
            datamap = {k: np.asarray(v) for k, v in data.items()}
        return {k: np.asarray(v, dtype=np.float64) for k, v in datamap.items()}

    def run(
        self,
        data: Mapping[str, np.ndarray] | Sequence[np.ndarray] | Sequence[Mapping[str, np.ndarray]] | np.ndarray,
        steps: int | None = None,
        scheduled: bool | None = None,
        output_only: bool = True,
        extra_steps: int = 0,
    ) -> dict[str, np.ndarray]:
        self._assert_loaded()
        assert output_only, 'Internal tracing is not supported; expose debug signals as output ports'

        t0 = self.t
        data = self.canonicalize_inp_data(data)

        for port in self.fsm.inp_signals:
            assert port.name in data, f'Missing input port {port.name} in data'

        is_scheduled = True
        for port in self.fsm.inp_signals + self.fsm.out_signals:
            is_scheduled &= port.schedule is not None
        if scheduled is None:
            scheduled = is_scheduled
        if not is_scheduled and scheduled:
            raise ValueError('Cannot run in scheduled mode when not all signals have schedules')

        data = {port.name: _as_port_matrix(port, data[port.name]) for port in self.fsm.inp_signals}

        if steps is None:
            if scheduled:
                steps = min(port.schedule.dense_idx_to_t(len(data[port.name]) - 1) for port in self.fsm.inp_signals) + 1  # type: ignore
            else:
                steps = min(len(data[port.name]) for port in self.fsm.inp_signals)

        total_steps = steps + extra_steps

        # Per-port sample buffers, in the config's signal order (inputs first, then outputs)
        input_data = (P_F64 * len(self.fsm.inp_signals))()
        input_n_samples = (ctypes.c_size_t * len(self.fsm.inp_signals))()
        for i, port in enumerate(self.fsm.inp_signals):
            samples = data[port.name]
            input_data[i] = _ptr(samples, P_F64)
            input_n_samples[i] = samples.shape[0]

        results = dict[str, NDArray[np.float64]]()
        output_data = (P_F64 * len(self.fsm.out_signals))()
        for j, port in enumerate(self.fsm.out_signals):
            if scheduled:
                n_outputs = port.schedule.n_valid_samples_between(t0, t0 + total_steps)  # type: ignore
            else:
                n_outputs = total_steps
            out = np.empty((n_outputs, port.size), dtype=np.float64)
            results[port.name] = out
            output_data[j] = _ptr(out, P_F64)

        self._lib.fsm_run(self._handle, input_data, input_n_samples, output_data, total_steps, int(scheduled))

        return results

    def predict(
        self,
        data: Mapping[str, np.ndarray] | Sequence[np.ndarray] | Sequence[Mapping[str, np.ndarray]] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        _period = set()
        for port in self.fsm.inp_signals + self.fsm.out_signals:
            assert port.schedule is not None, f'Port {port.name} does not have a schedule'
            _period.add(port.schedule.period)
        assert len(_period) == 1, 'All signals must have the same schedule period'
        extra_steps = max(port.schedule.bias for port in self.fsm.out_signals)  # type: ignore

        self.soft_reset()
        return self.run(data, extra_steps=max(extra_steps - 1, 0), scheduled=True, output_only=True)

    def __del__(self):
        self._destroy()
