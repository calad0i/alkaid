import gzip
import json
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TypeVar

import numpy as np

from ..types import ALIR_SPEC_VERSION, CombLogic, JSONEncoder, Precision, _quantize
from .ordering import topo_check_and_sort


class Dir(Enum):
    IN = 1
    OUT = -1
    INTERNAL = 0


@dataclass
class ModuloSchedule:
    toggle: tuple[int, ...]
    period: int
    bias: int = field(init=False)

    def __post_init__(self):
        assert self.period >= 0, 'Period must be non-negative'
        toggle = self.toggle
        if self.period == 0:
            self.bias = 0
            self.toggle = (0,)
            return
        assert max(toggle) - min(toggle) < self.period, f'Toggle values ({toggle}) must be within one period ({self.period})'
        self.bias = min(toggle)
        self.toggle = tuple((t - self.bias) for t in toggle)

    @cached_property
    def valid_mask(self) -> tuple[bool, ...]:
        if self.period == 0:
            return (True,)
        valid = np.searchsorted(self.toggle, np.arange(self.period), side='right') % 2 == 1
        return valid.tolist()

    @cached_property
    def cum_valid_mask(self) -> tuple[int, ...]:
        cum_valid = np.cumsum(self.valid_mask)
        return cum_valid.tolist()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModuloSchedule):
            return False
        return self.toggle == other.toggle and self.period == other.period and self.bias == other.bias

    def to_list(self):
        return [tuple(t + self.bias for t in self.toggle), self.period]

    def check(self, t: int) -> bool:
        if self.period == 0:
            return True
        return t >= self.bias and self.valid_mask[(t - self.bias) % self.period]

    def t_to_dense_idx(self, t: int) -> int:
        "t-th valid step to dense idx"
        if self.period == 0:
            return t
        t = t - self.bias
        return self.cum_valid_mask[t % self.period] + (t // self.period) * self.cum_valid_mask[-1] - 1

    def n_valid_samples_between(self, t0: int, t1: int) -> int:
        "Number of valid steps between t0 (inclusive) and t1 (exclusive)"
        if t1 <= t0:
            return 0
        i0, i1 = self.t_to_dense_idx(t0), self.t_to_dense_idx(t1)
        return max(i1 + 1, 0) - max(i0 + 1, 0)

    def dense_idx_to_t(self, idx: int) -> int:
        "idx-th valid step to t"
        if self.period == 0:
            return idx
        period_count = idx // self.cum_valid_mask[-1]
        idx_in_period = self.cum_valid_mask.index((idx % self.cum_valid_mask[-1]) + 1)
        return period_count * self.period + idx_in_period + self.bias


class Signal:
    def __init__(
        self,
        name: str,
        exposed: bool,
        precisions: tuple[Precision, ...] | tuple[tuple[bool | int, int, int], ...],
        rst_if: 'Signal | None' = None,
        rst_to: tuple[float, ...] | tuple[int, ...] | None = None,
        reg: bool = True,
        schedule: ModuloSchedule | None = None,
        mode: str = '',
        view: tuple[int, int] | None = None,
        attrs: str | None = None,
        _dynamic_bias: 'tuple[Signal, int] | None' = None,
    ):
        assert mode in ('', 'r', 'w', 'rw'), 'Mode must be one of "", "r", "w", "rw"'
        self._rst_to: tuple[float, ...] | None = None
        if mode == 'r' and exposed:
            assert not reg, 'Input signal cannot be register. Capture to internal reg if needed'
        if rst_to is not None:
            assert len(rst_to) == len(precisions), 'Reset value length must match precision length'
            self._rst_to = tuple(_quantize(x, *kif) for x, kif in zip(rst_to, precisions))
        elif rst_if is not None:
            self._rst_to = tuple(0.0 for _ in precisions)
        self._precisions = tuple(Precision(bool(kif[0]), kif[1], kif[2]) for kif in precisions)

        view = view or (0, len(precisions))
        if _dynamic_bias is not None:
            idx, mult = _dynamic_bias
            assert idx.size == 1
            prec = idx.precisions[0]
            assert not prec.signed and prec.fractional == 0
            i, j = view
            _prec0 = self._precisions[i:j]
            _prec1 = self._precisions[i : i + mult]
            n = j - i
            N = len(self._precisions) - i
            while N >= i + n:
                assert _prec0 == self._precisions[i : i + n] and _prec1 == self._precisions[i : i + mult], (
                    'Dynamic bias requires repeated patterns in precision'
                )
                i += mult

        self.name = name
        self.exposed = exposed
        self.rst_if = rst_if
        self.reg = reg
        self.schedule = schedule
        self.mode = mode
        self._view = view
        self.schedule = ModuloSchedule(*schedule) if isinstance(schedule, Sequence) else schedule
        self.attrs = attrs
        self._dynamic_bias = _dynamic_bias

    @property
    def jump_width(self) -> int:
        if self._dynamic_bias is None:
            return 0
        _, mult = self._dynamic_bias
        return sum(map(sum, self._precisions[self.view[0] : self.view[0] + mult]))

    @property
    def view(self) -> tuple[int, int]:
        return self._view

    @property
    def raw(self) -> 'Signal':
        if self.view == (0, len(self._precisions)):
            return self
        r = copy(self)
        r._view = (0, len(self._precisions))
        return r

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signal):
            return False
        return self.__dict__ == other.__dict__

    @property
    def size(self) -> int:
        return len(self.precisions)

    def to_list(self) -> list:
        """Serialize the signal to JSON-native types.

        ``rst_if`` is stored by name; the reference is re-linked on load.
        """
        return [
            self.name,
            self.exposed,
            [list(prec) for prec in self._precisions],
            self.rst_if.to_list() if self.rst_if is not None else None,
            list(self._rst_to) if self._rst_to is not None else None,
            self.reg,
            self.schedule.to_list() if self.schedule is not None else None,
            self.mode,
            self.attrs,
            list(self.view),
            [self._dynamic_bias[0].to_list(), self._dynamic_bias[1]] if self._dynamic_bias is not None else None,
        ]

    @classmethod
    def from_list(cls, lst: list) -> 'Signal':
        name, exposed, precisions, _rst_if_name, rst_to, reg, schedule, mode, attrs, view, _dynamic_bias = lst
        return cls(
            name,
            exposed,
            tuple(Precision(*prec) for prec in precisions),
            rst_if=Signal.from_list(_rst_if_name) if _rst_if_name is not None else None,
            rst_to=tuple(rst_to) if rst_to is not None else None,
            reg=reg,
            schedule=schedule,
            mode=mode,
            view=tuple(view),
            attrs=attrs,
            _dynamic_bias=(Signal.from_list(_dynamic_bias[0]), _dynamic_bias[1]) if _dynamic_bias is not None else None,
        )

    def __getitem__(self, idx: int | slice) -> 'Signal':
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        assert idx.step is None or idx.step == 1, 'Step is not supported for Signal slicing'
        r = copy(self)
        r._view = (self.view[0] + idx.start, self.view[0] + idx.stop)
        return r

    @property
    def bitwidths(self) -> tuple[int, ...]:
        return tuple(sum(prec) for prec in self.precisions)

    @property
    def width(self) -> int:
        return sum(self.bitwidths)

    @property
    def precisions(self):
        return self._precisions[self.view[0] : self.view[1]]

    @property
    def rst_to(self):
        if self._rst_to is None:
            return None
        return self._rst_to[self.view[0] : self.view[1]]

    def __repr__(self):
        return f'Signal({self.name}[{self.view[0]}:{self.view[1]}])'

    def read(self):
        if 'r' not in self.mode:
            self.mode = 'r' + self.mode

    def write(self):
        if 'w' not in self.mode:
            self.mode = self.mode + 'w'


@dataclass
class Conn:
    src: Signal
    dst: Signal
    enable_if: Signal | None = None
    alt_src: Signal | None = None

    def __post_init__(self):
        if self.enable_if is not None:
            assert self.enable_if.size == 1 and self.enable_if.precisions[0] == Precision(False, 1, 0), (
                'Enable signal must be a single-bit boolean'
            )

        assert self.src.size == self.dst.size, 'Source and destination views must have the same width'

        if self.alt_src is not None:
            assert self.alt_src.size == self.dst.size, 'Alternative source view must match destination width'
            assert self.src.reg == self.alt_src.reg, 'Source and alternative source must both be registers or both be wires'

    @property
    def clocked(self) -> bool:
        return self.dst.reg

    def to_dict(self) -> dict:
        return {
            'src': self.src.to_list(),
            'dst': self.dst.to_list(),
            'enable_if': self.enable_if.to_list() if self.enable_if is not None else None,
            'alt_src': self.alt_src.to_list() if self.alt_src is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Conn':
        return cls(
            Signal.from_list(d['src']),
            Signal.from_list(d['dst']),
            Signal.from_list(d['enable_if']) if d['enable_if'] is not None else None,
            Signal.from_list(d['alt_src']) if d['alt_src'] is not None else None,
        )


def _comb_io_signals(name: str, comb: CombLogic) -> tuple[Signal, Signal]:
    prec_in = tuple(qint.kif for qint in comb.inp_qint)
    prec_out = tuple(qint.kif for qint in comb.out_qint)
    sig_in = Signal(f'INTERNAL_{name}_inp', False, prec_in, reg=False, mode='r')
    sig_out = Signal(f'INTERNAL_{name}_outp', False, prec_out, reg=False, mode='w')
    return sig_in, sig_out


def _check_single_assignment(conns: Sequence[Conn]):
    """Raise if two conns drive overlapping bits of the same signal."""
    writes: dict[str, list[tuple[int, int, int]]] = {}
    for i, conn in enumerate(conns):
        lo, hi = conn.dst.view
        writes.setdefault(conn.dst.name, []).append((lo, hi, i))
    for name, ws in writes.items():
        ws.sort()
        for (s0, e0, i0), (s1, e1, i1) in zip(ws, ws[1:]):
            assert e0 < s1, f'Double assignment on {name}[{max(s0, s1)}:{min(e0, e1)}] by conns #{i0} and #{i1}'


class Buffer(np.ndarray):
    def __new__(cls, sig: Signal, dtype=np.float64):
        obj = super().__new__(cls, sig.size, dtype)
        obj._changed = False
        if sig.rst_to is not None:
            obj[:] = sig.rst_to
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self._changed = getattr(obj, '_changed', False)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._changed = True


def _remove_const_logic(logic: dict[str, CombLogic], conns: Sequence[Conn]) -> tuple[dict[str, CombLogic], tuple[Conn, ...]]:

    _logic, conns = copy(logic), copy(conns)
    const_signals: dict[str, Signal] = {}
    for name, comb in logic.items():
        if comb.shape[0] != 0:
            continue
        rst_to = tuple(map(float, comb([], quantize=False)))
        precisions = tuple(qint.kif for qint in comb.out_qint)
        const_signals[f'INTERNAL_{name}_outp'] = Signal(
            f'INTERNAL_{name}_const',
            exposed=False,
            precisions=precisions,
            rst_if=None,
            rst_to=rst_to,
            reg=True,
            mode='r',
        )
        del _logic[name]

    if not const_signals:
        return logic, tuple(conns)
    logic = _logic

    T = TypeVar('T', Signal, None)

    def remap_sig(sig: T) -> T:
        if sig is None or sig.name not in const_signals:
            return sig
        const = const_signals[sig.name]
        return const[sig.view[0] : sig.view[1]]

    def remap_conn(conn: Conn):
        return Conn(
            remap_sig(conn.src),
            conn.dst,
            enable_if=remap_sig(conn.enable_if),  # type: ignore
            alt_src=remap_sig(conn.alt_src),  # type: ignore
        )

    conns = tuple(remap_conn(conn) for conn in conns)
    return logic, conns


class FSM:
    """Finite State Machine representation with combinational logic and register connections."""

    def __init__(
        self,
        logic: dict[str, CombLogic],
        conns: tuple[Conn, ...],
        _sorted=False,
    ):
        if not _sorted:
            logic, conns = _remove_const_logic(logic, conns)
        self.logic = logic

        for conn in conns:
            conn.src.read()
            conn.dst.write()
            if conn.enable_if is not None:
                conn.enable_if.read()
            if conn.alt_src is not None:
                conn.alt_src.read()

        comb_conns = tuple(conn for conn in conns if not conn.clocked)
        self.reg_conns = tuple(conn for conn in conns if conn.clocked)

        self._set_signals(conns)

        if not _sorted:
            comb_conns = tuple(topo_check_and_sort(comb_conns))
        self.comb_conns = comb_conns
        _check_single_assignment(self.reg_conns)

        self._has_emu = False

    @property
    def inp_signals(self) -> tuple[Signal, ...]:
        return tuple(sig for sig in self.signals.values() if sig.exposed and sig.mode == 'r')

    @property
    def out_signals(self) -> tuple[Signal, ...]:
        return tuple(sig for sig in self.signals.values() if sig.exposed and 'w' in sig.mode)

    @property
    def internal_signals(self) -> tuple[Signal, ...]:
        return tuple(sorted([sig for sig in self.signals.values() if not sig.exposed], key=lambda s: (s.reg, s.name)))

    def _set_signals(self, conns: Sequence[Conn]):
        signals: dict[str, Signal] = {}

        seen_refs: set[int] = set()

        def _add_signal(sig: Signal):
            if id(sig) in seen_refs:
                return
            seen_refs.add(id(sig))
            if sig.name not in signals:
                signals[sig.name] = sig.raw
            else:
                if 'r' in sig.mode:
                    signals[sig.name].read()
                if 'w' in sig.mode:
                    signals[sig.name].write()
            if sig.rst_if is not None:
                _add_signal(sig.rst_if)
            if sig._dynamic_bias is not None:
                bias_sig, _ = sig._dynamic_bias
                _add_signal(bias_sig)

        for conn in conns:
            for sig in conn.src, conn.dst, conn.enable_if, conn.alt_src:
                if sig is not None:
                    _add_signal(sig)

        # for name, comb in self.logic.items():
        #     sig_in, sig_out = _comb_io_signals(name, comb)
        #     if sig_in.name not in signals:
        #         signals[sig_in.name] = sig_in
        #     if sig_out.name not in signals:
        #         signals[sig_out.name] = sig_out
        # assert sig_in.name in signals, f'Comb logic {name} input signal {sig_in.name} not found in connections'
        # assert sig_out.name in signals, f'Comb logic {name} output signal {sig_out.name} not found in connections'

        self.signals = signals

    @property
    def wires(self) -> dict[str, Signal]:
        return {name: sig for name, sig in self.signals.items() if not sig.reg}

    @property
    def regs(self) -> dict[str, Signal]:
        return {name: sig for name, sig in self.signals.items() if sig.reg}

    def to_dict(self) -> dict:
        return {
            'meta': 'ALIRFSM',
            'spec_version': ALIR_SPEC_VERSION,
            'fsm': {
                'logic': dict(self.logic),
                'conns': self.comb_conns + self.reg_conns,
                'signal_order': list(self.signals),
            },
        }

    @classmethod
    def from_dict(cls, d: dict, raw: bool = False) -> 'FSM':
        if not raw:
            assert d['meta'] == 'ALIRFSM', 'Not an ALIR FSM'
            assert d['spec_version'] == ALIR_SPEC_VERSION, f'Unsupported ALIR spec version {d["spec_version"]}'
            d = d['fsm']

        logic = {name: CombLogic.from_dict(cl, raw=True) for name, cl in d['logic'].items()}
        conns = tuple(Conn.from_dict(c) for c in d['conns'])
        fsm = cls(logic, conns, _sorted=True)
        fsm.signals = {name: fsm.signals[name] for name in d['signal_order']}
        return fsm

    def save(self, path: str | Path, compresslevel: int = 6):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if str(path).endswith('.gz'):
            with gzip.open(path, 'wt', encoding='utf-8', compresslevel=compresslevel) as f:
                json.dump(self, f, cls=JSONEncoder, separators=(',', ':'))
        else:
            with open(path, 'w') as f:
                json.dump(self, f, cls=JSONEncoder, separators=(',', ':'))

    @classmethod
    def load(cls, path: str | Path):
        """Load from a JSON file; accepts gzip (detected by magic bytes)."""
        with open(path, 'rb') as fb:
            head = fb.read(2)
            fb.seek(0)
            if head == b'\x1f\x8b':  # gzip magic bytes
                data = json.loads(gzip.decompress(fb.read()).decode('utf-8'))
            else:
                data = json.loads(fb.read().decode('utf-8'))
        return cls.from_dict(data)

    def _init_emu(self):
        if not self._has_emu:
            self._emu = FSMEmu(self)
            self._has_emu = True

    def run(
        self,
        data: Mapping[str, np.ndarray] | Sequence[np.ndarray] | Sequence[Mapping[str, np.ndarray]] | np.ndarray,
        steps: int | None = None,
        scheduled: bool | None = None,
        output_only: bool = True,
        extra_steps: int = 0,
    ) -> dict[str, np.ndarray]:
        self._init_emu()
        return self._emu.run(data, steps=steps, scheduled=scheduled, output_only=output_only, extra_steps=extra_steps)

    def predict(
        self, data: Mapping[str, np.ndarray] | Sequence[np.ndarray] | Sequence[Mapping[str, np.ndarray]] | np.ndarray
    ) -> dict[str, np.ndarray]:
        self._init_emu()
        return self._emu.predict(data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FSM):
            return False
        return self.logic == other.logic and self.comb_conns == other.comb_conns and self.reg_conns == other.reg_conns


class FSMEmu:
    def __init__(self, fsm: FSM):
        self.fsm = fsm
        self._t = 0
        self._init_buffers()

    def _init_buffers(self):
        self.buffers = {sig.name: Buffer(sig, dtype=np.float64) for sig in self.fsm.signals.values()}

    def _eval_buf(self, name: str):
        if not (name.startswith('INTERNAL_') and name.endswith('_outp')):
            return self.buffers[name]
        base = name.removeprefix('INTERNAL_').removesuffix('_outp')
        sig_in_name = f'INTERNAL_{base}_inp'
        val_in = self.buffers[sig_in_name]
        if not val_in._changed:
            return self.buffers[name]
        comb = self.fsm.logic[base]
        self.buffers[name][:] = comb.predict(val_in, n_threads=1, ignore_lookup_oob=True)
        return self.buffers[name]

    def _get_bias(self, sig: Signal) -> int:
        if sig._dynamic_bias is None:
            return 0
        bias_sig, mult = sig._dynamic_bias
        bias_val = self.buffers[bias_sig.name][bias_sig.view[0]]
        return int(bias_val) * mult

    def _is_reset(self, sig: Signal) -> bool:
        if sig.rst_if is None:
            return False
        rst_val = self.buffers[sig.rst_if.name][sig.rst_if.view[0]]
        return bool(rst_val)  # active-high: reset asserted when the rst_if line is 1

    def _eval_conn(self, conn: Conn, staged: dict[str, Buffer] | None = None):
        staged = staged if staged is not None else self.buffers
        if self._is_reset(conn.dst):
            return
        src_sig = conn.src
        if conn.enable_if is not None:
            en = conn.enable_if
            cond = self._eval_buf(en.name)[en.view[0] + self._get_bias(en)]
            if not cond:
                if conn.alt_src is None:
                    return  # load-enable hold
                src_sig = conn.alt_src
        src = self._eval_buf(src_sig.name)
        if not self.buffers[src_sig.name]._changed:
            return
        dst = staged[conn.dst.name]
        _bias_src, _bias_dst = self._get_bias(src_sig), self._get_bias(conn.dst)
        s_src = slice(src_sig.view[0] + _bias_src, src_sig.view[1] + _bias_src)
        s_dst = slice(conn.dst.view[0] + _bias_dst, conn.dst.view[1] + _bias_dst)
        dst[s_dst] = src[s_src]

    def eval(self):
        for conn in self.fsm.comb_conns:
            self._eval_conn(conn)

    def tick(self):
        staged: dict[str, Buffer] = {sig.name: self.buffers[sig.name].copy() for sig in self.fsm.signals.values() if sig.reg}
        for conn in self.fsm.reg_conns:
            self._eval_conn(conn, staged)
        for name, buf in staged.items():
            if self._is_reset(self.fsm.signals[name]):
                rst_to = self.fsm.signals[name].rst_to
                self.buffers[name][:] = rst_to
            else:
                self.buffers[name][:] = buf
                self.buffers[name]._changed = buf._changed
        self._t += 1

    def soft_reset(self):
        for sig in self.fsm.signals.values():
            if sig.reg and sig.rst_to is not None and sig.rst_if is not None and sig.rst_if.exposed:
                self.buffers[sig.name][:] = sig.rst_to
        self._t = 0

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
        return datamap

    def run(
        self,
        data: Mapping[str, np.ndarray] | Sequence[np.ndarray] | Sequence[Mapping[str, np.ndarray]] | np.ndarray,
        steps: int | None = None,
        scheduled: bool | None = None,
        output_only: bool = True,
        extra_steps: int = 0,
    ) -> dict[str, np.ndarray]:

        t0 = self._t
        data = self.canonicalize_inp_data(data)

        for port in self.fsm.inp_signals:
            assert port.name in data, f'Missing input port {port.name} in data'

        all_scheduled = all(port.schedule is not None for port in self.fsm.inp_signals + self.fsm.out_signals)
        scheduled = scheduled if scheduled is not None else all_scheduled
        if scheduled:
            assert all_scheduled, 'Scheduled run requires all input and output signals to have schedules'

        if not steps:
            if scheduled:
                steps = min(port.schedule.dense_idx_to_t(len(data[port.name]) - 1) for port in self.fsm.inp_signals) + 1  # type: ignore
            else:
                steps = min(len(data[port.name]) for port in self.fsm.inp_signals)

        results = dict[str, np.ndarray]()
        for port in self.fsm.out_signals:
            if scheduled:
                n_outputs = port.schedule.n_valid_samples_between(t0, t0 + steps + extra_steps)  # type: ignore
            else:
                n_outputs = steps + extra_steps
            results[port.name] = np.empty((n_outputs, port.size), dtype=np.float64)
        if not output_only:
            for port in self.fsm.internal_signals:
                results[port.name] = np.empty((steps + extra_steps, port.size), dtype=np.float64)

        for _ in range(steps + extra_steps):
            for port in self.fsm.inp_signals:
                if scheduled:
                    if not port.schedule.check(self._t):  # type: ignore
                        continue
                    idx = port.schedule.n_valid_samples_between(t0, self._t + 1) - 1  # type: ignore
                else:
                    idx = self._t - t0
                if idx < len(data[port.name]):
                    self.buffers[port.name][:] = data[port.name][idx]

            self.eval()
            self.tick()
            self.eval()

            for port in self.fsm.out_signals:
                if scheduled:
                    if not port.schedule.check(self._t):  # type: ignore
                        continue
                    idx = port.schedule.n_valid_samples_between(t0, self._t) - 1  # type: ignore
                else:
                    idx = self._t - t0 - 1
                results[port.name][idx] = self.buffers[port.name]
            if not output_only:
                for port in self.fsm.internal_signals:
                    results[port.name][self._t] = self.buffers[port.name]

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
