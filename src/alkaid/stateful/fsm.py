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

from ..types import ALIR_SPEC_VERSION, CombLogic, JSONEncoder, Precision, QInterval, _quantize
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
        assert self.period > 0, 'Period must be positive'
        toggle = self.toggle
        assert max(toggle) - min(toggle) < self.period, f'Toggle values ({toggle}) must be within one period ({self.period})'
        self.bias = min(toggle)
        self.toggle = tuple((t - self.bias) for t in toggle)

    @cached_property
    def valid_mask(self) -> tuple[bool, ...]:
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
        return t >= self.bias and self.valid_mask[(t - self.bias) % self.period]

    def t_to_dense_idx(self, t: int) -> int:
        "t-th valid step to dense idx"
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
        period_count = idx // self.cum_valid_mask[-1]
        idx_in_period = self.cum_valid_mask.index((idx % self.cum_valid_mask[-1]) + 1)
        return period_count * self.period + idx_in_period + self.bias


class Signal:
    def __init__(
        self,
        name: str,
        exposed: bool,
        precisions: tuple[Precision, ...],
        rst_if: 'Signal | None' = None,
        rst_to: tuple[float, ...] | tuple[int, ...] | None = None,
        reg: bool = True,
        schedule: ModuloSchedule | None = None,
        mode: str = '',
        view_interval: tuple[int, int] | None = None,
        attrs: str | None = None,
    ):
        assert mode in ('', 'r', 'w', 'rw'), 'Mode must be one of "", "r", "w", "rw"'
        if mode in ('r', 'w') and not exposed:
            assert rst_to is not None
        self._rst_to: tuple[float, ...] | None = None
        if mode == 'r' and exposed:
            assert not reg, 'Input signal cannot be register. Capture to internal reg if needed'
        if rst_to is not None:
            assert len(rst_to) == len(precisions), 'Reset value length must match precision length'
            self._rst_to = tuple(_quantize(x, *kif) for x, kif in zip(rst_to, precisions))
        elif rst_if is not None:
            self._rst_to = tuple(0.0 for _ in precisions)
        self._precisions = tuple(Precision(*kif) for kif in precisions)

        self.name = name
        self.exposed = exposed
        self.rst_if = rst_if
        self.reg = reg
        self.schedule = schedule
        self.mode = mode
        self.view_interval = view_interval or (0, len(precisions))
        self.schedule = ModuloSchedule(*schedule) if isinstance(schedule, Sequence) else schedule
        self.attrs = attrs

    def __len__(self):
        return self.size

    def read(self):
        self.mode = 'r' + self.mode if 'r' not in self.mode else self.mode

    def write(self):
        self.mode = self.mode + 'w' if 'w' not in self.mode else self.mode

    @property
    def raw(self) -> 'Signal':
        if len(self.view_interval) == (0, len(self._precisions)):
            return self
        r = copy(self)
        r.view_interval = (0, len(self._precisions))
        return r

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signal):
            return False
        return self.__dict__ == other.__dict__

    @property
    def size(self) -> int:
        return len(self.precisions)

    @property
    def qint(self) -> tuple[QInterval, ...]:
        return tuple(prec.qint for prec in self.precisions)

    def to_dict(self) -> dict:
        return self.__dict__

    def to_list(self) -> list:
        """Serialize the full (un-sliced) signal to JSON-native types.

        ``rst_if`` is stored by name; the reference is re-linked on load.
        """
        return [
            self.name,
            self.exposed,
            [list(prec) for prec in self._precisions],
            self.rst_if.name if self.rst_if is not None else None,
            list(self._rst_to) if self._rst_to is not None else None,
            self.reg,
            self.schedule.to_list() if self.schedule is not None else None,
            self.mode,
            self.attrs,
        ]

    @classmethod
    def from_list(cls, lst: list) -> 'Signal':
        name, exposed, precisions, _rst_if_name, rst_to, reg, schedule, mode, attrs = lst
        return cls(
            name,
            exposed,
            tuple(Precision(*prec) for prec in precisions),
            rst_if=None,
            rst_to=tuple(rst_to) if rst_to is not None else None,
            reg=reg,
            schedule=schedule,
            mode=mode,
            attrs=attrs,
        )

    def __getitem__(self, idx: int | slice) -> 'Signal':
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        assert idx.step is None or idx.step == 1, 'Step is not supported for Signal slicing'
        r = copy(self)
        r.view_interval = (self.view_interval[0] + idx.start, self.view_interval[0] + idx.stop)
        return r

    @property
    def bitwidths(self) -> tuple[int, ...]:
        return tuple(sum(prec) for prec in self.precisions)

    @property
    def width(self) -> int:
        return sum(self.bitwidths)

    @property
    def precisions(self):
        return self._precisions[self.view_interval[0] : self.view_interval[1]]

    @property
    def rst_to(self):
        if self._rst_to is None:
            return None
        return self._rst_to[self.view_interval[0] : self.view_interval[1]]

    def __repr__(self):
        return f'Signal({self.name}[{self.view_interval[0]}:{self.view_interval[1]}])'


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
    sig_in = Signal(f'__{name}_in', False, prec_in, reg=False)
    sig_out = Signal(f'__{name}_out', False, prec_out, reg=False)
    return sig_in, sig_out


def _check_single_assignment(conns: Sequence[Conn]):
    """Raise if two conns drive overlapping bits of the same signal."""
    writes: dict[str, list[tuple[int, int, int]]] = {}
    for i, conn in enumerate(conns):
        lo, hi = conn.dst.view_interval
        writes.setdefault(conn.dst.name, []).append((lo, hi, i))
    for name, ws in writes.items():
        ws.sort()
        for (s0, e0, i0), (s1, e1, i1) in zip(ws, ws[1:]):
            if e0 > s1:
                raise ValueError(f'Double assignment on {name}[{max(s0, s1)}:{min(e0, e1)}] by conns #{i0} and #{i1}')


class Buffer(np.ndarray):
    def __new__(cls, sig: Signal, dtype=np.float64):
        obj = super().__new__(cls, sig.size, dtype)
        if sig.rst_to is not None:
            obj[:] = sig.rst_to
        obj._changed = False
        return obj

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
        const_signals[f'__{name}_out'] = Signal(
            f'__{name}_const',
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
        return const[sig.view_interval[0] : sig.view_interval[1]]

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
        self.logic = dict(logic)
        conns = tuple(conns)
        if not _sorted:
            logic, conns = _remove_const_logic(self.logic, conns)

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

        for conn in conns:
            for sig in conn.src, conn.dst, conn.enable_if, conn.alt_src:
                if sig is None or sig.name in signals:
                    continue
                signals[sig.name] = sig.raw

        for sig in list(signals.values()):
            while sig.rst_if is not None and sig.rst_if.name not in signals:
                sig = sig.rst_if
                signals[sig.name] = sig.raw

        for name, comb in self.logic.items():
            sig_in, sig_out = _comb_io_signals(name, comb)
            signals[sig_in.name] = sig_in
            signals[sig_out.name] = sig_out

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
        return cls(logic, conns, _sorted=True)

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
        if not (name.startswith('__') and name.endswith('_out')):
            return self.buffers[name]
        base = name[2:-4]
        sig_in_name = f'__{base}_in'
        val_in = self.buffers[sig_in_name]
        if not val_in._changed:
            return self.buffers[name]
        comb = self.fsm.logic[base]
        self.buffers[name][:] = comb.predict(val_in, n_threads=1, ignore_lookup_oob=True)
        return self.buffers[name]

    def _eval_conn(self, conn: Conn):
        if conn.enable_if is None:
            src = self._eval_buf(conn.src.name)
        else:
            if conn.alt_src is None:
                return  # no update
            src0, src1 = self._eval_buf(conn.src.name), self._eval_buf(conn.alt_src.name)  # type: ignore
            cond = self._eval_buf(conn.enable_if.name)[0]
            src = src0 if cond else src1
        if not self.buffers[conn.src.name]._changed:
            return
        dst = self.buffers[conn.dst.name]
        s_src, s_dst = slice(*conn.src.view_interval), slice(*conn.dst.view_interval)
        dst[s_dst] = src[s_src]

    def eval(self):
        for conn in self.fsm.comb_conns:
            self._eval_conn(conn)

    def tick(self):
        for conn in self.fsm.reg_conns:
            self._eval_conn(conn)
        self._t += 1

    def reset(self):
        for sig in self.fsm.signals.values():
            if sig.reg and sig.rst_to is not None:
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
        scheduled: bool = True,
        output_only: bool = True,
        extra_steps: int = 0,
    ) -> dict[str, np.ndarray]:

        t0 = self._t
        data = self.canonicalize_inp_data(data)

        for port in self.fsm.inp_signals:
            assert port.name in data, f'Missing input port {port.name} in data'

        if scheduled:
            for port in self.fsm.inp_signals + self.fsm.out_signals:
                assert port.schedule is not None, f'Port {port.name} does not have a schedule'

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
                    idx = self._t - t0
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

        self.reset()
        return self.run(data, extra_steps=extra_steps - 1, scheduled=True, output_only=True)
