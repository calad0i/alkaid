import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import NamedTuple
from warnings import warn

import numpy as np

from ..types import ALIR_SPEC_VERSION, CombLogic, JSONEncoder, Precision, QInterval, _quantize


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


@dataclass
class NamedPort:
    name: str
    dir: Dir
    precisions: tuple[Precision, ...]
    rst_to: tuple[float, ...] | None = None
    schedule: ModuloSchedule | None = None
    need_rst: bool = True

    def __post_init__(self):
        if self.rst_to is not None:
            assert len(self.rst_to) == len(self.precisions), 'Reset value length must match precision length'
            self.rst_to = tuple(_quantize(x, *kif) for x, kif in zip(self.rst_to, self.precisions))
        self.precisions = tuple(Precision(*kif) for kif in self.precisions)
        if self.need_rst and self.rst_to is None:
            self.rst_to = tuple(0.0 for _ in self.precisions)
        self.dir = Dir(self.dir)
        if isinstance(self.schedule, list):
            self.schedule = ModuloSchedule(*self.schedule)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NamedPort):
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


class NamedLogic(NamedTuple):
    name: str
    logic: CombLogic

    @classmethod
    def from_list(cls, lst: list) -> 'NamedLogic':
        assert len(lst) == 2, 'Invalid logic list'
        name, logic_dict = lst
        logic = CombLogic.from_dict(logic_dict, raw=True)
        return cls(name, logic)


class AddrMap(NamedTuple):
    """Represent copying data from src[src_interval] to dst[dst_interval]
    If src is a logic, it reads from its output; if dst is a logic, it writes to its input.
    """

    src: str
    src_interval: tuple[int, int]
    dst: str
    dst_interval: tuple[int, int]

    @classmethod
    def from_list(cls, lst: list) -> 'AddrMap':
        assert len(lst) == 4, 'Invalid addr map list'
        src, src_interval, dst, dst_interval = lst
        return cls(src, tuple(src_interval), dst, tuple(dst_interval))


def _check_dir_and_bound(fsm: 'FSM'):
    for addr_map in fsm.addr_maps:
        src_obj = fsm.instances[addr_map.src]
        dst_obj = fsm.instances[addr_map.dst]
        src_int, dst_int = addr_map.src_interval, addr_map.dst_interval
        if isinstance(src_obj, NamedLogic):
            n_src = src_obj.logic.shape[1]
        else:
            n_src = src_obj.size
            assert src_obj.dir in (Dir.IN, Dir.INTERNAL), f'Port {src_obj.name} cannot be read from'

        if isinstance(dst_obj, NamedLogic):
            n_dst = dst_obj.logic.shape[0]
        else:
            n_dst = dst_obj.size
            assert dst_obj.dir in (Dir.OUT, Dir.INTERNAL), f'Port {dst_obj.name} cannot be written to'

        assert 0 <= src_int[0] < src_int[1] <= n_src, f'Invalid src interval {src_int} for {src_obj.name}'
        assert 0 <= dst_int[0] < dst_int[1] <= n_dst, f'Invalid dst interval {dst_int} for {dst_obj.name}'
        assert src_int[1] - src_int[0] == dst_int[1] - dst_int[0], 'Src and dst intervals must have the same size'


def _check_io(fsm: 'FSM'):
    _buf_read_counts = {p.name: np.zeros(p.size, dtype=np.uint64) for p in fsm.ports}
    _buf_write_counts = {p.name: np.zeros(p.size, dtype=np.uint64) for p in fsm.ports}
    _logic_io_read_counts = {l.name: np.zeros(l.logic.shape[1], dtype=np.uint64) for l in fsm.logic}
    _logic_io_write_counts = {l.name: np.zeros(l.logic.shape[0], dtype=np.uint64) for l in fsm.logic}

    read_counts = {**_buf_read_counts, **_logic_io_read_counts}
    write_counts = {**_buf_write_counts, **_logic_io_write_counts}

    for addr_map in fsm.addr_maps:
        src_obj = fsm.instances[addr_map.src]
        dst_obj = fsm.instances[addr_map.dst]
        read_counts[src_obj.name][addr_map.src_interval[0] : addr_map.src_interval[1]] += 1
        write_counts[dst_obj.name][addr_map.dst_interval[0] : addr_map.dst_interval[1]] += 1
        assert isinstance(src_obj, NamedPort) or isinstance(dst_obj, NamedPort), (
            f'{addr_map.src} to {addr_map.dst} must involve at least one port'
        )

    for p in fsm.ports:
        if p.dir in (Dir.OUT, Dir.INTERNAL):
            write_count = write_counts[p.name]
            assert np.all(write_count <= 1), f'Port {p.name} has elements written more than once'
            if p.dir == Dir.INTERNAL:
                read_count = read_counts[p.name]
                assert np.all((read_count > 0) <= (write_count > 0)), (
                    f'Non-inp port {p.name} has elements read without being written'
                )
                if np.any(read_count == 0):
                    warn(f'Port {p.name} has unused elements')

    for logic in fsm.logic:
        read_count = _logic_io_read_counts[logic.name]
        write_count = _logic_io_write_counts[logic.name]
        assert np.all(write_count <= 1), f'Logic {logic.name} has output elements written more than once'
        # ignore collapsed io pins
        write_count[np.sum(logic.logic.inp_kifs, axis=0) == 0] = 1
        read_count[np.sum(logic.logic.out_kifs, axis=0) == 0] = 1
        if not np.all(write_count == 1):
            warn(f'Logic {logic.name} has input elements never written to')
        if not np.all(read_count > 0):
            warn(f'Logic {logic.name} has output elements never used')


class FSM:
    def __init__(
        self,
        logic: tuple[NamedLogic, ...],
        ports: tuple[NamedPort, ...],
        addr_maps: tuple[AddrMap, ...],
    ):
        self.logic = logic
        self.ports = ports
        self.addr_maps = addr_maps
        self._has_emu = False

        _check_dir_and_bound(self)
        _check_io(self)

    def get_logic(self, name: str) -> CombLogic:
        obj = self.instances[name]
        assert isinstance(obj, NamedLogic), f'{name} is not a logic'
        return obj.logic

    def get_port(self, name: str) -> NamedPort:
        obj = self.instances[name]
        assert isinstance(obj, NamedPort), f'{name} is not a port'
        return obj

    @cached_property
    def instances(self) -> dict[str, NamedLogic | NamedPort]:
        instances = dict[str, NamedLogic | NamedPort]()
        for logic in self.logic:
            assert logic.name not in instances, f'Duplicate name {logic.name}'
            instances[logic.name] = logic
        for port in self.ports:
            assert port.name not in instances, f'Duplicate name {port.name}'
            instances[port.name] = port
        return instances

    @cached_property
    def inp_ports(self) -> tuple[NamedPort, ...]:
        return tuple(p for p in self.ports if p.dir == Dir.IN)

    @cached_property
    def out_ports(self) -> tuple[NamedPort, ...]:
        return tuple(p for p in self.ports if p.dir == Dir.OUT)

    @cached_property
    def internal_ports(self) -> tuple[NamedPort, ...]:
        return tuple(p for p in self.ports if p.dir == Dir.INTERNAL)

    @cached_property
    def port_to_logic_map(self) -> tuple[AddrMap, ...]:
        return tuple(
            addr_map
            for addr_map in self.addr_maps
            if isinstance(self.instances[addr_map.src], NamedPort) and isinstance(self.instances[addr_map.dst], NamedLogic)
        )

    @cached_property
    def logic_to_port_map(self) -> tuple[AddrMap, ...]:
        return tuple(
            addr_map
            for addr_map in self.addr_maps
            if isinstance(self.instances[addr_map.dst], NamedPort) and isinstance(self.instances[addr_map.src], NamedLogic)
        )

    @cached_property
    def port_to_port_map(self) -> tuple[AddrMap, ...]:
        return tuple(
            addr_map
            for addr_map in self.addr_maps
            if isinstance(self.instances[addr_map.src], NamedPort) and isinstance(self.instances[addr_map.dst], NamedPort)
        )

    def sinks_to(self, name: str) -> tuple[NamedLogic | NamedPort, ...]:
        assert name in self.instances, f'No instance named {name}'
        return tuple(self.instances[_map.dst] for _map in self.addr_maps if _map.src == name)

    def sources_from(self, name: str) -> tuple[NamedLogic | NamedPort, ...]:
        assert name in self.instances, f'No instance named {name}'
        return tuple(self.instances[_map.src] for _map in self.addr_maps if _map.dst == name)

    @classmethod
    def from_dict(cls, d: dict, raw=False) -> 'FSM':
        if not raw:
            assert d['meta'] == 'ALIRFSM', 'Invalid FSM dict'
            assert d['spec_version'] == ALIR_SPEC_VERSION, 'Unsupported FSM spec version'
            d = d['fsm']

        _logic = tuple(NamedLogic.from_list(l) for l in d['logic'])
        _ports = tuple(NamedPort(**p) for p in d['ports'])
        _addr_maps = tuple(AddrMap.from_list(m) for m in d['addr_maps'])
        return cls(_logic, _ports, _addr_maps)

    def dump_dict(self) -> dict:
        return {
            'meta': 'ALIRFSM',
            'spec_version': ALIR_SPEC_VERSION,
            'fsm': {
                'logic': self.logic,
                'ports': self.ports,
                'addr_maps': self.addr_maps,
            },
        }

    def save(self, path: str | Path, compresslevel: int = 6):
        dump = self.dump_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if str(path).endswith('.gz'):
            with gzip.open(path, 'wt', encoding='utf-8', compresslevel=compresslevel) as f:
                json.dump(dump, f, cls=JSONEncoder, separators=(',', ':'))
        else:
            with open(path, 'w') as f:
                json.dump(dump, f, cls=JSONEncoder, separators=(',', ':'))

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

    def _init_emulator(self):
        if self._has_emu:
            return
        self._emu = FSMEmulator(self)
        self._has_emu = True

    def step(self):
        self._init_emulator()
        self._emu.step()

    def reset(self):
        self._init_emulator()
        self._emu.reset()

    def run(
        self,
        data: dict[str, np.ndarray] | Sequence[np.ndarray] | Sequence[dict[str, np.ndarray]] | np.ndarray,
        steps: int | None = None,
        scheduled: bool = True,
        output_only: bool = True,
    ) -> dict[str, np.ndarray]:
        self._init_emulator()
        return self._emu.run(data, steps, scheduled, output_only)

    def predict(
        self,
        data: dict[str, np.ndarray] | Sequence[np.ndarray] | Sequence[dict[str, np.ndarray]] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        self._init_emulator()
        return self._emu.predict(data)

    @property
    def states(self) -> dict[str, np.ndarray]:
        assert self._has_emu, 'Emulator not initialized yet'
        return self._emu._port_buffers

    def __eq__(self, other: object):
        if not isinstance(other, FSM):
            return False
        return self.logic == other.logic and self.ports == other.ports and self.addr_maps == other.addr_maps


class FSMEmulator:
    def __init__(self, fsm: FSM):
        self.fsm = fsm
        self._init_buffer(False)
        self._t = 0

    def _create_port_buffers(self, exclude_inputs: bool) -> dict[str, np.ndarray]:
        return {p.name: np.zeros(p.size, dtype=np.float64) for p in self.fsm.ports if not exclude_inputs or p.dir != Dir.IN}

    def _init_buffer(self, exclude_inputs: bool):
        self._port_buffers = self._create_port_buffers(exclude_inputs)
        self._logic_io = {
            logic.name: (
                np.empty(logic.logic.shape[0], dtype=np.float64),
                np.empty(logic.logic.shape[1], dtype=np.float64),
            )
            for logic in self.fsm.logic
        }

    def step(self):
        for _map in self.fsm.port_to_logic_map:
            s = slice(*_map.src_interval)
            d = slice(*_map.dst_interval)
            self._logic_io[_map.dst][0][d] = self._port_buffers[_map.src][s]

        for name, (inp_arr, out_arr) in self._logic_io.items():
            comb = self.fsm.get_logic(name)
            if comb.shape[0] == 0:
                out_arr[:] = comb([], quantize=False)
            else:
                out_arr[:] = comb.predict(inp_arr, n_threads=1, ignore_lookup_oob=True)

        new_port_buffers = self._create_port_buffers(True)
        for _map in self.fsm.logic_to_port_map:
            s = slice(*_map.src_interval)
            d = slice(*_map.dst_interval)
            new_port_buffers[_map.dst] = self._logic_io[_map.src][1][s]

        for _map in self.fsm.port_to_port_map:
            s = slice(*_map.src_interval)
            d = slice(*_map.dst_interval)
            new_port_buffers[_map.dst] = self._port_buffers[_map.src][s]

        for k, v in new_port_buffers.items():
            self._port_buffers[k][:] = v

        del new_port_buffers
        self._t += 1

    def _has_enough_data(self, data: dict[str, np.ndarray], t0: int, t1: int, scheduled: bool):
        for port in self.fsm.inp_ports:
            if scheduled:
                assert port.schedule is not None, f'Port {port.name} does not have a schedule'
                n_required = port.schedule.t_to_dense_idx(t1) - port.schedule.t_to_dense_idx(t0)
            else:
                n_required = t1 - t0
            assert len(data[port.name]) >= n_required, f'Not enough data for port {port.name} from t={t0} to t={t1}'
            if len(data[port.name]) != n_required:
                warn(
                    f'Port {port.name} has more data than required from t={t0} to t={t1} ({len(data[port.name])} > {n_required})'
                )

    def reset(self):
        for port in self.fsm.ports:
            if port.rst_to is None or not port.need_rst:
                continue
            self._port_buffers[port.name][:] = port.rst_to
        self._t = 0

    def run(
        self,
        data: Mapping[str, np.ndarray] | Sequence[np.ndarray] | Sequence[Mapping[str, np.ndarray]] | np.ndarray,
        steps: int | None = None,
        scheduled: bool = True,
        output_only: bool = True,
        extra_steps: int = 0,
    ) -> dict[str, np.ndarray]:

        t0 = self._t
        datamap: dict[str, np.ndarray]
        if isinstance(data, np.ndarray):
            assert len(self.fsm.inp_ports) == 1, 'Data array provided for multiple input ports'
            datamap = {self.fsm.inp_ports[0].name: data}
        elif isinstance(data, Sequence) and not isinstance(data, Mapping):
            assert len(data) > 0, 'Data sequence cannot be empty'
            _data = data[0]
            if isinstance(_data, Mapping):
                datamap = {k: np.concatenate([d[k] for d in data]) for k in _data.keys()}
            else:
                assert isinstance(_data, np.ndarray)
                assert len(data) == len(self.fsm.inp_ports)
                datamap = {port.name: data[i] for i, port in enumerate(self.fsm.inp_ports)}  # type: ignore
        else:
            assert isinstance(data, Mapping)
            datamap = {k: np.asarray(v) for k, v in data.items()}

        for port in self.fsm.inp_ports:
            assert port.name in datamap, f'Missing input port {port.name} in data'

        if scheduled:
            for port in self.fsm.inp_ports + self.fsm.out_ports:
                assert port.schedule is not None, f'Port {port.name} does not have a schedule'

        if not steps:
            if scheduled:
                steps = min(port.schedule.dense_idx_to_t(len(datamap[port.name]) - 1) for port in self.fsm.inp_ports) + 1  # type: ignore
            else:
                steps = min(len(datamap[port.name]) for port in self.fsm.inp_ports)

        results = dict[str, np.ndarray]()
        for port in self.fsm.out_ports:
            if scheduled:
                n_outputs = port.schedule.n_valid_samples_between(t0, t0 + steps + extra_steps)  # type: ignore
            else:
                n_outputs = steps + extra_steps
            results[port.name] = np.empty((n_outputs, port.size), dtype=np.float64)
        if not output_only:
            for port in self.fsm.internal_ports:
                results[port.name] = np.empty((steps + extra_steps, port.size), dtype=np.float64)

        for _ in range(steps + extra_steps):
            for port in self.fsm.inp_ports:
                if scheduled:
                    if not port.schedule.check(self._t):  # type: ignore
                        continue
                    idx = port.schedule.n_valid_samples_between(t0, self._t + 1) - 1  # type: ignore
                else:
                    idx = self._t - t0
                if idx < len(datamap[port.name]):
                    self._port_buffers[port.name][:] = datamap[port.name][idx]

            self.step()

            for port in self.fsm.out_ports:
                if scheduled:
                    if not port.schedule.check(self._t):  # type: ignore
                        continue
                    idx = port.schedule.n_valid_samples_between(t0, self._t) - 1  # type: ignore
                else:
                    idx = self._t - t0
                results[port.name][idx] = self._port_buffers[port.name]
            if not output_only:
                for port in self.fsm.internal_ports:
                    results[port.name][self._t] = self._port_buffers[port.name]
        return results

    def predict(
        self,
        data: Mapping[str, np.ndarray] | Sequence[np.ndarray] | Sequence[Mapping[str, np.ndarray]] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        _period = set()
        for port in self.fsm.inp_ports + self.fsm.out_ports:
            assert port.schedule is not None, f'Port {port.name} does not have a schedule'
            _period.add(port.schedule.period)
        assert len(_period) == 1, 'All ports must have the same schedule period'
        extra_steps = max(port.schedule.bias for port in self.fsm.out_ports)  # type: ignore

        self.reset()
        return self.run(data, extra_steps=extra_steps - 1, scheduled=True, output_only=True)
