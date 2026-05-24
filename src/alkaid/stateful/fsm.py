import gzip
import json
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import NamedTuple
from warnings import warn

import numpy as np

from ..types import ALIR_SPEC_VERSION, CombLogic, JSONEncoder, Precision, QInterval


class Dir(Enum):
    IN = 1
    OUT = -1
    INTERNAL = 0


class ModuloSchedule(NamedTuple):
    toggle: tuple[int, ...]
    period: int

    def check(self, i: int) -> bool:
        return np.searchsorted(self.toggle, i % self.period, side='right') % 2 == 1


class NamedPort(NamedTuple):
    name: str
    dir: Dir
    precisions: tuple[Precision, ...]
    rst_to: tuple[float, ...] | None = None
    schedule: ModuloSchedule | None = None
    need_rst: bool = True

    @property
    def size(self) -> int:
        return len(self.precisions)

    @property
    def qint(self) -> tuple[QInterval, ...]:
        return tuple(kif.qint for kif in self.precisions)

    @classmethod
    def from_list(cls, lst: list) -> 'NamedPort':
        assert len(lst) in (5, 6), 'Invalid port list'
        name, dir_str, kifs_list, rst_to_list, sched, *rest = lst
        dir = Dir(dir_str)
        kifs = tuple(Precision(*kif) for kif in kifs_list)
        rst_to = tuple(rst_to_list) if rst_to_list is not None else None
        sched = ModuloSchedule(*sched) if sched is not None else None
        need_rst = bool(rest[0]) if rest else True
        return cls(name, dir, kifs, rst_to, sched, need_rst)


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
                assert np.all((read_count > 0) >= (write_count > 0)), (
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
        _ports = tuple(NamedPort.from_list(p) for p in d['ports'])
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

    def trace(self, data: dict[str, np.ndarray], steps: int) -> dict[str, np.ndarray]:
        _data = {}
        for p in self.inp_ports:
            assert p.name in data, f'Missing input port {p.name} in data'
            arr = data[p.name]
            assert arr.shape[1] == p.size
            if arr.shape[0] >= steps:
                _data[p.name] = arr[:steps]
            else:
                diff = steps - arr.shape[0]
                _min, _max, _ = np.array(p.qint).T
                a, b = _max - _min, _min
                pad = np.random.rand(diff, p.size) * a + b
                _data[p.name] = np.concatenate([arr, pad], axis=0)

        states = {p.name: np.zeros((steps, p.size), dtype=np.float64) for p in self.ports}

        _logic_io = {
            logic.name: (
                np.empty(logic.logic.shape[0], dtype=np.float64),
                np.empty(logic.logic.shape[1], dtype=np.float64),
            )
            for logic in self.logic
        }

        for port in self.ports:
            if port.rst_to is not None:
                states[port.name][0] = port.rst_to

        for t in range(steps):
            for port in self.inp_ports:
                states[port.name][t] = _data[port.name][t]

            for _map in self.port_to_logic_map:
                s = slice(*_map.src_interval)
                d = slice(*_map.dst_interval)
                if self.get_port(_map.src).dir == Dir.IN:
                    _t = t
                else:
                    _t = max(0, t - 1)
                _logic_io[_map.dst][0][d] = states[_map.src][_t, s]

            for name, (inp_arr, out_arr) in _logic_io.items():
                out_arr[:] = self.get_logic(name)(inp_arr)

            for _map in self.logic_to_port_map:
                s = slice(*_map.src_interval)
                d = slice(*_map.dst_interval)
                states[_map.dst][t, d] = _logic_io[_map.src][1][s]

            for _map in self.port_to_port_map:
                s = slice(*_map.src_interval)
                d = slice(*_map.dst_interval)
                if self.get_port(_map.src).dir == Dir.IN:
                    _t = t
                else:
                    _t = max(0, t - 1)
                states[_map.dst][t, d] = states[_map.src][_t, s]

        return states
