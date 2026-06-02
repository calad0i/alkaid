from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from ....types import CombLogic, Pipeline, Precision, QInterval


class BitMap(NamedTuple):
    src: tuple[int, int]
    dst: tuple[int, int]

    def can_merge(self, other: 'BitMap'):
        return self.src[1] == other.src[0] and self.dst[1] == other.dst[0]

    def merge(self, other: 'BitMap') -> 'BitMap':
        return BitMap((self.src[0], other.src[1]), (self.dst[0], other.dst[1]))

    @property
    def _type(self):
        """0: const 0

        1: 1 to N copy (sign ext)

        2: regular copy
        """
        w0, w1 = self.src[1] - self.src[0], self.dst[1] - self.dst[0]
        if w0 == 0:
            return 0
        if w0 == w1:
            return 2
        assert w0 == 1
        return 1


def gen_io_map(precs0: Sequence[Precision], precs1: Sequence[Precision], merge: bool = False):
    N = len(precs0)
    assert len(precs1) == N

    bw_map = list[BitMap]()
    idx0, idx1 = 0, 0
    for p0, p1 in zip(precs0, precs1):
        int0, frac0 = sum(p0[:2]), p0[2]
        int1, frac1 = sum(p1[:2]), p1[2]

        n_rightpad = frac1 - frac0
        if n_rightpad > 0:
            bw_map.append(BitMap((-1, -1), (idx1, idx1 + n_rightpad)))
            idx1 += n_rightpad
        else:
            idx0 -= n_rightpad

        n_copy = min(int0, int1) + min(frac0, frac1)
        if n_copy > 0:
            bw_map.append(BitMap((idx0, idx0 + n_copy), (idx1, idx1 + n_copy)))
            idx0 += n_copy
            idx1 += n_copy

        n_leftpad = int1 - int0
        if n_leftpad > 0:
            if p0.signed:
                _map = BitMap((idx0 - 1, idx0), (idx1, idx1 + n_leftpad))
            else:
                _map = BitMap((-1, -1), (idx1, idx1 + n_leftpad))
            idx1 += n_leftpad
            bw_map.append(_map)
        else:
            idx0 -= n_leftpad

    _reg_copy = [b for b in bw_map if b._type == 2]
    _bcast_copy = [b for b in bw_map if b._type == 1]
    _const_zero = [b for b in bw_map if b._type == 0]

    if merge:
        for i in range(len(_reg_copy) - 1, 0, -1):
            left, right = _reg_copy[i - 1], _reg_copy[i]
            if left.can_merge(right):
                _reg_copy[i - 1] = left.merge(right)
                _reg_copy.pop(i)
        for i in range(len(_const_zero) - 1, 0, -1):
            left, right = _const_zero[i - 1], _const_zero[i]
            if left.can_merge(right):
                _const_zero[i - 1] = left.merge(right)
                _const_zero.pop(i)

    bw_map = sorted(_reg_copy + _bcast_copy + _const_zero, key=lambda b: b.dst[0])
    dsts = [b.dst for b in bw_map]
    assert all(dsts[i][1] == dsts[i + 1][0] for i in range(len(dsts) - 1))
    return bw_map, (idx0, idx1)


def gen_io_map_sugar(qints: Sequence[QInterval], direction: str, merge: bool):
    assert direction in ('inp', 'out')
    precs = [qint.kif for qint in qints]
    _kif = np.max(precs, axis=0)
    precs_uniform = [Precision(bool(_kif[0]), int(_kif[1]), int(_kif[2]))] * len(precs)
    precs0, precs1 = (precs_uniform, precs) if direction == 'inp' else (precs, precs_uniform)
    return gen_io_map(precs0, precs1, merge=merge)


def gen_assignments(map_out: list[BitMap], name_inp: str, name_out: str):
    out_assignment = []
    for (ii, ji), (io, jo) in map_out:
        if ji - ii == jo - io:
            out_assignment.append(f'assign {name_out}[{jo - 1}:{io}] = {name_inp}[{ji - 1}:{ii}];')
        elif ji - ii == 1:
            out_assignment.append(f'assign {name_out}[{jo - 1}:{io}] = {{{jo - io}{{{name_inp}[{ii}]}}}};')
        else:
            assert ii == ji == -1, f'Unexpected map_out entry: {(ii, ji), (io, jo)}'
            out_assignment.append(f"assign {name_out}[{jo - 1}:{io}] = {jo - io}'b0;")
    out_assignment_str = '\n    '.join(out_assignment)
    return out_assignment_str


def generate_io_wrapper(sol: CombLogic | Pipeline, module_name: str, pipelined: bool = False):

    map_inp, (w_uniform_inp, w_inp) = gen_io_map_sugar(sol.inp_qint, direction='inp', merge=True)
    map_out, (w_out, w_uniform_out) = gen_io_map_sugar(sol.out_qint, direction='out', merge=True)

    inp_assignment_str = gen_assignments(map_inp, 'model_inp', 'packed_inp')
    out_assignment_str = gen_assignments(map_out, 'packed_out', 'model_out')

    clk_and_rst_inp, clk_and_rst_bind = '', ''
    if pipelined:
        clk_and_rst_inp = '\n   input clk,'
        clk_and_rst_bind = '\n        .clk(clk),'

    return f"""`timescale 1 ns / 1 ps

module {module_name}_wrapper ({clk_and_rst_inp}
    // verilator lint_off UNUSEDSIGNAL
    input [{w_uniform_inp - 1}:0] model_inp,
    // verilator lint_on UNUSEDSIGNAL
    output [{w_uniform_out - 1}:0] model_out
);
    wire [{w_inp - 1}:0] packed_inp;
    wire [{w_out - 1}:0] packed_out;

    {inp_assignment_str}

    {module_name} op ({clk_and_rst_bind}
        .model_inp(packed_inp),
        .model_out(packed_out)
    );

    {out_assignment_str}

endmodule
"""
