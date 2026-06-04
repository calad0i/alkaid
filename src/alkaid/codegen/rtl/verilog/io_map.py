from collections.abc import Sequence
from typing import NamedTuple

from ....types import Precision


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


def gen_io_map(
    precs0: Sequence[Precision],
    precs1: Sequence[Precision],
    merge: bool = False,
    bias0: int = 0,
    bias1: int = 0,
) -> tuple[list[BitMap], tuple[int, int]]:
    N = len(precs0)
    assert len(precs1) == N

    bw_map = list[BitMap]()
    idx0, idx1 = bias0, bias1
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
