import heapq
import typing
from collections.abc import Callable, Sequence
from math import prod
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

if typing.TYPE_CHECKING:
    from ..fixed_variable import FVariable
    from ..fixed_variable_array import FVArray


T = typing.TypeVar('T', 'FVariable', float, np.floating, '_ArgPair')
TA = TypeVar('TA', 'FVArray', NDArray[np.integer | np.floating])


class Packet:
    def __init__(self, v):
        self.value = v

    def __gt__(self, other: 'Packet') -> bool:  # type: ignore
        from ..fixed_variable import FVariable

        a, b = self.value, other.value

        if isinstance(a, FVariable):
            if isinstance(b, FVariable):
                la, lb = round(a.latency), round(b.latency)
                if la != lb:
                    return la > lb
                if b._factor > 0 and a._factor < 0:
                    return False
                if b._factor < 0 and a._factor > 0:
                    return True
                return sum(a.kif[:2]) > sum(b.kif[:2])
            return True

        return False

    def __lt__(self, other: 'Packet') -> bool:  # type: ignore
        return not self.__gt__(other)


def _reduce(operator: Callable[[T, T], T], arr: Sequence[T]) -> T:
    from ..fixed_variable import FVariable

    if isinstance(arr, np.ndarray):
        arr = arr.ravel().tolist()
    assert len(arr) > 0, 'Array must not be empty'
    if len(arr) == 1:
        return arr[0]
    dtype = arr[0].__class__
    if not issubclass(dtype, FVariable):
        r = operator(arr[0], arr[1])
        for i in range(2, len(arr)):
            r = operator(r, arr[i])
        return r

    heap = [Packet(v) for v in arr]  # type: ignore
    heapq.heapify(heap)
    while len(heap) > 1:
        v1 = heapq.heappop(heap).value
        v2 = heapq.heappop(heap).value
        v = operator(v1, v2)
        heapq.heappush(heap, Packet(v))  # type: ignore
    return heap[0].value


def _prepare_reduce(arr: np.ndarray, axis: int | Sequence[int] | None, keepdims: bool):
    """Normalize axes, transpose, and reshape for reduction."""
    all_axis = tuple(range(arr.ndim))
    axis = axis if axis is not None else all_axis
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    axis = tuple(a if a >= 0 else a + arr.ndim for a in axis)

    xpose_axis = sorted(all_axis, key=lambda a: (a in axis) * 1000 + a)
    if keepdims:
        target_shape = tuple(d if ax not in axis else 1 for ax, d in enumerate(arr.shape))
    else:
        target_shape = tuple(d for ax, d in enumerate(arr.shape) if ax not in axis)

    dim_contract = prod(arr.shape[a] for a in axis)
    flat = np.transpose(arr, xpose_axis).reshape(-1, dim_contract)
    return flat, target_shape, dim_contract


def reduce(operator: Callable[[T, T], T], arr: TA, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> TA:
    """
    Reduce the array by the operator over the specified axis.
    """
    from ..fixed_variable_array import FVArray

    flat, target_shape, _ = _prepare_reduce(arr, axis, keepdims)
    r = np.array([_reduce(operator, flat[i]) for i in range(flat.shape[0])]).reshape(target_shape)

    if isinstance(arr, FVArray):
        r = FVArray(r, arr.solver_options, hwconf=arr.hwconf)
    return r if r.shape != () or keepdims else r.item()  # type: ignore


class _ArgPair:
    """Wraps (value, index) for argmin/argmax reduction via ``reduce``."""

    __slots__ = ('val', 'idx')

    def __init__(self, val, idx):
        self.val = val
        self.idx = idx


def argreduce(arr: 'FVArray', axis: int | Sequence[int] | None = None, keepdims: bool = False, minimize=True):
    """Reduction returning the index of the min or max element."""
    from ..fixed_variable import FVariable
    from ..fixed_variable_array import FVArray

    flat, target_shape, _ = _prepare_reduce(arr, axis, keepdims)
    _flat = np.empty(flat.shape, dtype=object)
    for i in range(flat.shape[0]):
        for j in range(flat.shape[1]):
            _flat[i, j] = _ArgPair(flat[i, j], FVariable.from_const(float(j), hwconf=arr.hwconf))

    def op(a: _ArgPair, b: _ArgPair) -> _ArgPair:
        cmp = (a.val <= b.val) if minimize else (a.val >= b.val)
        return _ArgPair(cmp.msb_mux(a.val, b.val), cmp.msb_mux(a.idx, b.idx))

    r = np.array([_reduce(op, _flat[i]) for i in range(_flat.shape[0])]).reshape(target_shape)
    r = np.vectorize(lambda p: p.idx)(r)
    if isinstance(arr, FVArray):
        r = FVArray(r, arr.solver_options, hwconf=arr.hwconf)
    return r if r.shape != () or keepdims else r.item()  # type: ignore
