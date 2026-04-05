"""Histogram and searchsorted operations for FVariable arrays."""

import typing
from typing import Literal

import numpy as np
from numpy.typing import NDArray

if typing.TYPE_CHECKING:
    from ..fixed_variable_array import FVArray

_range = range


def _searchsorted_thermometer(a, v, side):
    """Thermometer-encoded searchsorted: sum of per-edge comparisons.

    All N comparisons happen in parallel, then a tree-sum reduces them.
    Better for short sequences (small N) where the parallel comparisons
    fit in a single pipeline stage.

    Hardware cost per element: N comparators + (N-1) adders.
    """
    if side == 'right':
        return np.sum(v[..., None] >= a, axis=-1)  # type: ignore
    else:
        return np.sum(v[..., None] > a, axis=-1)  # type: ignore


def _searchsorted_bsearch(a, v, side):
    """Binary-search searchsorted with bunch of muxs.

    Uses ceil(log2(N+1)) sequential stages.  Each stage resolves one bit
    of the result by selecting a pre-computed comparison via a msb_mux tree.

    Hardware cost per element: N comparators + ~N msb_mux (selection tree).
    """
    from math import ceil, log2

    from ..fixed_variable import FVariable
    from ..fixed_variable_array import FVArray

    N = len(np.asarray(a).ravel())
    k = max(1, ceil(log2(N + 1)))

    flat = np.asarray(v).ravel()
    hwconf = a.hwconf if isinstance(a, FVArray) else v.hwconf
    solver_options = a.solver_options if isinstance(a, FVArray) else v.solver_options

    _max = np.array([float(x.high) if isinstance(x, FVariable) else x for x in flat]).max()
    _max = FVariable.from_const(float(_max + 1), hwconf=hwconf)

    a_flat = np.asarray(a).ravel()
    n_pad = 2**k - 1 - N
    padded = np.concatenate([a_flat, np.full(n_pad, _max)])
    results = np.empty((len(flat), k), dtype=object)

    def _bin_index(addr: list[FVariable], values: list[FVariable]):
        """Select one value from *values* using *selectors* as a binary address (MSB-first)."""
        assert len(values) == 1 << len(addr)
        if len(values) == 1:
            return values[0]
        mid = len(values) // 2
        lo = _bin_index(addr[1:], values[:mid])
        hi = _bin_index(addr[1:], values[mid:])
        return addr[0].msb_mux(hi, lo)

    for j, x in enumerate(flat):
        # Pre-compute all N_full comparisons.
        if side == 'left':
            all_cmp = [x > e for e in padded]
        else:
            all_cmp = [x >= e for e in padded]

        # Binary search: resolve one bit per stage, selecting which
        # pre-computed comparison to use via a msb_mux tree.
        bits = []
        for stage in range(k):
            SN = k - 1 - stage
            n_entries = 1 << stage
            cmp_indices = [(i << (SN + 1)) + (1 << SN) - 1 for i in range(n_entries)]
            stage_cmps = [all_cmp[idx] for idx in cmp_indices]

            cmp = stage_cmps[0] if stage == 0 else _bin_index(bits, stage_cmps)
            bits.append(cmp)

        for _k in range(k):
            results[j, _k] = bits[_k] * 2.0 ** (k - 1 - _k)

    return FVArray(results, solver_options, hwconf=hwconf).sum(axis=-1)


def searchsorted(
    a: 'FVArray | NDArray',
    v: 'FVArray | NDArray',
    side: str = 'left',
    sorter: 'str | NDArray | None' = None,
) -> 'FVArray | NDArray':
    """Synthesisable searchsorted.  Either *a* or *v* (or both) may be a
    FVArray.

    Parameters
    ----------
    a : FVArray or ndarray
        Sorted 1-D array of edges.  May be symbolic (FVArray)
        or compile-time constants (plain ndarray).
    v : FVArray or ndarray
        Values to search for.
    side : {'left', 'right'}
        ``'left'``: first index where ``a[i-1] < v <= a[i]``.
        ``'right'``: first index where ``a[i-1] <= v < a[i]``.
    sorter : str or None
        Implementation strategy.

        * ``'thermometer'`` — N parallel comparators + tree-sum.  Works for
          both constant and symbolic edges.  Good for short edge arrays
          where all comparisons fit in one pipeline stage.
        * ``'bsearch'`` — binary search with msb_mux trees.
          Uses ceil(log2(N+1)) sequential stages; each stage maps to a
          pipeline stage.  Works with both constant and symbolic edges.
        * ``None`` (default) — ``'thermometer'`` when ``len(a) <= 8`` or
          *a* is a FVArray, ``'bsearch'`` otherwise.
    """
    from ..fixed_variable import FVariable
    from ..fixed_variable_array import FVArray

    if not isinstance(v, FVArray) and not isinstance(a, FVArray):
        return np.searchsorted(a, v, side=side)  # type: ignore

    N = len(np.asarray(a).ravel())
    fva: FVArray = a if isinstance(a, FVArray) else v  # type: ignore
    if N == 0:
        z = FVariable.from_const(0.0, hwconf=fva.hwconf)
        return FVArray(np.full(np.shape(v), z), fva.solver_options, hwconf=fva.hwconf)

    if sorter is None:
        sorter = 'thermometer' if N <= 8 else 'bsearch'

    if sorter == 'thermometer':
        return _searchsorted_thermometer(a, v, side)
    elif sorter == 'bsearch':
        return _searchsorted_bsearch(a, v, side)
    else:
        raise ValueError(f"sorter must be 'thermometer', 'bsearch', or None, got {sorter!r}")


def histogram(
    a: 'FVArray | NDArray',
    bins: 'int | NDArray' = 10,
    range: 'tuple[float, float] | None' = None,
    weights: 'FVArray | NDArray | None' = None,
    density: Literal[False] = False,
) -> 'tuple[FVArray | NDArray, NDArray]':
    """Compute histogram using thermometer code counting.

    Bin edges must be compile-time constants. Only internal edges are compared;
    out-of-range elements naturally land in the first/last bins.

    Parameters
    ----------
    a : FVArray or ndarray
        Input data. Flattened before processing.
    bins : int or 1-D array-like
        If int: number of equal-width bins (``range`` required).
        If array: monotonically increasing bin edges.
    range : (float, float) or None
        Required when ``bins`` is an int.
    weights : FVArray or ndarray or None
        Per-element weights (same shape as ``a``). When given, each element
        contributes its weight instead of 1 to the bin count.
    density : bool
        Not supported, raises ValueError if True.

    Returns
    -------
    counts : FVArray or ndarray
    bin_edges : ndarray
    """
    from ..fixed_variable_array import FVArray

    assert not density, 'density=True is not supported'

    if not isinstance(a, FVArray):
        return np.histogram(a, bins=bins, range=range, weights=weights)

    if isinstance(bins, (int, np.integer)):
        if range is None:
            raise ValueError('range=(lo, hi) required when bins is an int')
        lo, hi = float(range[0]), float(range[1])
        if lo >= hi:
            raise ValueError(f'range must satisfy lo < hi, got ({lo}, {hi})')
        edges = np.linspace(lo, hi, int(bins) + 1)
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError('bins must be 1-D with at least 2 edges')
        if not np.all(np.diff(edges) > 0):
            raise ValueError('bin edges must be strictly increasing')

    flat = a.ravel()
    M = len(flat)

    if weights is not None:
        _weights = np.asarray(weights).ravel()
        if _weights.size != M:
            raise ValueError(f'weights must have same length as a, got {len(weights)} vs {M}')
    else:
        _weights = 1

    if M == 0:
        return FVArray(np.zeros(len(edges) - 1), a.solver_options, hwconf=a.hwconf), edges

    # Thermometer encoding

    _cum1 = np.sum((flat[None, :] >= edges[:-1, None]) * _weights, axis=-1)  # type: ignore
    _cum2 = np.sum((flat[None, :] > edges[-1:, None]) * _weights, axis=-1)  # type: ignore
    cum = np.concatenate([_cum1, _cum2], axis=0)

    counts = cum[:-1] - cum[1:]

    return counts, edges  # type: ignore
