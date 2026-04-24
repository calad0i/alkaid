import heapq
from collections.abc import Callable

import numpy as np

from ...types import CombLogic
from .cse import is_used_in
from .dce import _index_remap


def _order_ops(comb: CombLogic, sorter: Callable[[CombLogic], dict[int, int]]) -> CombLogic:
    idx_map = sorter(comb)
    remapped_ops = [_index_remap(comb.ops[old_idx], idx_map) for old_idx in idx_map.keys()]
    new_out_idxs = [idx_map[idx] if idx >= 0 else -1 for idx in comb.out_idxs]
    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        new_out_idxs,
        comb.out_shifts,
        comb.out_negs,
        remapped_ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )


def canon_sort_map(comb: CombLogic) -> dict[int, int]:
    """old_idx -> new_idx"""
    order = np.zeros((len(comb.ops), 7), dtype=np.float32)
    for i, op in enumerate(comb.ops):
        order[i, 6] = op.latency
        order[i, 5] = max((order[idx, 5] for idx in op.input_ids), default=-1) + 1
        order[i, 4] = -op.opcode
        order[i, 1:4] = op.qint
        order[i, 0] = op.data >> 32 & 0xFFFFFFFF if op.opcode == 6 else op.data
    return {int(j): int(i) for i, j in enumerate(np.lexsort(order.T))}


def canonical_sort(comb: CombLogic) -> CombLogic:
    "Order ops by topo order/latency"
    return _order_ops(comb, canon_sort_map)


def topo_bandwidth_sort(comb: CombLogic) -> CombLogic:
    "order ops by for topo bandwidth minimization"
    topo_level = []
    for op in comb.ops:
        _level = max((topo_level[i] for i in op.input_ids), default=0) + 1
        topo_level.append(_level)

    used_in = is_used_in(comb)
    last_used_in = [max(i for i in g) for g in used_in.values()]
    max_inp_idx = [max((i for i in op.input_ids), default=-1) for op in comb.ops]
    n_ref = [len(g) for g in used_in.values()]
    killed_at = [0] * len(comb.ops)
    for j in last_used_in:
        if j >= 0:
            killed_at[j] += 1

    keys = [(-killed_at[i], -max_inp_idx[i], n_ref[i], i) for i in range(len(comb.ops))]

    ready: list[tuple[int, int, int, int]] = []
    n_blocking = [len(set(op.input_ids)) for op in comb.ops]
    for i, n in enumerate(n_blocking):
        if n == 0:
            ready.append(keys[i])
    heapq.heapify(ready)

    new_idx: list[int] = []
    while ready:
        _, _, _, i = heapq.heappop(ready)
        for j in used_in[i]:
            if j < 0:  # in out_idx
                continue
            n_blocking[j] -= 1
            if n_blocking[j] == 0:
                heapq.heappush(ready, keys[j])
        new_idx.append(i)
    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_idx)}
    return _order_ops(comb, lambda _: idx_map)
