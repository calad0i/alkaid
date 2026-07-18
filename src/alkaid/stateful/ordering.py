from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fsm import Conn, Signal


def topo_check_and_sort(conns: Sequence[Conn]) -> list[Conn]:
    """Order combinational connections so that every read happens after its write.

    Each conn drives its ``dst`` view from up to three reads (``src``,
    ``alt_src``, ``enable_if``), plus the index of every dynamically biased view.
    A conn must run after every conn that drives an element it reads, so the
    result is a topological order of the read-after-write dependency graph.  The
    order within each disconnected subgraph is preserved; subgraphs may interleave freely.

    Dependencies are tracked at *interval* granularity rather than per signal, so
    e.g. ``A[0:10] = B`` together with ``B[10:20] = A`` is not a loop -- the bits
    do not overlap.  Combinational-logic ports ``INTERNAL_name_inp`` / ``INTERNAL_name_outp`` are
    contracted into one node: every output bit is assumed to depend on every
    input bit, so reading any output bit waits on writing any input bit.

    Raises:
        ValueError: on a combinational loop, or a double assignment (two conns
            driving overlapping bits of the same signal).
    """
    conns = list(conns)
    n = len(conns)

    def _node(name: str) -> tuple[str, bool]:
        if name.startswith('INTERNAL_'):
            base, sep, port = name.rpartition('_')
            if sep and port in ('in', 'out'):
                return base, True
        return name, False

    # writes grouped by actual signal name -> (start, stop, conn_idx), for the
    # double-assignment check.
    writes_by_signal: dict[str, list[tuple[int, int, int]]] = {}
    # writes / reads grouped by dependency node, for the edge construction.
    dep: dict[str, dict] = {}

    def _slot(key: str, node_level: bool) -> dict:
        d = dep.get(key)
        if d is None:
            d = dep[key] = {'w': [], 'r': [], 'node': node_level}
        return d

    for i, conn in enumerate(conns):
        # Signal views are absolute element intervals.
        ws, we = conn.dst.view
        writes_by_signal.setdefault(conn.dst.name, []).append((ws, we, i))
        dkey, dnode = _node(conn.dst.name)
        _slot(dkey, dnode)['w'].append((ws, we, i))

        src_reads: list[Signal] = [conn.src]
        if conn.alt_src is not None:
            src_reads.append(conn.alt_src)
        if conn.enable_if is not None:
            src_reads.append(conn.enable_if)
        for sig in (conn.src, conn.dst, conn.enable_if, conn.alt_src):
            if sig is not None and sig._dynamic_bias is not None:
                src_reads.append(sig._dynamic_bias[0])
        for sig in src_reads:
            rs, re = sig.view
            rkey, rnode = _node(sig.name)
            _slot(rkey, rnode)['r'].append((rs, re, i))

    # --- no double assignment: write intervals on a signal must be disjoint ---
    for name, ws in writes_by_signal.items():
        ws.sort()
        for (s0, e0, i0), (s1, e1, i1) in zip(ws, ws[1:]):
            if e0 > s1:
                raise ValueError(f'Double assignment on {name}[{max(s0, s1)}:{min(e0, e1)}] by conns #{i0} and #{i1}')

    # --- edges writer -> reader ---
    edges: set[tuple[int, int]] = set()
    for key, slot in dep.items():
        node_writes, node_reads = slot['w'], slot['r']
        if not node_writes or not node_reads:
            continue
        if slot['node']:
            # contracted comb node: every input write feeds every output read
            for _, _, wi in node_writes:
                for _, _, ri in node_reads:
                    if wi == ri:
                        raise ValueError(f'Combinational self-loop at {key} (conn #{wi})')
                    edges.add((wi, ri))
            continue
        # writes are disjoint here (checked above) -> sorting by start also sorts
        # by stop, so the writes overlapping a read form a contiguous range.
        node_writes.sort()
        starts = [w[0] for w in node_writes]
        stops = [w[1] for w in node_writes]
        for ra, rb, ri in node_reads:
            lo = bisect.bisect_right(stops, ra)  # first write with stop > ra
            hi = bisect.bisect_left(starts, rb)  # first write with start >= rb
            for k in range(lo, hi):
                wi = node_writes[k][2]
                if wi == ri:
                    raise ValueError(f'Combinational self-loop on {key} (conn #{wi})')
                edges.add((wi, ri))

    # --- Kahn topological sort, preserving input order among ready conns ---
    adj: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for w, r in edges:
        adj[w].append(r)
        indeg[r] += 1

    ready = deque(i for i in range(n) if indeg[i] == 0)
    order: list[int] = []
    while ready:
        u = ready.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)

    if len(order) != n:
        looped = [i for i in range(n) if indeg[i] > 0]
        raise ValueError(f'Combinational loop among conns {looped}')

    return [conns[i] for i in order]
