from collections.abc import Sequence
from math import ceil

import numpy as np

from ..stateful.fsm import FSM, Conn, ModuloSchedule, Signal, _comb_io_signals
from ..types import CombLogic, Op
from .passes import canonical_sort, dead_code_elimin


def _index_remap(op: Op, idx_map: dict[int, int]) -> Op:
    if op.opcode == -1:
        return op
    addr = tuple(idx_map.get(i, i) for i in op.addr)
    return Op(addr, op.opcode, op.data, op.qint, op.latency, op.cost)


def _split_comb_logic(
    comb: CombLogic,
    n_stages: int | None = None,
    latency_cutoff: float | None = None,
    verbose=True,
) -> tuple[CombLogic, ...]:
    assert (n_stages is not None) + (latency_cutoff is not None) == 1, (
        'Exactly one of n_stages and latency_cutoff must be specified.'
    )

    comb = canonical_sort(comb)
    latencies = [op.latency for op in comb.ops]
    _latency = latencies[-1]
    n_stages = n_stages if n_stages is not None else max(ceil(_latency / latency_cutoff), 1)  # type: ignore
    assert n_stages > 0, 'Number of stages must be greater than 0.'
    lat_cutoffs = np.linspace(0, _latency, n_stages + 1)[1:]
    split_idxs = [0] + np.searchsorted(latencies, lat_cutoffs, side='right').tolist()
    if verbose:
        print(f'Latency cutoffs for splitting: {lat_cutoffs.tolist()}')

    staged_ops = [comb.ops[i:j] for i, j in zip(split_idxs[:-1], split_idxs[1:])]

    ext_req_idx: list[list[int]] = []
    'all idxs required by this or later stages from earlier stages'
    last_req = set(comb.out_idxs) - {-1}
    for ii in range(len(staged_ops) - 1, -1, -1):
        _req = {i for op in staged_ops[ii] for i in op.input_ids}
        _all_req = last_req.union(_req)
        _ext_req = {i for i in _all_req if i < split_idxs[ii]}
        last_req = _ext_req
        ext_req_idx.append(sorted(_ext_req))
    ext_req_idx = ext_req_idx[::-1] + [comb.out_idxs]

    index_maps: list[dict[int, int]] = []
    staged_ops_remap: list[list[Op]] = []
    for ii, _ops in enumerate(staged_ops):
        index_map: dict[int, int] = {}
        'global idx -> local idx'
        ops: list[Op] = []
        inp_idx = 0
        for i, gi in enumerate(ext_req_idx[ii]):
            _op = comb.ops[gi]
            index_map[gi] = i
            if _op.opcode != 5:
                ops.append(Op((), -1, (inp_idx,), _op.qint, _op.latency, 0))
                inp_idx += 1
            else:
                ops.append(_op)  # const copy
        index_maps.append(index_map)

        global_base_idx = split_idxs[ii]
        local_base_idx = len(ext_req_idx[ii])
        for i, op in enumerate(_ops):
            ops.append(_index_remap(op, index_map))
            index_map[global_base_idx + i] = local_base_idx + i
        staged_ops_remap.append(ops)

    ext_req_idx = [[i for i in req if i < 0 or comb.ops[i].opcode != 5] for req in ext_req_idx]
    # const vars copied to individual stages

    combs: list[CombLogic] = []
    for ii, ops in enumerate(staged_ops_remap):
        if ii == 0:
            inp_shifts = comb.inp_shifts
            n_in = comb.shape[0]
        else:
            n_in = len(ext_req_idx[ii])
            inp_shifts = [0] * n_in
        index_map = index_maps[ii]
        if ii == n_stages - 1:
            out_idxs = [index_map[i] for i in comb.out_idxs]
            n_out = len(out_idxs)
            out_shifts = comb.out_shifts
            out_negs = comb.out_negs
        else:
            out_idxs = [index_map[i] for i in ext_req_idx[ii + 1]]
            n_out = len(out_idxs)
            out_shifts = [0] * n_out
            out_negs = [False] * n_out

        _comb = CombLogic(
            shape=(n_in, n_out),
            inp_shifts=inp_shifts,
            out_idxs=out_idxs,
            out_shifts=out_shifts,
            out_negs=out_negs,
            ops=ops,
            carry_size=comb.carry_size,
            adder_size=comb.adder_size,
            lookup_tables=comb.lookup_tables,
        )
        combs.append(dead_code_elimin(_comb))

    return tuple(combs)


def _stages_to_fsm(stages: Sequence[CombLogic], reg_inp=True, reg_out=True) -> FSM:
    lat = len(stages)
    assert lat > 0, 'FSM stage tuple must not be empty'

    inp_precisions = tuple(qint.kif for qint in stages[0].inp_qint)
    out_precisions = tuple(qint.kif for qint in stages[-1].out_qint)
    io_period = 0 if lat == 1 and not reg_inp and not reg_out else 1

    conns: list[Conn] = []
    inp_sig = Signal(
        'model_inp',
        True,
        inp_precisions,
        reg=False,
        mode='r',
        schedule=ModuloSchedule((0,), io_period),
    )
    out_sig = Signal(
        'model_out',
        True,
        out_precisions,
        reg=reg_out,
        mode='w',
        schedule=ModuloSchedule((lat - 1 + reg_out + reg_inp,), io_period),
    )

    if reg_inp:
        inp_sig_reg = Signal(
            'model_inp_reg',
            False,
            inp_precisions,
            reg=True,
            mode='rw',
        )
        conns.append(Conn(inp_sig, inp_sig_reg))
        inp_sig = inp_sig_reg

    ports: list[Signal] = [inp_sig]
    for i in range(1, lat):
        kifs = tuple(qint.kif for qint in stages[i].inp_qint)
        ports.append(Signal(f'stage{i}_inp', False, kifs, reg=True, mode=''))
    ports.append(out_sig)

    logic: dict[str, CombLogic] = {}
    for i, comb in enumerate(stages):
        n_in, n_out = comb.shape
        if n_in == 0 and n_out == 0:
            continue
        name = f'logic{i}'
        logic[name] = comb
        sig_in, sig_out = _comb_io_signals(name, comb)
        if n_in > 0:
            conns.append(Conn(ports[i], sig_in))
        if n_out > 0:
            conns.append(Conn(sig_out, ports[i + 1]))

    return FSM(logic, tuple(conns))


def to_pipeline(
    comb: CombLogic,
    n_stages: int | None = None,
    latency_cutoff: float | None = None,
    verbose=True,
    reg_inp: bool = True,
    reg_out: bool = True,
) -> FSM:
    """Split a `CombLogic` program into latency-balanced pipeline stages.

    Exactly one of `n_stages` and `latency_cutoff` must be specified. The
    resulting tuple is an internal staging detail; public callers should use
    `to_pipeline`, which lowers the stages directly to an `FSM`.

    Parameters
    ----------
    comb : CombLogic
        The combinational logic to be pipelined into multiple stages.
    n_stages : int | None
        Number of stages to create.
    latency_cutoff : float | None
        Maximum target latency per stage. The final stage count is derived
        from the total operation latency.
    verbose : bool
        Whether to print the latency cutoffs used for splitting.
    reg_inp : bool
        Whether to register the FSM inputs.
    reg_out : bool
        Whether to register the FSM outputs.

    Returns
    -------
    FSM
        An FSM representing the pipelined version of the input combinational logic.
    """
    stages = _split_comb_logic(comb, n_stages=n_stages, latency_cutoff=latency_cutoff, verbose=verbose)
    return _stages_to_fsm(stages, reg_inp=reg_inp, reg_out=reg_out)
