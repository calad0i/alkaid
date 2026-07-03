from ..._binary import get_lsb_loc
from ...types import CombLogic, Op, QInterval


def canonicalize(comb: CombLogic) -> CombLogic:
    comb = const_propagation(comb)
    comb = canonicalize_msb_wraps(comb)
    comb = canonicalize_outputs(comb)
    return comb


def canonicalize_msb_wraps(comb: CombLogic) -> CombLogic:
    """Normalize wraps that already describe the source MSB bit."""

    ops = comb.ops.copy()
    uses: dict[int, list[int]] = {i: [] for i in range(len(ops))}
    for i, op in enumerate(ops):
        for idx in op.addr:
            uses[idx].append(i)
    for idx in comb.out_idxs:
        if idx >= 0:
            uses[idx].append(-1)

    for i, op in enumerate(ops):
        if op.opcode != 3 or op.data[0] == 0:
            continue
        if op.qint.min != 0 or op.qint.max != op.qint.step:
            continue
        src_k, src_i, _ = ops[op.addr[0]].qint.kif
        step = 2.0 ** (src_i + int(src_k) - 1)
        if op.qint.step == step:
            ops[i] = Op(op.addr, op.opcode, (0,), op.qint, op.latency, op.cost)
            continue
    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        comb.out_idxs,
        comb.out_shifts,
        comb.out_negs,
        ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )


def const_propagation(comb: CombLogic) -> CombLogic:
    ops = comb.ops.copy()
    for i, op in enumerate(ops):
        if op.opcode in (-1, 5):
            continue
        if all(ops[j].opcode == 5 for j in op.input_ids):
            # constant propagation
            fake_buf = {j: ops[j].data[0] * ops[j].qint.step for j in op.input_ids}
            val: float = float(comb.exec_op(op, fake_buf, None))  # type: ignore
            step = 2.0 ** get_lsb_loc(val)
            ops[i] = Op((), 5, (int(val / step),), QInterval(val, val, step), op.latency, op.cost)
    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        comb.out_idxs,
        comb.out_shifts,
        comb.out_negs,
        ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )


def canonicalize_outputs(comb: CombLogic) -> CombLogic:
    ops = comb.ops.copy()
    out_idxs = comb.out_idxs.copy()
    out_shifts = comb.out_shifts.copy()
    out_negs = comb.out_negs.copy()
    for i in range(comb.shape[1]):
        idx, s, n = comb.out_idxs[i], comb.out_shifts[i], comb.out_negs[i]
        if idx < 0:
            out_idxs[i] = len(ops)
            ops.append(Op((), 5, (0,), QInterval(0.0, 0.0, 2.0**127), 0, 0))
            out_shifts[i] = 0
            out_negs[i] = False
            continue

        op = comb.ops[idx]
        if ops[idx].opcode == 5:
            if s or n:
                val = op.data[0] * op.qint.step * 2**s * (-1 if n else 1)
                out_idxs[i] = len(ops)
                step = 2.0 ** get_lsb_loc(val)
                op = Op((), 5, (int(val / step),), QInterval(val, val, step), 0, 0)
                ops.append(op)
                out_shifts[i] = 0
                out_negs[i] = False
            continue

        if n:
            out_idxs[i] = len(ops)
            qint = ops[idx].qint
            ops.append(Op((idx,), -2, (), QInterval(-qint.max, -qint.min, qint.step), op.latency, 0))
            out_negs[i] = False

    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        out_idxs,
        out_shifts,
        out_negs,
        ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )
