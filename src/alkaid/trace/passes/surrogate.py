from math import ceil

import numpy as np

from ..._binary import iceil_log2, overlap_counts
from ...trace import HWConfig
from ...trace.fixed_variable import LookupTable
from ...types import CombLogic, Op, QInterval, _iter_sum_terms
from .cse import is_used_in


def _cadd_consumer_width(idx: int, ops: list[Op], used_in: dict[int, set[int]]) -> set[int]:
    ret: set[int] = set()
    for cidx in used_in[idx]:
        if cidx < 0:
            ret.add(-1)  # output
        else:
            if ops[cidx].opcode not in (-2, 3, 9):
                ret.add(ops[cidx].opcode)
            else:
                ret.update(_cadd_consumer_width(cidx, ops, used_in))
    return ret


def _is_const_descendent(idx: int, ops: list[Op], cache: dict[int, float]) -> float:
    if idx in cache:
        return cache[idx]
    op = ops[idx]
    if op.opcode == 5:  # CONST
        cache[idx] = 1.0
        return 1.0
    if op.opcode == 10:  # bin bitops
        return max(_is_const_descendent(op.addr[0], ops, cache), _is_const_descendent(op.addr[1], ops, cache))
    if op.opcode in (-2, 3, 9):  # NEG, wrap, unary bitops
        res = _is_const_descendent(op.addr[0], ops, cache)
        cache[idx] = res
        return res
    if op.opcode == 6:  # MUX
        a, b = _is_const_descendent(op.addr[0], ops, cache), _is_const_descendent(op.addr[1], ops, cache)
        res = 0.5 * (a + b)
        cache[idx] = res
        return res
    cache[idx] = 0.0
    return 0.0


def _bit_high(qint: QInterval) -> int:
    signed, integers, _fractional = qint.kif
    return int(signed) + integers - 1


def _bit_low(qint: QInterval) -> int:
    return -qint.kif.fractional


def cost_lat_add(qint0: QInterval, qint1: QInterval, shift1: int, n_add: int, n_accum: int, is_sub: bool = False):
    left, overlap, right = overlap_counts(qint0, qint1, shift1)
    if overlap <= 0:  # bit concat
        return 0, 0

    if is_sub:
        cost = max(left + overlap + right - 1, 1)
    else:
        cost = left + overlap
    # bw_add = left + overlap
    # cost = (max(bw_add - 1, 1) + n_add - 1) // n_add
    lat = (cost - 1 + n_accum - 1) // n_accum * 0.025390625 + 1.09375
    return cost, lat


def cost_lat_mul(qint0: QInterval, qint1: QInterval, n_add: int, n_accum: int):
    _min0, _max0 = min(qint0.min, 0), max(qint0.max, 0)
    _min1, _max1 = min(qint1.min, 0), max(qint1.max, 0)
    b0, b1 = iceil_log2((_max0 - _min0) / qint0.step), iceil_log2((_max1 - _min1) / qint1.step)
    cost1 = b0 * (b1 + n_add - 1) // n_add
    cost2 = b1 * (b0 + n_add - 1) // n_add
    cost = min(cost1, cost2)
    lat1 = b0 * ((b1 - 1 + n_accum - 1) // n_accum) * 0.025390625 + 1.09375
    lat2 = b1 * ((b0 - 1 + n_accum - 1) // n_accum) * 0.025390625 + 1.09375
    lat = min(lat1, lat2)
    return cost * 0.75, lat


def _count_luts_rec(bit_nd: np.ndarray, LUT_X: int = 6) -> float:
    """Count LUT6s for one output bit. Greedy: picks axis with most identical halves."""
    d = bit_nd.ndim
    if d <= LUT_X:
        return int(np.unique(bit_nd).size > 1)

    flat_size = 1 << (d - 1)
    halves = np.stack([np.moveaxis(bit_nd, ax, 0).reshape(2, flat_size) for ax in range(d)])  # (d, 2, flat_size)
    matches = np.sum(halves[:, 0] == halves[:, 1], axis=1)
    best_ax = int(np.argmax(matches))

    if matches[best_ax] == flat_size:
        left = np.take(bit_nd, 0, axis=best_ax)
        return _count_luts_rec(left, LUT_X)

    left = np.take(bit_nd, 0, axis=best_ax)
    right = np.take(bit_nd, 1, axis=best_ax)
    return _count_luts_rec(left, LUT_X) + _count_luts_rec(right, LUT_X)


def _count_luts(bit_nd: np.ndarray, LUT_X: int = 6) -> float:
    """Count LUT6s needed for one output bit.

    Tries all axes at the top level (exhaustive), greedy below.
    """
    d = bit_nd.ndim
    if d <= LUT_X:
        return 0.0 if len(np.unique(bit_nd)) == 1 else 1.0

    best_cost = float('inf')
    for ax in range(d):
        left = np.take(bit_nd, 0, axis=ax)
        right = np.take(bit_nd, 1, axis=ax)
        if np.array_equal(left, right):
            c = _count_luts_rec(left, LUT_X)
        else:
            c = _count_luts_rec(left, LUT_X) + _count_luts_rec(right, LUT_X)
        best_cost = min(best_cost, c)
    return best_cost


def cost_lat_lut(qint_in: QInterval, table: LookupTable, LUT_X: int, LUT_Y: int, skip_cost: bool = False):

    bw_in = sum(qint_in.kif)
    large_lut = bw_in - LUT_X > 6
    if large_lut:
        lat = max(bw_in - LUT_X, 1) * 0.5
    else:
        lut_extra_bits = max(bw_in - LUT_X, 0)
        # Base + F7/8/9 + fabric
        lat = 0.075 + 0.2 * min(lut_extra_bits, 3) + 0.8 * max(lut_extra_bits - 3, 0)

    if skip_cost:
        return 0, lat

    if large_lut:
        return 0.5 * 2.0 ** (bw_in - LUT_X), lat

    out_bw = sum(table.spec.out_kif)
    data = table.padded_table(qint_in)
    int_data = np.nan_to_num(data, nan=0).astype(np.int64)

    total_cost = 0.0
    for b in range(out_bw):
        bit_vals = (int_data >> b) & 1

        if np.all(bit_vals == bit_vals[0]):
            continue

        bit_nd = bit_vals.reshape((2,) * bw_in)
        total_cost += _count_luts(bit_nd, LUT_X)
    total_cost *= max(1.0 + (0.5625 - 0.109375 * out_bw), 1)

    return total_cost, lat


def cost_lat_mux(qint0: QInterval, qint1: QInterval, shift1: int, LUT_X: int, LUT_Y: int):
    return sum(overlap_counts(qint0, qint1, shift1)) * 2.0 ** (LUT_Y - LUT_X), 0.5


def cost_relu(qint_out: QInterval, LUT_X: int = 6, LUT_Y: int = 5):
    return sum(qint_out.kif) * 0.5, 0


def cost_lat_bin_bitops(qint0: QInterval, qint1: QInterval, shift1: int, LUT_X: int, LUT_Y: int):
    x, y, z = overlap_counts(qint0, qint1, shift1)
    if y <= 0:
        return 0, 0
    cost = 3 * y / LUT_Y * 2 ** (LUT_Y - LUT_X)
    lat = 0.5
    return cost, lat


def cost_cadd(qint_out: QInterval, value: int, fractional: int):
    value = abs(value)
    if value == 0:
        return 0
    trailing_zeros = (value & -value).bit_length() - 1
    const_low = trailing_zeros - fractional
    carry_start = max(_bit_low(qint_out), const_low)
    carry_span = max(_bit_high(qint_out) - carry_start + 1, 0)
    return max(carry_span - 1, 0) * 0.15


def _ternary_add_suffix_width(idx: int, ops: list[Op]) -> int:
    op = ops[idx]
    raw_terms = tuple((addr, 1 if plus else -1, shift) for addr, plus, shift in _iter_sum_terms(op))
    kifs = [ops[addr].qint.kif for addr, _, _ in raw_terms]
    term_fracs = [kif.fractional - shift for kif, (_, _, shift) in zip(kifs, raw_terms)]
    align_f = max(term_fracs)
    drop_lsbs = align_f - op.qint.kif.fractional
    out_width = sum(op.qint.kif)
    out_high = drop_lsbs + out_width - 1
    pads = [align_f - term_frac for term_frac in term_fracs]
    negates = [sign < 0 for _, sign, _ in raw_terms]

    min_pad = min(pads)
    low_count = sum(pad == min_pad for pad in pads)
    low_input = pads.index(min_pad)
    no_next_pad = out_high + 2
    next_pads = [pad if pad > min_pad else no_next_pad for pad in pads]
    upper_start = min(next_pads)
    low_copy_lo = max(drop_lsbs, min_pad)
    low_copy_hi = min(out_high, upper_start - 1)

    lowcopy = (
        low_count == 1
        and not negates[low_input]
        and upper_start != no_next_pad
        and upper_start > drop_lsbs
        and low_copy_hi >= low_copy_lo
    )
    if not lowcopy:
        return out_width
    return max(out_high - upper_start + 1, 0)


def cost_lat_ternary_add(idx: int, ops: list[Op], n_accum: int):
    suffix_width = _ternary_add_suffix_width(idx, ops)
    if suffix_width <= 0:
        return 0, 0
    cost = suffix_width + ceil(suffix_width / 8)
    lat = ((suffix_width - 1 + n_accum - 1) // n_accum * 0.03125 + 1.0) * 1.3125
    return cost, lat


def cost_lat_op(
    idx: int,
    ops: list[Op],
    hwconf: HWConfig,
    lut: tuple[LookupTable, ...] | None,
    used_in: dict[int, set[int]],
    _skip_op8_cost: bool = False,
) -> tuple[float, float]:
    LUT_X, LUT_Y = 6, 5
    n_add, n_carry = hwconf.adder_size % 65535, hwconf.carry_size % 65535
    op = ops[idx]
    _cache: dict[int, float] = {}
    match op.opcode:
        case -2:  # neg
            c, l = 0, 0
        case -1:  # READ
            c, l = 0, 0
        case 0 | 1:  # +/-
            op0, op1 = ops[op.addr[0]], ops[op.addr[1]]
            qint0, qint1 = op0.qint, op1.qint
            shift1 = op.data[0]
            c, l = cost_lat_add(qint0, qint1, shift1, n_add, n_carry, op.opcode == 1)
        case 2:  # relu(-)
            qint_in = ops[op.addr[0]].qint
            if qint_in.min >= 0:
                return 0, 0  # no-op for non-negative
            c, l = cost_relu(op.qint, LUT_X, LUT_Y)
        case 3:  # WRAP
            return 0, 0
        case 4:  # cadd: absorbed if consumer is not add/sub/mux
            eff_consumers = _cadd_consumer_width(idx, ops, used_in)
            if any(cop in (0, 1, 6) for cop in eff_consumers):
                return cost_cadd(op.qint, op.data[0], op.data[1]), 0
            return 0, 0
        case 5:  # const
            return 0, 0
        case 6:  # msb_mux
            out_bw = sum(op.qint.kif)
            sf = _is_const_descendent(idx, ops, _cache)
            return out_bw * (0.5 - 0.36 * sf), 0.5
        case 7:  # mul
            qint0, qint1 = ops[op.addr[0]].qint, ops[op.addr[1]].qint
            c, l = cost_lat_mul(qint0, qint1, n_add, n_carry)
        case 8:  # lut
            qint_in = ops[op.addr[0]].qint
            # qint_out = op.qint
            assert lut is not None
            c, l = cost_lat_lut(qint_in, lut[op.data[0]], LUT_X, LUT_Y, skip_cost=_skip_op8_cost)
            if _skip_op8_cost:
                c = op.cost
        case 9:  # unary bitops: absorbed
            c, l = 0, 0
        case 10:  # bin bitops
            qint0, qint1 = ops[op.addr[0]].qint, ops[op.addr[1]].qint
            shift = op.data[0]
            c, l = cost_lat_bin_bitops(qint0, qint1, shift, LUT_X, LUT_Y)
        case 11:  # accum
            c, l = cost_lat_ternary_add(idx, ops, n_carry)
        case _:
            raise NotImplementedError(f'Unsupported opcode: {op.opcode}')
    return c, l


def _with_cost_lat(op: Op, cost, lat) -> Op:
    return Op(op.addr, op.opcode, op.data, op.qint, lat, cost)


def add_surrogate(comb: CombLogic, _skip_op8_cost=False) -> CombLogic:
    "Add surrogate cost and latency"
    new_ops = []
    ops = comb.ops
    hwconf = HWConfig(comb.adder_size, comb.carry_size, -1)
    used_in = is_used_in(comb)
    for idx, op in enumerate(comb.ops):
        cost, lat = cost_lat_op(idx, ops, hwconf, comb.lookup_tables, used_in, _skip_op8_cost=_skip_op8_cost)
        lat = lat + max(tuple(new_ops[j].latency for j in op.input_ids) + (0,))
        new_ops.append(_with_cost_lat(op, cost, lat))
    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        comb.out_idxs,
        comb.out_shifts,
        comb.out_negs,
        new_ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )
