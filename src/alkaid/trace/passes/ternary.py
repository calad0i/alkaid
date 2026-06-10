from ...types import CombLogic, Op


# (#fused adders, root width, widest child input, total child input width, -child idx)
# 大就是好(x
class _Score(tuple[int, int, int, int, int]):
    def __add__(self, other: '_Score'):  # type: ignore
        return _Score(x + y for x, y in zip(self, other))

    def __sub__(self, other: '_Score'):
        return _Score(x - y for x, y in zip(self, other))


def _width(op: Op) -> int:
    signed, integers, fractional = op.qint.kif
    return int(signed) + int(integers) + int(fractional)


def _edge_score(ops: list[Op], _root_idx: int, child_idx: int) -> _Score:
    child_inp_ws = [_width(ops[idx]) for idx in ops[child_idx].addr]
    ret = (1, _width(ops[child_idx]), max(child_inp_ws), sum(child_inp_ws), -child_idx)
    return _Score(ret)


def _fusion_leafs(comb: CombLogic) -> dict[int, int]:
    ops = comb.ops
    ref_count = comb.ref_count
    leafs: dict[int, list[int]] = {}
    root_of: dict[int, int] = {}
    candidate_nodes: set[int] = set()

    for root_idx, root in enumerate(ops):
        if root.opcode not in (0, 1) or ref_count[root_idx] == 0:
            continue

        for leaf_idx in root.addr:
            if ref_count[leaf_idx] != 1 or ops[leaf_idx].opcode not in (0, 1):
                continue
            leafs.setdefault(root_idx, []).append(leaf_idx)
            leafs.setdefault(leaf_idx, [])
            root_of[leaf_idx] = root_idx
            candidate_nodes.add(root_idx)
            candidate_nodes.add(leaf_idx)

    if not leafs:
        return {}

    zero = _Score((0, 0, 0, 0, 0))
    free: dict[int, _Score] = {}
    blocked: dict[int, _Score] = {}

    # root_idx -> leaf_idx
    chosen_leaf: dict[int, int] = {}

    for idx in sorted(candidate_nodes):
        base = zero
        for leaf_idx in leafs.get(idx, []):
            base = base + free[leaf_idx]

        blocked[idx] = base
        best = base
        best_child = None
        for leaf_idx in leafs.get(idx, []):
            score = base - free[leaf_idx] + _edge_score(ops, idx, leaf_idx) + blocked[leaf_idx]
            if score > best:
                best = score
                best_child = leaf_idx
        free[idx] = best
        if best_child is not None:
            chosen_leaf[idx] = best_child

    # root_idx -> leaf_idx
    pairs: dict[int, int] = {}

    def collect(idx: int, root_matched: bool) -> None:
        leaf_idx = None if root_matched else chosen_leaf.get(idx, None)
        if leaf_idx is not None:
            pairs[idx] = leaf_idx
            collect(leaf_idx, True)
        for other_child_idx in leafs.get(idx, []):
            if other_child_idx != leaf_idx:
                collect(other_child_idx, False)

    for root_idx in sorted(candidate_nodes - set(root_of)):
        collect(root_idx, False)

    return dict(sorted(pairs.items()))


def _binary_terms(op: Op) -> list[tuple[int, int, int]]:
    rhs_sign = -1 if op.opcode == 1 else 1
    return [(op.addr[0], 1, 0), (op.addr[1], rhs_sign, op.data[0])]


def _to_ternary_terms(ops: list[Op], root_idx: int, child_idx: int) -> list[tuple[int, int, int]]:
    root = ops[root_idx]
    rhs_sign = -1 if root.opcode == 1 else 1
    if child_idx == root.addr[0]:
        return _binary_terms(ops[child_idx]) + [(root.addr[1], rhs_sign, root.data[0])]
    else:
        _a = [(idx, sign * rhs_sign, shift + root.data[0]) for idx, sign, shift in _binary_terms(ops[child_idx])]
        _b = [(root.addr[0], 1, 0)]
        return _a + _b


def fuse_ternary_adders(comb: CombLogic) -> CombLogic:
    """Fuse single-use binary add/sub trees into opcode 11 ternary adders."""

    fusion_leafs = _fusion_leafs(comb)
    if not fusion_leafs:
        return comb

    ops = comb.ops.copy()
    for root_idx, child_idx in fusion_leafs.items():
        root = ops[root_idx]
        terms = _to_ternary_terms(comb.ops, root_idx, child_idx)
        addr = tuple(idx for idx, _, _ in terms)
        data = tuple(value for _, sign, shift in terms for value in (1 if sign > 0 else 0, shift))
        ops[root_idx] = Op(addr, 11, data, root.qint, root.latency, root.cost)

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
