from collections.abc import Sequence
from dataclasses import dataclass

from ...types import CombLogic, Precision, _iter_sum_terms


@dataclass(frozen=True)
class TernaryTerm:
    addr: int
    width: int
    signed: int
    negate: int
    pad: int


@dataclass(frozen=True)
class TernaryLayout:
    terms: tuple[TernaryTerm, TernaryTerm, TernaryTerm]
    out_width: int
    drop_lsbs: int


def ternary_layout(
    sol: CombLogic,
    op_idx: int,
    kifs: Sequence[Precision],
    widths: Sequence[int],
) -> TernaryLayout:
    ops = sol.ops
    op = ops[op_idx]
    raw_terms = tuple((addr, 1 if plus else -1, shift) for addr, plus, shift in _iter_sum_terms(op))
    assert len(raw_terms) == 3

    term_fracs = [kifs[idx].fractional - shift for idx, _, shift in raw_terms]
    align_f = max(term_fracs)
    drop_lsbs = align_f - kifs[op_idx].fractional
    assert drop_lsbs >= 0

    terms = tuple(
        TernaryTerm(
            addr=idx,
            width=widths[idx],
            signed=int(kifs[idx].signed),
            negate=int(sign < 0),
            pad=align_f - term_frac,
        )
        for (idx, sign, _), term_frac in zip(raw_terms, term_fracs, strict=True)
    )
    assert len(terms) == 3
    return TernaryLayout(terms, widths[op_idx], drop_lsbs)


def _generic_values(layout: TernaryLayout) -> list[int]:
    values = []
    for term in layout.terms:
        values.extend([term.width, term.signed, term.negate, term.pad])
    return values


def verilog_ternary_line(
    sol: CombLogic,
    op_idx: int,
    out_def: str,
    kifs: Sequence[Precision],
    widths: Sequence[int],
) -> str:
    layout = ternary_layout(sol, op_idx, kifs, widths)
    params = ','.join(map(str, [*_generic_values(layout), layout.out_width, layout.drop_lsbs]))
    ports = [f'v{term.addr}[{term.width - 1}:0]' for term in layout.terms]
    ports.append(f'v{op_idx}[{layout.out_width - 1}:0]')
    return f'{out_def} ternary_adder #({params}) op_{op_idx} ({",".join(ports)});'


def vhdl_ternary_line(
    sol: CombLogic,
    op_idx: int,
    kifs: Sequence[Precision],
    widths: Sequence[int],
) -> str:
    layout = ternary_layout(sol, op_idx, kifs, widths)
    generics = []
    for pos, term in enumerate(layout.terms):
        generics.extend(
            [
                f'BW_INPUT{pos}=>{term.width}',
                f'SIGNED{pos}=>{term.signed}',
                f'NEGATE{pos}=>{term.negate}',
                f'PAD{pos}=>{term.pad}',
            ]
        )
    generics.extend([f'BW_OUT=>{layout.out_width}', f'DROP_LSBS=>{layout.drop_lsbs}'])

    ports = [f'in{pos}=>v{term.addr}' for pos, term in enumerate(layout.terms)]
    ports.append(f'result=>v{op_idx}')
    return f'op_{op_idx}:entity work.ternary_adder generic map({",".join(generics)}) port map({",".join(ports)});'
