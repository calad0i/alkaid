from collections.abc import Sequence

from ....types import CombLogic, Precision
from ..verilog.ternary import ternary_layout


def ternary_line(
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
