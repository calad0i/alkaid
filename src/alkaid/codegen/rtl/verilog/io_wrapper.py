from collections.abc import Sequence

import numpy as np

from ....types import CombLogic, Pipeline, Precision, QInterval
from .io_map import BitMap, gen_io_map


def gen_io_map_sugar(qints: Sequence[QInterval], direction: str, merge: bool):
    assert direction in ('inp', 'out')
    precs = [qint.kif for qint in qints]
    _kif = np.max(precs, axis=0)
    precs_uniform = [Precision(bool(_kif[0]), int(_kif[1]), int(_kif[2]))] * len(precs)
    precs0, precs1 = (precs_uniform, precs) if direction == 'inp' else (precs, precs_uniform)
    return gen_io_map(precs0, precs1, merge=merge)


def gen_assignments(map_out: list[BitMap], name_inp: str, name_out: str):
    out_assignment = []
    for (ii, ji), (io, jo) in map_out:
        if ji - ii == jo - io:
            out_assignment.append(f'assign {name_out}[{jo - 1}:{io}] = {name_inp}[{ji - 1}:{ii}];')
        elif ji - ii == 1:
            out_assignment.append(f'assign {name_out}[{jo - 1}:{io}] = {{{jo - io}{{{name_inp}[{ii}]}}}};')
        else:
            assert ii == ji == -1, f'Unexpected map_out entry: {(ii, ji), (io, jo)}'
            out_assignment.append(f"assign {name_out}[{jo - 1}:{io}] = {jo - io}'b0;")
    out_assignment_str = '\n    '.join(out_assignment)
    return out_assignment_str


def generate_io_wrapper(sol: CombLogic | Pipeline, module_name: str, pipelined: bool = False):

    map_inp, (w_uniform_inp, w_inp) = gen_io_map_sugar(sol.inp_qint, direction='inp', merge=True)
    map_out, (w_out, w_uniform_out) = gen_io_map_sugar(sol.out_qint, direction='out', merge=True)

    inp_assignment_str = gen_assignments(map_inp, 'model_inp', 'packed_inp')
    out_assignment_str = gen_assignments(map_out, 'packed_out', 'model_out')

    clk_and_rst_inp, clk_and_rst_bind = '', ''
    if pipelined:
        clk_and_rst_inp = '\n   input clk,'
        clk_and_rst_bind = '\n        .clk(clk),'

    return f"""`timescale 1 ns / 1 ps

module {module_name}_wrapper ({clk_and_rst_inp}
    // verilator lint_off UNUSEDSIGNAL
    input [{w_uniform_inp - 1}:0] model_inp,
    // verilator lint_on UNUSEDSIGNAL
    output [{w_uniform_out - 1}:0] model_out
);
    wire [{w_inp - 1}:0] packed_inp;
    wire [{w_out - 1}:0] packed_out;

    {inp_assignment_str}

    {module_name} op ({clk_and_rst_bind}
        .model_inp(packed_inp),
        .model_out(packed_out)
    );

    {out_assignment_str}

endmodule
"""
