from math import ceil, log2

import numpy as np

from ....types import CombLogic
from .lookup import lookup_name
from .ternary import ternary_line


def ssa_gen(sol: CombLogic, neg_repo: dict[int, tuple[int, str]], print_latency: bool = False):
    ops = sol.ops
    kifs = [op.qint.kif for op in ops]
    widths = list(map(sum, kifs))
    inp_widths = sol.inp_kifs.sum(axis=0)
    _inp_widths = np.concatenate([[0], np.cumsum(inp_widths)])
    inp_idxs = np.stack([_inp_widths[1:] - 1, _inp_widths[:-1]], axis=1)

    signals = []
    assigns = []
    ref_count = sol.ref_count

    for i, op in enumerate(ops):
        if ref_count[i] == 0:
            continue

        bw = widths[i]
        if bw == 0:
            continue

        signals.append(f'signal v{i}:std_logic_vector({bw - 1} downto {0});')

        match op.opcode:
            case -2:  # Negation
                a = op.addr[0]
                bw0, v0 = widths[a], f'v{a}'
                is_signed = int(ops[a].qint.min < 0)
                line = f'op_{i}:entity work.negative generic map(BW_IN=>{bw0},BW_OUT=>{bw},IN_SIGNED=>{is_signed}) port map(neg_in=>{v0},neg_out=>v{i});'

            case -1:  # Input marker
                i0, i1 = inp_idxs[op.data[0]]
                line = f'v{i} <= model_inp({i0} downto {i1});'

            case 0 | 1:  # Common binary a+/-b<<shift oprs
                assert len(op.addr) == 2 and len(op.data) == 1
                a, b = op.addr
                data_shift = op.data[0]
                p0, p1 = kifs[a], kifs[b]
                bw0, bw1 = widths[a], widths[b]
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                shift = data_shift + f0 - f1
                dlsbs = max(f0, f1 - data_shift) - kifs[i][2]
                line = f'op_{i}:entity work.shift_adder generic map(BW_INPUT0=>{bw0},BW_INPUT1=>{bw1},SIGNED0=>{s0},SIGNED1=>{s1},BW_OUT=>{bw},DROP_LSBS=>{dlsbs},SHIFT1=>{shift},IS_SUB=>{op.opcode}) port map(in0=>v{a},in1=>v{b},result=>v{i});'

            case 2:  # ReLU
                a = op.addr[0]
                lsb_bias = kifs[a][2] - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias
                v0_name = f'v{a}'
                bw0 = widths[a]
                if ops[a].qint.min < 0:
                    if bw > 1:
                        line = f'v{i} <= {v0_name}({i0} downto {i1}) and ({bw - 1} downto 0 => not {v0_name}({bw0 - 1}));'
                    else:
                        line = f'v{i}(0) <= {v0_name}(0) and (not {v0_name}({bw0 - 1}));'
                else:
                    line = f'v{i} <= {v0_name}({i0} downto {i1});'

            case 3 | -3:  # Explicit quantization
                a = op.addr[0]
                relative_shift = op.data[0]
                lsb_bias = kifs[a][2] - relative_shift - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias
                v0_name = f'v{a}'
                bw0 = widths[a]
                if i0 >= bw0:
                    assert ops[a].qint.min < 0, f'{i}, {a}'

                    if i1 >= bw0:
                        v0_name = f'({i0 - i1} downto 0 => {v0_name}({bw0 - 1}))'
                    else:
                        v0_name = f'({i0 - bw0} downto 0 => {v0_name}({bw0 - 1})) & {v0_name}({bw0 - 1} downto {i1})'
                    line = f'v{i} <= {v0_name};'
                else:
                    line = f'v{i} <= {v0_name}({i0} downto {i1});'

            case 4:  # constant addition
                a = op.addr[0]
                val, f1 = op.data
                bw0, bw1 = widths[a], ceil(log2(abs(val) + 1))
                s0, _, f0 = kifs[a]
                s0, s1 = int(s0), int(val < 0)
                shift = f0 - f1
                v1 = f'{bin(abs(val))[2:]}'
                dlsbs = max(f0, f1) - kifs[i][2]

                line = f'op_{i}:entity work.shift_adder generic map(BW_INPUT0=>{bw0},BW_INPUT1=>{bw1},SIGNED0=>{s0},SIGNED1=>0,BW_OUT=>{bw},DROP_LSBS=>{dlsbs},SHIFT1=>{shift},IS_SUB=>{s1}) port map(in0=>v{a},in1=>"{v1}",result=>v{i});'
            case 5:  # constant
                num = op.data[0]
                if num < 0:
                    num = 2**bw + num
                bin_val = format(num, f'0{bw}b')
                line = f'v{i} <= "{bin_val}";'

            case 6:  # MSB Muxing
                a, b, k = op.addr
                p0, p1 = kifs[a], kifs[b]
                bwk, bw0, bw1 = widths[k], widths[a], widths[b]
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                fo = kifs[i][2]
                shift1 = fo - f1 + op.data[0]
                shift0 = fo - f0
                assert shift0 == 0 or shift1 == 0, f'{i}, {op}, shift0={shift0}, shift1={shift1}'
                shift = shift1 * (shift1 > 0) - shift0 * (shift0 > 0)
                v0, v1 = f'v{a}', f'v{b}'
                if bw0 == 0:
                    v0, bw0 = 'B"0"', 1
                if bw1 == 0:
                    v1, bw1 = 'B"0"', 1
                line = f'op_{i}:entity work.mux generic map(BW_INPUT0=>{bw0},BW_INPUT1=>{bw1},SIGNED0=>{s0},SIGNED1=>{s1},BW_OUT=>{bw},SHIFT1=>{shift}) port map(key=>v{k}({bwk - 1}),in0=>{v0},in1=>{v1},result=>v{i});'

            case 7:  # Multiplication
                a, b = op.addr
                bw0, bw1 = widths[a], widths[b]
                s0, s1 = int(kifs[a][0]), int(kifs[b][0])
                line = f'op_{i}:entity work.multiplier generic map(BW_INPUT0=>{bw0},BW_INPUT1=>{bw1},SIGNED0=>{s0},SIGNED1=>{s1},BW_OUT=>{bw}) port map(in0=>v{a},in1=>v{b},result=>v{i});'

            case 8:  # Lookup Table
                name = lookup_name(sol, op)
                a = op.addr[0]
                line = f'op_{i}:entity work.{name} port map(inp=>v{a},outp=>v{i});'

            case 9:  # Bitwise unary ops
                v0_name = f'v{op.addr[0]}'
                match op.data[0]:
                    case 0:  # NOT
                        line = f'v{i} <= not {v0_name};'
                    case 1:  # ANY
                        line = f'v{i}(0) <= or_reduce({v0_name});'
                    case 2:  # ALL
                        line = f'v{i}(0) <= and_reduce({v0_name});'
                    case _:
                        raise ValueError(f'Unknown unary bitwise op {op.data} for operation {i} ({op})')
            case 10:  # Bitwise Binary
                a, b = op.addr
                data_shift, subop = op.data
                shift = data_shift + kifs[a][2] - kifs[b][2]

                bw0, v0_name = widths[a], f'v{a}'
                s0 = ops[a].qint.min < 0
                bw1, v1_name = widths[b], f'v{b}'
                s1 = ops[b].qint.min < 0

                s0, s1 = int(s0), int(s1)

                line = f'op_{i}:entity work.binop generic map(BW_INPUT0=>{bw0},BW_INPUT1=>{bw1},SIGNED0=>{s0},SIGNED1=>{s1},BW_OUT=>{bw},SHIFT1=>{shift},SUBOP=>{subop}) port map(in0=>{v0_name},in1=>{v1_name},result=>v{i});'
            case 11:
                line = ternary_line(sol, i, kifs, widths)
            case _:
                raise ValueError(f'Unknown opcode {op.opcode} for operation {i} ({op})')

        if print_latency:
            line += f' -- {op.latency}'
        assigns.append(line)
    return signals, assigns


def output_gen(sol: CombLogic, neg_repo: dict[int, tuple[int, str]]):
    assigns = []
    signals = []
    widths = np.atleast_1d(sol.out_kifs.sum(axis=0)).tolist()
    _widths = np.cumsum([0] + widths)
    out_idxs = np.stack([_widths[1:] - 1, _widths[:-1]], axis=1)
    for i, idx in enumerate(sol.out_idxs):
        if idx < 0:
            continue
        i0, i1 = out_idxs[i]
        if i0 == i1 - 1:
            continue
        bw = widths[i]
        assert not sol.out_negs[i]
        assigns.append(f'model_out({i0} downto {i1}) <= v{idx}({bw - 1} downto {0});')
    return signals, assigns


def comb_logic_gen(sol: CombLogic, fn_name: str, print_latency: bool = False, timescale: str | None = None):
    inp_bits = sol.inp_kifs.sum()
    out_bits = sol.out_kifs.sum()

    neg_repo: dict[int, tuple[int, str]] = {}
    ssa_signals, ssa_assigns = ssa_gen(sol, neg_repo=neg_repo, print_latency=print_latency)
    output_signals, output_assigns = output_gen(sol, neg_repo)
    blk = '\n    '

    extra_lib = ''
    if any(op.opcode == 9 and op.data[0] == 1 for op in sol.ops):
        extra_lib += 'use ieee.std_logic_misc.or_reduce;\n'
    if any(op.opcode == 9 and op.data[0] == 2 for op in sol.ops):
        extra_lib += 'use ieee.std_logic_misc.and_reduce;\n'

    code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
{extra_lib}

entity {fn_name} is port(
    model_inp:in std_logic_vector({inp_bits - 1} downto {0});
    model_out:out std_logic_vector({out_bits - 1} downto {0})
);
end entity {fn_name};

architecture rtl of {fn_name} is
    {blk.join(ssa_signals + output_signals)}


begin
    {blk.join(ssa_assigns + output_assigns)}

end architecture rtl;

"""
    return code
