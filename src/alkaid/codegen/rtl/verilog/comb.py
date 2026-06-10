from hashlib import sha256
from math import ceil, log2
from uuid import UUID

import numpy as np

from ....types import CombLogic, Op
from .._ternary_codegen import verilog_ternary_line


def gen_memfile(sol: CombLogic, op: Op) -> str:
    assert op.opcode == 8
    assert sol.lookup_tables is not None
    table = sol.lookup_tables[op.data[0]]
    width = sum(table.spec.out_kif)
    ndigits = ceil(width / 4)
    data = table.padded_table(sol.ops[op.addr[0]].qint)
    mem_lines = []
    for v in data:
        if np.isnan(v):
            line = 'X' * ndigits
        else:
            line = f'{hex(int(v) & ((1 << width) - 1))[2:].upper().zfill(ndigits)}'
        mem_lines.append(line)
    return '\n'.join(mem_lines)


def get_table_name_memfile(sol: CombLogic, op: Op) -> tuple[str, str]:
    memfile = gen_memfile(sol, op)
    hash_obj = sha256(memfile.encode('utf-8'))
    _int = int(hash_obj.hexdigest()[:32], 16)
    uuid = UUID(int=_int, version=4)
    return f'table_{str(uuid)}.mem', memfile


def ssa_gen(sol: CombLogic, neg_repo: dict[int, tuple[int, str]], print_latency: bool = False) -> list[str]:
    ops = sol.ops
    kifs = [op.qint.kif for op in ops]
    widths = list(map(sum, kifs))
    inp_widths = sol.inp_kifs.sum(axis=0)
    _inp_widths = np.concatenate([[0], np.cumsum(inp_widths)])
    inp_idxs = np.stack([_inp_widths[1:] - 1, _inp_widths[:-1]], axis=1)

    lines: list[str] = []
    ref_count = sol.ref_count

    for i, op in enumerate(ops):
        if ref_count[i] == 0:
            continue

        bw = widths[i]
        v = f'v{i}[{bw - 1}:0]'
        _def = f'wire [{bw - 1}:0] v{i};'
        if bw == 0:
            continue

        match op.opcode:
            case -2:  # Negation
                a = op.addr[0]
                bw0, v0 = widths[a], f'v{a}'
                is_signed = int(ops[a].qint.min < 0)
                line = f'{_def} negative #({bw0}, {bw}, {is_signed}) op_{i} ({v0}, {v});'

            case -1:  # Input marker
                i0, i1 = inp_idxs[op.data[0]]
                line = f'{_def} assign {v} = model_inp[{i0}:{i1}];'

            case 0 | 1:  # Common binary a+/-b<<shift oprs
                assert len(op.addr) == 2 and len(op.data) == 1
                a, b = op.addr
                data_shift = op.data[0]
                p0, p1 = kifs[a], kifs[b]  # precision -> keep_neg, integers (no sign), fractional

                bw0, bw1 = widths[a], widths[b]  # width
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                shift = data_shift + f0 - f1
                v0, v1 = f'v{a}[{bw0 - 1}:0]', f'v{b}[{bw1 - 1}:0]'
                dlsbs = max(f0, f1 - data_shift) - kifs[i][2]

                line = f'{_def} shift_adder #({bw0},{bw1},{s0},{s1},{bw},{dlsbs},{shift},{op.opcode}) op_{i} ({v0},{v1},{v});'

            case 2:  # ReLU
                a = op.addr[0]
                lsb_bias = kifs[a][2] - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias

                v0_name = f'v{a}'
                bw0 = widths[a]

                if ops[a].qint.min < 0:
                    line = f'{_def} assign {v} = {v0_name}[{i0}:{i1}] & {{{bw}{{~{v0_name}[{bw0 - 1}]}}}};'
                else:
                    line = f'{_def} assign {v} = {v0_name}[{i0}:{i1}];'

            case 3:  # Explicit quantization
                a = op.addr[0]
                lsb_bias = kifs[a][2] - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias
                v0_name = f'v{a}'
                bw0 = widths[a]

                if i0 >= bw0:
                    assert ops[a].qint.min < 0, f'{i}, {a}'

                    if i1 >= bw0:
                        v0_name = f'{{{i0 - i1 + 1}{{{v0_name}[{bw0 - 1}]}}}}'
                    else:
                        v0_name = f'{{{{{i0 - bw0 + 1}{{{v0_name}[{bw0 - 1}]}}}}, {v0_name}[{bw0 - 1}:{i1}]}}'
                    line = f'{_def} assign {v} = {v0_name};'
                else:
                    line = f'{_def} assign {v} = {v0_name}[{i0}:{i1}];'

            case 4:  # constant addition
                a = op.addr[0]
                val, f1 = op.data
                bw0, bw1 = widths[a], ceil(log2(abs(val) + 1))
                s0, _, f0 = kifs[a]
                s0, s1 = int(s0), int(val < 0)
                shift = f0 - f1
                v0 = f'v{a}[{bw0 - 1}:0]'
                v1 = f"{bw1}'{bin(abs(val))[1:]}"
                dlsbs = max(f0, f1) - kifs[i][2]

                line = f'{_def} shift_adder #({bw0},{bw1},{s0},0,{bw},{dlsbs},{shift},{s1}) op_{i} ({v0},{v1},{v});'

            case 5:  # constant
                num = op.data[0]
                if num < 0:
                    num = 2**bw + num
                line = f"{_def} assign {v} = '{bin(num)[1:]};"

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
                vk, v0, v1 = f'v{k}[{bwk - 1}]', f'v{a}[{bw0 - 1}:0]', f'v{b}[{bw1 - 1}:0]'
                if bw0 == 0:
                    v0, bw0 = "1'b0", 1
                if bw1 == 0:
                    v1, bw1 = "1'b0", 1

                line = f'{_def} mux #({bw0},{bw1},{s0},{s1},{bw},{shift}) op_{i} ({vk},{v0},{v1},{v});'

            case 7:  # Multiplication
                a, b = op.addr
                bw0, bw1 = widths[a], widths[b]  # width
                s0, s1 = int(kifs[a][0]), int(kifs[b][0])
                v0, v1 = f'v{a}[{bw0 - 1}:0]', f'v{b}[{bw1 - 1}:0]'

                line = f'{_def} multiplier #({bw0},{bw1},{s0},{s1},{bw}) op_{i} ({v0},{v1},{v});'

            case 8:  # Lookup Table
                name = get_table_name_memfile(sol, op)[0]
                a = op.addr[0]
                bw0 = widths[a]

                line = f'{_def} lookup_table #({bw0},{bw},"{name}") op_{i} (v{a}, {v});'

            case 9:  # Bitwise Unary
                v0_name = f'v{op.addr[0]}'
                match op.data[0]:
                    case 0:  # NOT
                        line = f'{_def} assign {v} = ~{v0_name};'
                    case 1:  # OR with self (reduction)
                        line = f'{_def} assign {v} = |{v0_name};'
                    case 2:  # AND with self (reduction)
                        line = f'{_def} assign {v} = &{v0_name};'
                    case _:
                        raise ValueError(f'Unknown bitwise operation {op.data} for operation {i} ({op})')

            case 10:  # Bitwise Binary
                a, b = op.addr
                data_shift, subop = op.data
                shift = data_shift + kifs[a][2] - kifs[b][2]

                bw0, v0_name = widths[a], f'v{a}'
                s0 = ops[a].qint.min < 0

                bw1, v1_name = widths[b], f'v{b}'
                s1 = ops[b].qint.min < 0

                s0, s1 = int(s0), int(s1)

                line = f'{_def} binop #({bw0},{bw1},{s0},{s1},{bw},{shift},{subop}) op_{i} ({v0_name}, {v1_name}, {v});'

            case 11:
                line = verilog_ternary_line(sol, i, _def)

            case _:
                raise ValueError(f'Unknown opcode {op.opcode} for operation {i} ({op})')

        if print_latency:
            line += f' // {op.latency}'
        lines.append(line)
    return lines


def output_gen(sol: CombLogic, neg_repo: dict[int, tuple[int, str]]) -> list[str]:
    lines = []
    widths = sol.out_kifs.sum(axis=0).tolist()

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
        lines.append(f'assign model_out[{i0}:{i1}] = v{idx}[{bw - 1}:0];')
    return lines


def comb_logic_gen(sol: CombLogic, fn_name: str, print_latency: bool = False, timescale: str | None = None):
    inp_bits = sol.inp_kifs.sum()
    out_bits = sol.out_kifs.sum()

    fn_signature = [
        f'module {fn_name} (',
        f'    input [{inp_bits - 1}:0] model_inp,',
        f'    output [{out_bits - 1}:0] model_out',
        ');',
    ]

    neg_repo: dict[int, tuple[int, str]] = {}
    ssa_lines = ssa_gen(sol, neg_repo=neg_repo, print_latency=print_latency)
    output_lines = output_gen(sol, neg_repo)

    indent = '    '
    base_indent = '\n'
    body_indent = base_indent + indent
    code = f"""{base_indent[1:]}{base_indent.join(fn_signature)}

    // verilator lint_off UNUSEDSIGNAL
    // Explicit quantization operation will drop bits if exists

    {body_indent.join(ssa_lines)}

    // verilator lint_on UNUSEDSIGNAL

    {body_indent.join(output_lines)}

endmodule
"""
    if timescale is not None:
        code = f'{timescale}\n\n{code}'
    return code


def table_mem_gen(sol: CombLogic) -> dict[str, str]:
    if not sol.lookup_tables:
        return {}
    mem_files = {}
    for op in sol.ops:
        if not op.opcode == 8:
            continue
        name, memfile = get_table_name_memfile(sol, op)
        mem_files[name] = memfile
    return mem_files
