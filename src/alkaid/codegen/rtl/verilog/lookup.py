from hashlib import sha256
from math import ceil
from uuid import UUID

import numpy as np

from ....types import CombLogic, Op


def table_hex(sol: CombLogic, op: Op) -> list[str]:
    assert op.opcode == 8
    assert sol.lookup_tables is not None
    width = sum(op.qint.kif)
    ndigits = ceil(width / 4)
    table = sol.lookup_tables[op.data[0]]
    data = table.padded_table(sol.ops[op.addr[0]].qint)
    lines = []
    for v in data:
        if np.isnan(v):
            line = 'X' * ndigits
        else:
            line = f'{hex(int(v) & ((1 << width) - 1))[2:].upper().zfill(ndigits)}'
        lines.append(line)
    return lines


def lookup_name(sol: CombLogic, op: Op) -> str:
    bw_in = sum(sol.ops[op.addr[0]].qint.kif)
    bw_out = sum(op.qint.kif)
    text = f'{bw_in}\n{bw_out}\n' + '\n'.join(table_hex(sol, op))
    uuid = UUID(int=int(sha256(text.encode('utf-8')).hexdigest()[:32], 16), version=4)
    token = str(uuid).replace('-', '_')
    return f'table_{token}'


def lookup_ops(sol: CombLogic) -> list[Op]:
    return [op for op in sol.ops if op.opcode == 8 and sum(op.qint.kif) > 0]


def _lookup_def(sol: CombLogic, op: Op, name) -> str:
    bw_in = sum(sol.ops[op.addr[0]].qint.kif)
    bw_out = sum(op.qint.kif)
    addr_digits = max(1, ceil(bw_in / 4))

    body = [f'module {name}(input [{bw_in - 1}:0] in, output reg [{bw_out - 1}:0] out);', 'always @(*) case (in)']
    for addr, value in enumerate(table_hex(sol, op)):
        body.append(f"{bw_in}'h{addr:0{addr_digits}X}: out = {bw_out}'h{value};")
    body += [f"default: out = {{{bw_out}{{1'bx}}}};", 'endcase', 'endmodule']
    return '\n'.join(body)


def lookup_source(combs: list[CombLogic], timescale: str | None = None) -> str:
    defs = {}
    for comb in combs:
        for op in lookup_ops(comb):
            name = lookup_name(comb, op)
            if name not in defs:
                defs[name] = _lookup_def(comb, op, name)
    if not defs:
        return ''
    code = '\n\n'.join(defs.values())
    code = f'/* verilator lint_off DECLFILENAME */\n{code}\n/* verilator lint_on DECLFILENAME */'
    if timescale is not None:
        return f'{timescale}\n\n{code}\n'
    return f'{code}\n'
