from collections.abc import Sequence

import numpy as np

from ....stateful import FSM, Conn, Signal
from ....types import Precision
from .comb import comb_logic_gen
from .io_map import BitMap, gen_io_map


def _rst_bin(sig: Signal) -> str:
    assert sig.rst_to is not None
    s = 0
    number = 0
    for val, p in zip(sig.rst_to, sig.precisions):
        w = sum(p)
        n = round(float(val) * (2.0**p.fractional)) & ((1 << w) - 1)
        number = n << s | number
        s += w
    return f"{sig.width}'h{hex(number)[2:].upper()}"


def gen_assignments(
    map_out: list[BitMap],
    name_inp: str,
    name_out: str,
    clocked: bool,
    inp_offset: str | None = None,
    out_offset: str | None = None,
):
    def to_assign_str(name_inp: str, name_out: str):
        if not clocked:
            return f'assign {name_out} = {name_inp};'
        else:
            return f'{name_out} <= {name_inp};'

    def _name(name: str, lo: int, hi: int, offset: str | None) -> str:
        if offset is None:
            return f'{name}[{hi - 1}:{lo}]'
        start = f'{lo} + {offset}' if lo else offset
        return f'{name}[{start} +: {hi - lo}]'

    assignments = []
    for (ii, ji), (io, jo) in map_out:
        _name_out = _name(name_out, io, jo, out_offset)
        if ji - ii == jo - io:
            _name_inp = _name(name_inp, ii, ji, inp_offset)
        elif ji - ii == 1:
            bit = f'{name_inp}[{ii} + {inp_offset}]' if inp_offset is not None else f'{name_inp}[{ii}]'
            _name_inp = f'{{{jo - io}{{{bit}}}}}'
        else:
            assert ii == ji == -1, f'Unexpected map_out entry: {(ii, ji), (io, jo)}'
            _name_inp = f"{jo - io}'b0"
        assignments.append(to_assign_str(_name_inp, _name_out))
    return assignments


def _bit_offset(sig: Signal) -> int:
    return sum(sum(p) for p in sig.raw.precisions[: sig.view[0]])


def _get_dyn_offset(sig: Signal) -> str | None:
    """Verilog bit-offset expression `idx * jump_width` for a dynamically-biased view, else None."""
    if sig._dynamic_bias is None:
        return None
    idx = sig._dynamic_bias[0]
    lo = _bit_offset(sig)
    hi = lo + sum(idx.bitwidths)
    return f'{idx.name}[{hi - 1}:{lo}] * {sig.jump_width}'


def gen_assignments_conn(conn: Conn) -> str:
    src_off, dst_off = _get_dyn_offset(conn.src), _get_dyn_offset(conn.dst)
    io_map, _ = gen_io_map(conn.src.precisions, conn.dst.precisions, True, _bit_offset(conn.src), _bit_offset(conn.dst))
    assignments = gen_assignments(io_map, conn.src.name, conn.dst.name, conn.clocked, src_off, dst_off)
    assignments_str = '\n        '.join(assignments)
    esig = conn.enable_if
    if esig is not None:
        enable_sig_name = f'{esig.name}[{esig.view[0]}]' if esig.raw.width > 1 else esig.name
    else:
        enable_sig_name = None

    if conn.alt_src is not None:
        alt_io_map, _ = gen_io_map(
            conn.alt_src.precisions, conn.dst.precisions, True, _bit_offset(conn.alt_src), _bit_offset(conn.dst)
        )
        alt_assignments = gen_assignments(
            alt_io_map, conn.alt_src.name, conn.dst.name, conn.clocked, _get_dyn_offset(conn.alt_src), dst_off
        )
        alt_assignments_str = '\n        '.join(alt_assignments)
        assert enable_sig_name is not None
        block = f"""    if ({enable_sig_name}) begin: _enabled_{conn.dst.name}
        {assignments_str}
    end else begin: _alt_{conn.dst.name}
        {alt_assignments_str}
    end
"""
    else:
        block = assignments_str

    if conn.clocked:
        if conn.dst.rst_if is not None:
            rst_sig = conn.dst.rst_if
            rst_sig_name = f'{rst_sig.name}[{rst_sig.view[0]}]' if rst_sig.raw.width > 1 else rst_sig.name
            block = block.replace('\n', '\n    ')
            block = f'    if (~{rst_sig_name}) begin: _not_rst\n{block}\n    end'
        return block
    else:
        return block


def _padded_prec(prec: Sequence[Precision]) -> Precision:
    kif = np.max(prec, axis=0)
    return Precision(bool(kif[0]), int(kif[1]), int(kif[2]))


def _to_def_str(sig: Signal, pad=False, force_wire: bool = False) -> str:
    if not pad:
        width = sig.width
    else:
        width = sig.size * sum(_padded_prec(sig.precisions))
    if sig.reg and not force_wire:
        _type = 'reg' if not sig.attrs else f'{sig.attrs} reg'
    else:
        _type = 'wire' if not sig.attrs else f'{sig.attrs} wire'
    return f'{_type} [{width - 1}:0] {sig.name}'


def to_header(fsm: FSM, name: str, pad: bool):
    inputs = [f'input {_to_def_str(sig, pad)}' for sig in fsm.inp_signals]
    if pad:
        inputs = ['// verilator lint_off UNUSEDSIGNAL'] + inputs + ['// verilator lint_on UNUSEDSIGNAL']

    need_clk = any(sig.reg for sig in fsm.signals.values())
    if need_clk:
        inputs = ['input clk'] + inputs

    outputs = [f'output {_to_def_str(sig, pad, force_wire=pad)}' for sig in fsm.out_signals]

    io_defs = ',\n    '.join(inputs + outputs)
    module_header = f'module {name} (\n    {io_defs}\n);'
    return module_header


def fsm_logic_gen(
    fsm: FSM,
    name: str,
    print_latency=False,
    timescale: str | None = '`timescale 1 ns / 1 ps',
    comb_logic_gen_fn=None,
    no_shreg: bool = False,
):
    comb_logic_gen_fn = comb_logic_gen_fn or comb_logic_gen

    if no_shreg:
        for sig in fsm.signals.values():
            if not sig.reg:
                continue
            sig.attrs = '(* shreg_extract = "no" *)'

    # header def

    module_header = to_header(fsm, name, False)

    # signal declare

    intermediate = [f'{_to_def_str(sig)};' for sig in fsm.internal_signals]
    signal_defs = '    ' + '\n    '.join(intermediate)

    # comb logic

    comb_ops = [
        f'{name} _inst_{name} (.model_inp(INTERNAL_{name}_inp), .model_out(INTERNAL_{name}_outp));' for name in fsm.logic.keys()
    ]

    # connections

    conns = [gen_assignments_conn(conn) for conn in fsm.comb_conns]
    if fsm.reg_conns:
        conns_reg = [gen_assignments_conn(conn) for conn in fsm.reg_conns]
        conns_reg_str = '\n        '.join(conns_reg)
        reg_assignment_block = f"""\n\n    always @(posedge clk) begin
            {conns_reg_str}
        end"""
    else:
        reg_assignment_block = ''

    # initial

    sig_need_initial = [sig for sig in fsm.signals.values() if sig.reg and sig.rst_to is not None]
    rst_values = [_rst_bin(sig) for sig in sig_need_initial]
    rst_assignments_str = '\n        '.join([f'{sig.name} = {rst_val};' for sig, rst_val in zip(sig_need_initial, rst_values)])
    if sig_need_initial:
        initial = f"""\n\n    initial begin: _fsm_init_registers
        {rst_assignments_str}
    end"""
    else:
        initial = ''

    # reset

    sig_need_reset = [sig for sig in sig_need_initial if sig.rst_if is not None]
    if sig_need_reset:
        rst_blocks = []
        for sig in sig_need_reset:
            rst_sig = sig.rst_if
            assert rst_sig is not None
            rst_sig_name = f'{rst_sig.name}[{rst_sig.view[0]}]' if rst_sig.raw.width > 1 else rst_sig.name
            rst_val = _rst_bin(sig)
            rst_block = f"""    if ({rst_sig_name}) begin: _reset_{sig.name}
        {sig.name} <= {rst_val};
    end"""
            rst_blocks.append(rst_block)
        rst_block_str = '\n'.join(rst_blocks)
        rst_block_str = '\n\n    always @(posedge clk) begin\n' + rst_block_str.replace('\n', '\n    ') + '\n    end'
    else:
        rst_block_str = ''

    module_body = '\n    '.join([signal_defs] + [''] + comb_ops + [''] + conns)
    module_body += reg_assignment_block + initial + rst_block_str

    module = f'{module_header}\n\n{module_body}\n\nendmodule\n'
    if timescale:
        module = f'{timescale}\n\n{module}'

    ret: dict[str, str] = {}
    for _name, comb in fsm.logic.items():
        ret[_name] = comb_logic_gen_fn(comb, _name, print_latency=print_latency, timescale=timescale)
    assert name not in ret, f'FSM name {name} conflicts with logic name'
    ret[name] = module
    return ret


def gen_io_map_sugar(precs: Sequence[Precision], direction: str, merge: bool):
    assert direction in ('inp', 'out')
    precs_uniform = [_padded_prec(precs)] * len(precs)
    precs0, precs1 = (precs_uniform, precs) if direction == 'inp' else (precs, precs_uniform)
    return gen_io_map(precs0, precs1, merge=merge)


def generate_io_wrapper(fsm: FSM, module_name: str, timescale: str | None = '`timescale 1 ns / 1 ps'):

    header = to_header(fsm, f'{module_name}_wrapper', True)
    declares = [f'wire [{signal.width - 1}:0] {signal.name}_packed;' for signal in fsm.inp_signals + fsm.out_signals]
    assignments = []
    for sig in fsm.inp_signals:
        map_inp, _ = gen_io_map_sugar(sig.precisions, direction='inp', merge=True)
        assignments += gen_assignments(map_inp, sig.name, f'{sig.name}_packed', False)
    for sig in fsm.out_signals:
        map_out, _ = gen_io_map_sugar(sig.precisions, direction='out', merge=True)
        assignments += gen_assignments(map_out, f'{sig.name}_packed', sig.name, False)

    clocked = any(sig.reg for sig in fsm.signals.values())
    port_assignments = [f'.{signal.name}({signal.name}_packed)' for signal in fsm.inp_signals + fsm.out_signals]
    if clocked:
        port_assignments = ['.clk(clk)'] + port_assignments
    port_assignments_str = ',\n        '.join(port_assignments)
    port_assignments_str = f'{module_name} {module_name}_inst (\n        {port_assignments_str}\n    );'

    module_body = '\n    '.join(declares) + '\n\n    ' + '\n    '.join(assignments) + '\n\n    ' + port_assignments_str
    module = f'{header}\n\n{module_body}\n\nendmodule\n'
    if timescale:
        module = f'{timescale}\n\n{module}'
    return module
