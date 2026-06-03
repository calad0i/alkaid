import numpy as np

from ....stateful import FSM, Conn, Signal
from .comb import comb_logic_gen
from .io_wrapper import BitMap, gen_io_map


def _rst_bin(sig: Signal) -> str:
    assert sig.rst_to is not None
    val = np.round(
        np.array(sig.rst_to, dtype=np.float64) * 2.0 ** -np.array([prec.fractional for prec in sig.precisions])
    ).astype(np.uint64)
    s = 0
    number = 0
    for n, p in zip(val, sig.precisions):
        w = sum(p)
        number = n << s | number
        s += w
    return 'h' + hex(number)[2:].upper()


def gen_assignments(map_out: list[BitMap], name_inp: str, name_out: str, clocked: bool):
    def to_assign_str(name_inp: str, name_out: str):
        if not clocked:
            return f'assign {name_out} = {name_inp};'
        else:
            return f'{name_out} <= {name_inp};'

    assignments = []
    for (ii, ji), (io, jo) in map_out:
        name_out = f'{name_out}[{jo - 1}:{io}]'
        if ji - ii == jo - io:
            name_inp = f'{name_inp}[{ji - 1}:{ii}]'
        elif ji - ii == 1:
            name_inp = f'{{{jo - io}{{{name_inp}[{ii}]}}}}'
        else:
            assert ii == ji == -1, f'Unexpected map_out entry: {(ii, ji), (io, jo)}'
            name_inp = f"{jo - io}'b0"
        assignments.append(to_assign_str(name_inp, name_out))
    return assignments


def gen_assignments_conn(conn: Conn) -> str:
    io_map, _ = gen_io_map(conn.src.precisions, conn.dst.precisions, True, conn.src.view_interval[0], conn.dst.view_interval[0])
    assignments = gen_assignments(io_map, conn.src.name, conn.dst.name, conn.clocked)
    assignments_str = '\n        '.join(assignments)
    esig = conn.enable_if
    if esig is not None:
        enable_sig_name = f'{esig.name}[{esig.view_interval[0]}]' if esig.raw.width > 1 else esig.name
    else:
        enable_sig_name = None

    if conn.alt_src is not None:
        alt_io_map, _ = gen_io_map(
            conn.alt_src.precisions, conn.dst.precisions, True, conn.alt_src.view_interval[0], conn.dst.view_interval[0]
        )
        alt_assignments = gen_assignments(alt_io_map, conn.alt_src.name, conn.dst.name, conn.clocked)
        alt_assignments_str = '\n        '.join(alt_assignments)
        assert enable_sig_name is not None
        block = f"""    if ({enable_sig_name}) begin: _enabled
        {assignments_str}
    end else begin: _alt
        {alt_assignments_str}
    end
"""
    else:
        block = assignments_str

    if conn.clocked:
        if conn.dst.rst_if is not None:
            rst_sig = conn.dst.rst_if
            rst_sig_name = f'{rst_sig.name}[{rst_sig.view_interval[0]}]' if rst_sig.raw.width > 1 else rst_sig.name
            block = block.replace('\n', '\n    ')
            block = f'    if (~{rst_sig_name}) begin: _not_rst\n{block}\n    end'
        return block
    else:
        return block


def fsm_logic_gen(
    fsm: FSM,
    name: str,
    print_latency=False,
    timescale: str | None = '`timescale 1 ns / 1 ps',
    comb_logic_gen_fn=None,
    no_shreg: bool = False,
):
    comb_logic_gen_fn = comb_logic_gen_fn or comb_logic_gen

    def _to_def_str(sig: Signal) -> str:
        width = sig.width
        if sig.reg:
            _type = 'reg' if not sig.attrs else f'{sig.attrs} reg'
        else:
            _type = 'wire' if not sig.attrs else f'{sig.attrs} wire'
        return f'{_type} [{width - 1}:0] {sig.name}'

    if no_shreg:
        for sig in fsm.signals.values():
            if not sig.reg:
                continue
            sig.attrs = '(* shreg_extract = "no" *)'

    # header def

    inputs = [f'input {_to_def_str(sig)}' for sig in fsm.inp_signals]
    outputs = [f'output {_to_def_str(sig)}' for sig in fsm.out_signals]

    need_clk = any(sig.reg for sig in fsm.signals.values())
    if need_clk:
        inputs = ['input clk'] + inputs
    io_defs = ',\n    '.join(inputs + outputs)
    module_header = f'module {name} (\n    {io_defs}\n);'

    # signal declare

    intermediate = [f'{_to_def_str(sig)};' for sig in fsm.internal_signals]
    signal_defs = '    ' + '\n    '.join(intermediate)

    # comb logic

    comb_ops = [f'{name} _inst_{name} (.model_inp(__{name}_in), .model_out(__{name}_out));' for name in fsm.logic.keys()]

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
    rst_assignments_str = '\n        '.join(
        [f'assign {sig.name} = {rst_val};' for sig, rst_val in zip(sig_need_initial, rst_values)]
    )
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
            rst_sig_name = f'{rst_sig.name}[{rst_sig.view_interval[0]}]' if rst_sig.raw.width > 1 else rst_sig.name
            rst_val = _rst_bin(sig)
            rst_block = f"""    if (~{rst_sig_name}) begin: _reset_{sig.name}
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
