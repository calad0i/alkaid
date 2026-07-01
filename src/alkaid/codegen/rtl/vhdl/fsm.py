from collections.abc import Sequence

import numpy as np

from ....stateful import FSM, Conn, Signal
from ....types import Precision
from ..verilog.io_map import BitMap, gen_io_map
from .comb import comb_logic_gen
from .lookup import lookup_source


def _rst_literal(sig: Signal) -> str:
    assert sig.rst_to is not None
    shift = 0
    value = 0
    for rst, prec in zip(sig.rst_to, sig.precisions):
        width = sum(prec)
        raw = round(float(rst) * (2.0**prec.fractional)) & ((1 << width) - 1)
        value |= raw << shift
        shift += width
    return f'"{value:0{sig.width}b}"'


def _padded_prec(precs: Sequence[Precision]) -> Precision:
    kif = np.max(precs, axis=0)
    return Precision(bool(kif[0]), int(kif[1]), int(kif[2]))


def _bit_offset(sig: Signal) -> int:
    return sum(sum(prec) for prec in sig.raw.precisions[: sig.view[0]])


def _idx(i: int, offset: str | None) -> str:
    if offset is None:
        return str(i)
    return f'({offset})' if i == 0 else f'(({offset}) + {i})'


def _bits(name: str, lo: int, hi: int, offset: str | None = None) -> str:
    return f'{name}({_idx(hi - 1, offset)} downto {_idx(lo, offset)})'


def _dyn_offset(sig: Signal) -> str | None:
    if sig._dynamic_bias is None:
        return None
    idx = sig._dynamic_bias[0]
    lo = _bit_offset(sig)
    hi = lo + sum(idx.bitwidths)
    return f'to_integer(unsigned({_bits(idx.name, lo, hi)})) * {sig.jump_width}'


def _single_bit(sig: Signal) -> str:
    assert sig.size == 1 and sig.width == 1
    idx = sig.view[0] if sig.raw.width > 1 else 0
    return f'{sig.name}({idx})'


def _mapped_assignments(
    io_map: list[BitMap],
    src: str,
    dst: str,
    src_offset: str | None = None,
    dst_offset: str | None = None,
) -> list[str]:
    lines = []
    for (ii, ji), (io, jo) in io_map:
        lhs = _bits(dst, io, jo, dst_offset)
        if ji - ii == jo - io:
            rhs = _bits(src, ii, ji, src_offset)
        elif ji - ii == 1:
            rhs = f'(others => {src}({_idx(ii, src_offset)}))'
        else:
            assert ii == ji == -1, f'Unexpected map entry: {(ii, ji), (io, jo)}'
            rhs = "(others => '0')"
        lines.append(f'{lhs} <= {rhs};')
    return lines


def _assignments(src: Signal, dst: Signal) -> list[str]:
    io_map, _ = gen_io_map(src.precisions, dst.precisions, True, _bit_offset(src), _bit_offset(dst))
    return _mapped_assignments(io_map, src.name, dst.name, _dyn_offset(src), _dyn_offset(dst))


def _conn_block(conn: Conn, indent: int) -> str:
    pad = ' ' * indent
    lines = _assignments(conn.alt_src if conn.alt_src is not None else conn.src, conn.dst)
    src_lines = _assignments(conn.src, conn.dst)

    if conn.enable_if is None or conn.alt_src is None:
        return '\n'.join(f'{pad}{line}' for line in src_lines)

    inner = ' ' * (indent + 4)
    src = '\n'.join(f'{inner}{line}' for line in src_lines)
    alt = '\n'.join(f'{inner}{line}' for line in lines)
    return f"""{pad}if {_single_bit(conn.enable_if)} = '1' then
{src}
{pad}else
{alt}
{pad}end if;"""


def _conn_stmt(conn: Conn) -> str:
    if conn.enable_if is None or conn.alt_src is None:
        return _conn_block(conn, 4)
    return f"""    process(all) begin
{_conn_block(conn, 8)}
    end process;"""


def _register_process(fsm: FSM) -> str:
    conns_by_dst: dict[str, list[Conn]] = {}
    for conn in fsm.reg_conns:
        conns_by_dst.setdefault(conn.dst.name, []).append(conn)

    sections = []
    for sig in fsm.signals.values():
        if not sig.reg:
            continue
        body = '\n'.join(_conn_block(conn, 12) for conn in conns_by_dst.get(sig.name, ()))
        if sig.rst_if is not None and sig.rst_to is not None:
            body = body or '            null;'
            sections.append(
                f"        if {_single_bit(sig.rst_if)} = '1' then\n"
                f'            {sig.name} <= {_rst_literal(sig)};\n'
                f'        else\n'
                f'{body}\n'
                f'        end if;'
            )
        elif body:
            sections.append(body)

    if not sections:
        return ''
    body = '\n'.join(sections)
    return f"""
    process(clk) begin
        if rising_edge(clk) then
{body}
        end if;
    end process;"""


def _header(fsm: FSM, entity: str, pad: bool) -> str:
    def width(sig: Signal) -> int:
        return sig.size * sum(_padded_prec(sig.precisions)) if pad else sig.width

    ports = [f'{sig.name}: in std_logic_vector({width(sig) - 1} downto 0)' for sig in fsm.inp_signals]
    if any(sig.reg for sig in fsm.signals.values()):
        ports = ['clk: in std_logic'] + ports
    ports += [f'{sig.name}: out std_logic_vector({width(sig) - 1} downto 0)' for sig in fsm.out_signals]
    ports_str = ';\n    '.join(ports)
    return f"""entity {entity} is port(
    {ports_str}
);
end entity {entity};"""


def fsm_logic_gen(
    fsm: FSM,
    name: str,
    print_latency=False,
    timescale: str | None = None,
    comb_logic_gen_fn=None,
    no_shreg: bool = False,
):
    del timescale
    comb_logic_gen_fn = comb_logic_gen_fn or comb_logic_gen

    declarations = []
    for sig in fsm.internal_signals:
        init = f' := {_rst_literal(sig)}' if sig.reg and sig.rst_to is not None else ''
        declarations.append(f'signal {sig.name}: std_logic_vector({sig.width - 1} downto 0){init};')
    if no_shreg:
        declarations.append('attribute shreg_extract: string;')
        declarations += [f'attribute shreg_extract of {sig.name}: signal is "no";' for sig in fsm.signals.values() if sig.reg]

    instances = [
        f'inst_{logic}: entity work.{logic} port map(model_inp=>INTERNAL_{logic}_inp, model_out=>INTERNAL_{logic}_outp);'
        for logic in fsm.logic
    ]
    body = '\n    '.join(instances + [_conn_stmt(conn) for conn in fsm.comb_conns])
    decls = '\n    '.join(declarations)

    ret = {}
    for logic, comb in fsm.logic.items():
        ret[logic] = comb_logic_gen_fn(comb, logic, print_latency=print_latency, timescale=None)
    if comb_logic_gen_fn is comb_logic_gen:
        table_source = lookup_source(list(fsm.logic.values()))
        if table_source:
            ret[f'{name}_tables'] = table_source
    assert name not in ret, f'FSM name {name} conflicts with generated logic name'
    ret[name] = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

{_header(fsm, name, False)}

architecture rtl of {name} is
    {decls}
begin
    {body}{_register_process(fsm)}

end architecture rtl;
"""
    return ret


def gen_io_map_sugar(precs: Sequence[Precision], direction: str, merge: bool):
    assert direction in ('inp', 'out')
    uniform = [_padded_prec(precs)] * len(precs)
    precs0, precs1 = (uniform, precs) if direction == 'inp' else (precs, uniform)
    return gen_io_map(precs0, precs1, merge=merge)


def generate_io_wrapper(fsm: FSM, module_name: str, timescale: str | None = None):
    del timescale
    ports = fsm.inp_signals + fsm.out_signals
    packed = {sig.name: f'p{i}' for i, sig in enumerate(ports)}
    declarations = [f'signal {packed[sig.name]}: std_logic_vector({sig.width - 1} downto 0);' for sig in ports]

    assignments = []
    for sig in fsm.inp_signals:
        assignments += _mapped_assignments(
            gen_io_map_sugar(sig.precisions, direction='inp', merge=True)[0], sig.name, packed[sig.name]
        )
    for sig in fsm.out_signals:
        assignments += _mapped_assignments(
            gen_io_map_sugar(sig.precisions, direction='out', merge=True)[0], packed[sig.name], sig.name
        )

    port_map = [f'{sig.name}=>{packed[sig.name]}' for sig in ports]
    if any(sig.reg for sig in fsm.signals.values()):
        port_map = ['clk=>clk'] + port_map

    decls = '\n    '.join(declarations)
    assigns = '\n    '.join(assignments)
    port_map_str = ',\n        '.join(port_map)
    return f"""library ieee;
use ieee.std_logic_1164.all;

{_header(fsm, f'{module_name}_wrapper', True)}

architecture rtl of {module_name}_wrapper is
    {decls}
begin
    {assigns}

    {module_name}_inst: entity work.{module_name} port map(
        {port_map_str}
    );

end architecture rtl;
"""
