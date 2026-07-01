from ....types import CombLogic, Op
from ..verilog.lookup import lookup_name, lookup_ops, table_hex


def _hex_to_bin(value: str, width: int) -> str:
    bits = ''.join('XXXX' if char == 'X' else f'{int(char, 16):04b}' for char in value)
    return bits[-width:]


def _lookup_def(sol: CombLogic, op: Op, name: str) -> str:
    bw_in = sum(sol.ops[op.addr[0]].qint.kif)
    bw_out = sum(op.qint.kif)

    cases = []
    for addr, value in enumerate(table_hex(sol, op)):
        cases.append(f'            when {addr} => outp <= "{_hex_to_bin(value, bw_out)}";')
    case_body = '\n'.join(cases)
    return f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {name} is
    port (
        inp: in std_logic_vector({bw_in - 1} downto 0);
        outp: out std_logic_vector({bw_out - 1} downto 0)
    );
end entity {name};

architecture rtl of {name} is
begin
    process(inp)
    begin
        case to_integer(unsigned(inp)) is
{case_body}
            when others => outp <= (others => 'X');
        end case;
    end process;
end architecture rtl;"""


def lookup_source(combs: list[CombLogic]) -> str:
    defs = {}
    for comb in combs:
        for op in lookup_ops(comb):
            name = lookup_name(comb, op)
            if name not in defs:
                defs[name] = _lookup_def(comb, op, name)
    if not defs:
        return ''
    return '\n\n'.join(defs.values()) + '\n'
