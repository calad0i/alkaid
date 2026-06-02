from ....types import CombLogic, Pipeline
from ..verilog.io_wrapper import BitMap, gen_io_map_sugar


def _loc(i: int, j: int):
    return f'({i} downto {j})' if i != j else f'({i})'


def gen_assignments(bit_map: list[BitMap], name_inp: str, name_out: str):
    assignment = []
    for (ii, ji), (io, jo) in bit_map:
        if ji - ii == jo - io:
            assignment.append(f'{name_out}{_loc(jo - 1, io)} <= {name_inp}{_loc(ji - 1, ii)};')
        else:
            if ji - ii == 1:
                value = f'{name_inp}({ii})'
            else:
                assert ii == ji == -1, f'Unexpected map entry: {(ii, ji), (io, jo)}'
                value = "'0'"
            pad = f'(others => {value})' if jo - io > 1 else value
            assignment.append(f'{name_out}{_loc(jo - 1, io)} <= {pad};')
    return '\n    '.join(assignment)


def generate_io_wrapper(sol: CombLogic | Pipeline, module_name: str, pipelined: bool = False):

    map_inp, (w_uniform_inp, w_inp) = gen_io_map_sugar(sol.inp_qint, direction='inp', merge=True)
    map_out, (w_out, w_uniform_out) = gen_io_map_sugar(sol.out_qint, direction='out', merge=True)

    inp_assignment_str = gen_assignments(map_inp, 'model_inp', 'packed_inp')
    out_assignment_str = gen_assignments(map_out, 'packed_out', 'model_out')

    clk_and_rst_inp, clk_and_rst_bind = '', ''
    if pipelined:
        clk_and_rst_inp = '\n    clk:in std_logic;'
        clk_and_rst_bind = '\n        clk=>clk,'

    return f"""library ieee;
use ieee.std_logic_1164.all;
entity {module_name}_wrapper is port({clk_and_rst_inp}
    model_inp:in std_logic_vector({w_uniform_inp - 1} downto {0});
    model_out:out std_logic_vector({w_uniform_out - 1} downto {0})
);
end entity {module_name}_wrapper;

architecture rtl of {module_name}_wrapper is
    signal packed_inp:std_logic_vector({w_inp - 1} downto {0});
    signal packed_out:std_logic_vector({w_out - 1} downto {0});

begin
    {inp_assignment_str}

    op:entity work.{module_name} port map({clk_and_rst_bind}
        model_inp=>packed_inp,
        model_out=>packed_out
    );

    {out_assignment_str}

end architecture rtl;
"""
