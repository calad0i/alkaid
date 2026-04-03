from ...types import CombLogic


def _retrace(comb: CombLogic) -> CombLogic:
    from ..fixed_variable_array import FVArray
    from ..tracer import trace

    inp = FVArray.from_kif(*comb.inp_kifs).as_new()
    out = comb(inp, quantize=False)
    return trace(inp, out, optimize=False)
