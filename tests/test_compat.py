from pathlib import Path

import numpy as np

from alkaid.trace import FVArray, trace
from alkaid.trace.ops import quantize
from alkaid.types import CombLogic


def v2_operation(x):  # x[16,4]
    x = quantize(x, 1, 8, 4)
    a = np.maximum(x, 0)
    b = x[:, 1::2].T
    c = np.round(np.sin(b) * np.pi)
    d = np.repeat(c, 2, axis=0) * 3 + 4
    e = np.max(np.stack([d, -d * 2], axis=0), axis=0)
    f = c @ e.T
    g = np.einsum('ij,kj->ik', c, d)
    h = np.sum(g, axis=0)
    idx = np.argsort(h)[:2]
    j = h[idx].ravel()
    return j[None] + f[..., None] + a[-2::-4, :2] + ((a[0, -2:] > 0) & (a[-1, :2] > 0))


def test_compat_v2():
    inp = FVArray.new((16, 4))
    out = v2_operation(inp)
    comb = trace(inp, out)
    comb_load = CombLogic.load(Path(__file__).parent / '_legacy_models/v2.json.gz')
    assert all(
        op1.opcode == op2.opcode and op1.addr == op2.addr and op1.data == op2.data for op1, op2 in zip(comb.ops, comb_load.ops)
    )
