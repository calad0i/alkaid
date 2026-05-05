import random

import numpy as np
import pytest

from alkaid._binary import scm_solve
from alkaid.types import CombLogic


@pytest.mark.parametrize('k_b', [(0, 8), (1, 8), (2, 8), (3, 8), (0, 16), (1, 16), (2, 16), (0, 24), (1, 24), (0, 32)])
def test_scm_solve(k_b):
    k, b = k_b
    rng = np.random.default_rng(0xC0DE + 17 * k + b)
    values = np.round((rng.random(100) - 0.5) * 2 ** (b + 1)).astype(np.float64)
    for v in values:
        sol: CombLogic = scm_solve(v, k=k)
        r = sol([1.0], quantize=False)[0]
        assert r == v, f'k={k} v={v}: got {r}, expected {v}'


@pytest.mark.parametrize(
    ('constant', 'k', 'expected_adders'),
    [
        (805, 0, 3),
        (363, 0, 3),
        (4875, 0, 3),
        (3633, 0, 3),
        (2325, 1, 3),
        (105, 0, 2),
        (1395, 2, 3),
        (46425, 2, 4),
    ],
)
def test_scm_paper_regressions(constant, k, expected_adders):
    sol: CombLogic = scm_solve(float(constant), k=k)
    assert sol([1.0], quantize=False)[0] == constant
    assert sum(1 for op in sol.ops if op.opcode in (0, 1)) == expected_adders


@pytest.mark.parametrize(('k', 'expected_avg'), [(0, 4.38), (1, 4.33)])
def test_19b_avg_adder_gate(k, expected_avg):
    rng = random.Random(0xC0DE)
    values = [rng.randrange(1, 1 << 19) | 1 for _ in range(100)]
    total = 0
    for c in values:
        sol: CombLogic = scm_solve(float(c), k=k)
        total += sum(1 for op in sol.ops if op.opcode in (0, 1))
    assert total / len(values) <= expected_avg + 0.05
