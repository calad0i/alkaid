import typing
from collections.abc import Callable
from typing import TypedDict

import numpy as np

from .._binary import cmvm_solve, kernel_decompose
from ..types import CombLogic, Op, QInterval

if typing.TYPE_CHECKING:
    from ..trace import FVArray


class solver_options_t(TypedDict, total=False):
    method0: str
    method1: str
    hard_dc: int
    decompose_dc: int
    adder_size: int
    search_all_decompose_dc: bool
    offload_fn: None | Callable[[np.ndarray, 'FVArray'], np.ndarray]
    """
    Callable taking in (constant_matrix, fixed_variable_array) and returning
    a boolean mask of which weights to offload to multiplication operations.
    """


__all__ = ['cmvm_solve', 'QInterval', 'Op', 'CombLogic', 'kernel_decompose']
