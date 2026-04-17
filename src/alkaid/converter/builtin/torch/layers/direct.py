from collections.abc import Callable

import numpy as np
import torch
from torch import no_grad

from alkaid.trace import FVArray

from .functional import _functional, to_np_arr


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorized erf via torch (matches torch's float32 computation)."""
    return np.asarray(to_np_arr(torch.erf(torch.tensor(x))))


def _wrap_apply(fn):
    """Turn a numpy-only unary function into one that also works on FVArray via `.apply()`."""

    def wrapper(arr):
        if isinstance(arr, FVArray):
            return arr.apply(fn)
        return fn(arr)

    return wrapper


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)


def _gelu(x):
    # torch default is approximate=False (exact erf formulation)
    return 0.5 * x * (1 + _erf(x / np.sqrt(2)))


def _selu(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


def _softplus(x):
    return np.log1p(np.exp(x))


def _softsign(x):
    return x / (1 + np.abs(x))


def _hard_sigmoid(x):
    return np.clip(x / 6.0 + 0.5, 0.0, 1.0)


def _hard_silu(x):
    return x * _hard_sigmoid(x)


def _hard_tanh(x):
    return np.clip(x, -1, 1)


def _log_sigmoid(x):
    return -np.log1p(np.exp(-x))


torch_numpy_unary_map: dict[str, Callable] = {
    'linear': lambda x: x,
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': _wrap_apply(_sigmoid),
    'silu': _wrap_apply(_silu),
    'gelu': _wrap_apply(_gelu),
    'elu': _wrap_apply(_elu),
    'selu': _wrap_apply(_selu),
    'softplus': _softplus,
    'softsign': _wrap_apply(_softsign),
    'hardsigmoid': _wrap_apply(_hard_sigmoid),
    'hardswish': _wrap_apply(_hard_silu),
    'hardtanh': _hard_tanh,
    'logsigmoid': _log_sigmoid,
    'log_sigmoid': _log_sigmoid,
    'relu6': lambda x: np.clip(x, 0, 6),
    'tanh': np.tanh,
    'sinh': np.sinh,
    'cosh': np.cosh,
    'exp': np.exp,
    'expm1': np.expm1,
    'log': np.log,
    'log1p': np.log1p,
    'log2': np.log2,
    'log10': np.log10,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'abs': np.abs,
    'absolute': np.abs,
    'sign': np.sign,
    'round': np.round,
    'floor': np.floor,
    'ceil': np.ceil,
    'reciprocal': lambda x: 1 / x,
    'neg': lambda x: -x,
    'negative': lambda x: -x,
    'square': lambda x: x * x,
    'sqrt': np.sqrt,
    'rsqrt': lambda x: 1 / np.sqrt(x),
    'arcsin': np.arcsin,
    'arccos': np.arccos,
    'arctan': np.arctan,
    'asin': np.arcsin,
    'acos': np.arccos,
    'atan': np.arctan,
    'arcsinh': np.arcsinh,
    'arccosh': np.arccosh,
    'arctanh': np.arctanh,
    'asinh': np.arcsinh,
    'acosh': np.arccosh,
    'atanh': np.arctanh,
    'flatten': np.ravel,
}


# Names whose torch signature accepts extra kwargs (alpha, min_val, max_val, ...)
# or structural positional args (weight, start_dim) — registered explicitly in
# functional.py instead so the unary-map shim doesn't shadow them.
_EXPLICIT_IN_FUNCTIONAL = {'elu', 'flatten', 'hardtanh', 'leaky_relu', 'linear', 'prelu'}


for _name, _impl in torch_numpy_unary_map.items():
    if _name in _EXPLICIT_IN_FUNCTIONAL:
        continue
    _wrapped = no_grad()(_impl)
    for _ns in (torch, torch.nn.functional):
        if hasattr(_ns, _name):
            _functional(getattr(_ns, _name))(_wrapped)
