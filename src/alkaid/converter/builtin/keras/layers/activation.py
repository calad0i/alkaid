from typing import Any

import keras
import numpy as np
from keras import ops
from keras.layers import Activation, LeakyReLU, PReLU, ReLU

from alkaid.trace import FVArray
from alkaid.trace.ops import relu

from ._base import ReplayOperationBase


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorized erf via keras.ops (matches Keras' float32 computation)."""
    return np.asarray(ops.convert_to_numpy(ops.erf(x)))


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
    # Keras default is approximate=False (exact erf formulation)
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
    # Match JAX/Keras: relu6(x + 3) / 6 -- clip first then divide for float32-stable results
    return np.clip(x + 3.0, 0.0, 6.0) / 6.0


def _hard_silu(x):
    return x * _hard_sigmoid(x)


def _log_sigmoid(x):
    return -np.log1p(np.exp(-x))


def _hard_tanh(x):
    return np.clip(x, -1, 1)


keras_numpy_unary_map = {
    'linear': lambda x: x,
    'relu': lambda x: np.maximum(0, x),
    'tanh': np.tanh,
    'sigmoid': _wrap_apply(_sigmoid),
    'swish': _wrap_apply(_silu),
    'silu': _wrap_apply(_silu),
    'gelu': _wrap_apply(_gelu),
    'elu': _wrap_apply(_elu),
    'selu': _wrap_apply(_selu),
    'softplus': _softplus,
    'softsign': _wrap_apply(_softsign),
    'exponential': np.exp,
    'hard_silu': _wrap_apply(_hard_silu),
    'hard_sigmoid': _wrap_apply(_hard_sigmoid),
    'hard_swish': _wrap_apply(_hard_silu),
    'log_sigmoid': _log_sigmoid,
    'hard_tanh': _hard_tanh,
    'relu6': lambda x: np.clip(x, 0, 6),
}


def keras_unary_to_numpy(activation: Any, allow_unknown: bool = True) -> Any:
    assert activation is not keras.activations.softmax, 'Softmax activation is not supported in keras_activation_to_numpy.'
    if activation.__name__ in keras_numpy_unary_map:
        return keras_numpy_unary_map[activation.__name__]
    elif allow_unknown:
        return lambda x: x.apply(lambda y: ops.convert_to_numpy(activation(ops.convert_to_tensor(y))))
    else:
        raise ValueError(f'Unsupported activation: {activation}')


class ReplayReLU(ReplayOperationBase):
    handles = (ReLU, LeakyReLU, PReLU)

    def call(self, inputs: FVArray) -> FVArray:
        op = self.op
        if isinstance(op, ReLU):
            th, neg, maxv = op.threshold, op.negative_slope, op.max_value
        elif isinstance(op, LeakyReLU):
            th, neg, maxv = 0, op.negative_slope, None
        elif isinstance(op, PReLU):
            th, neg, maxv = 0, self._load_weight('alpha'), None
        else:
            raise TypeError(f'Unsupported activation layer: {type(op)}')

        if th == 0 and np.all(neg == 0) and maxv is None:
            return relu(inputs)

        pos_part = inputs if maxv is None else np.minimum(inputs, maxv)  # type: ignore

        if th != 0:
            z_cond = inputs - (th + 2.0 ** (-inputs.kif[2] - 1))
        else:
            z_cond = inputs.ravel()

        neg_part = (inputs[None] - th) * neg
        return np.where(z_cond < 0, neg_part, pos_part)  # type: ignore


class ReplayActivation(ReplayOperationBase):
    __activation_handled__ = True

    handles = (Activation,)

    def call(self, inputs: FVArray) -> FVArray:
        return keras_unary_to_numpy(self.op.activation, allow_unknown=True)(inputs)
