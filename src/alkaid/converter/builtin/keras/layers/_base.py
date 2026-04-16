import typing
from typing import Any

import keras
import numpy as np
from keras import ops
from keras.ops import convert_to_numpy

from alkaid.trace import FVariable, FVArray
from alkaid.trace.ops import relu


def _selu(arr: np.ndarray):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    if isinstance(arr, FVArray):
        return arr.apply(lambda x: scale * np.where(x > 0, x, alpha * (np.exp(x) - 1)))
    else:
        return scale * np.where(arr > 0, arr, alpha * (np.exp(arr) - 1))


def _glu(arr: np.ndarray):
    if isinstance(arr, FVArray):
        return arr.apply(lambda x: x * (1 / (1 + np.exp(-x))))
    else:
        return arr * (1 / (1 + np.exp(-arr)))


def _gelu(arr: np.ndarray):
    if isinstance(arr, FVArray):
        return arr.apply(lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    else:
        return 0.5 * arr * (1 + np.tanh(np.sqrt(2 / np.pi) * (arr + 0.044715 * np.power(arr, 3))))


def _swish(arr: np.ndarray):
    if isinstance(arr, FVArray):
        return arr.apply(lambda x: x / (1 + np.exp(-x)))
    else:
        return arr / (1 + np.exp(-arr))


def _softsign(arr: np.ndarray):
    if isinstance(arr, FVArray):
        return arr.apply(lambda x: x / (1 + np.abs(x)))
    else:
        return arr / (1 + np.abs(arr))


keras_numpy_unary_map = {
    keras.activations.linear: lambda x: x,
    keras.activations.relu: lambda x: np.maximum(0, x),
    keras.activations.tanh: np.tanh,
    keras.activations.sigmoid: lambda x: 1 / (1 + np.exp(-x)),
    keras.activations.swish: _swish,
    keras.activations.gelu: _gelu,
    keras.activations.elu: lambda x: np.where(x > 0, x, np.exp(x) - 1),
    keras.activations.selu: _selu,
    keras.activations.silu: lambda x: x / (1 + np.exp(-x)),
    keras.activations.softplus: lambda x: np.log1p(np.exp(x)),
    keras.activations.softsign: _softsign,
    keras.activations.exponential: lambda x: np.exp(x),
    keras.activations.hard_silu: lambda x: x * np.minimum(1, np.maximum(0, (x + 1) / 2)),
    keras.activations.hard_sigmoid: lambda x: np.minimum(1, np.maximum(0, (x + 1) / 2)),
    keras.activations.hard_swish: lambda x: x * np.minimum(1, np.maximum(0, (x + 1) / 2)),
    keras.activations.log_sigmoid: lambda x: -np.log1p(np.exp(-x)),
    keras.activations.glu: _glu,
}


def keras_activation_to_numpy(activation: Any) -> Any:
    assert activation is not keras.activations.softmax, 'Softmax activation is not supported in keras_activation_to_numpy.'
    if activation in keras_numpy_unary_map:
        return keras_numpy_unary_map[activation]
    else:
        return lambda x: ops.convert_to_numpy(activation(ops.convert_to_tensor(x)))


def to_np_arr(x: Any) -> np.ndarray:
    return np.asarray(convert_to_numpy(x))


_registry: dict[type, 'type[ReplayOperationBase]'] = {}


class HandlerRegMeta(type):
    """Metaclass for automatic registration of handler classes."""

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, typing.Any]):
        cls = super().__new__(mcs, name, bases, namespace)
        if name == 'ReplayOperationBase':
            return cls

        handles: type | tuple[type, ...] = namespace['handles']
        if not isinstance(handles, tuple):
            handles = (handles,)

        for handle in handles:
            _registry[handle] = cls  # type: ignore
        return cls


ARR_or_tuple_ARR = tuple[FVArray, ...] | FVArray


class ReplayOperationBase(metaclass=HandlerRegMeta):
    handles: tuple[type, ...] = ()
    __activation_handled__ = False

    def _load_weight(self, name: str) -> np.ndarray:
        w = getattr(self.op, name)
        if w is None:
            return np.array(0.0)
        return to_np_arr(w)

    def __init__(self, layer: 'keras.Operation'):
        assert isinstance(layer, self.handles)
        self.op: Any = layer

    def call(self, *args, **kwargs) -> ARR_or_tuple_ARR | dict[str, ARR_or_tuple_ARR]: ...

    @staticmethod
    def _normalize_to_tuple(
        v: FVArray | FVariable | tuple[FVArray | FVariable, ...],
    ) -> tuple[FVArray, ...]:
        if isinstance(v, FVArray):
            return (v,)
        elif isinstance(v, FVariable):
            return (FVArray(np.array([v])),)
        return tuple(FVArray(np.array(x)) if isinstance(x, FVariable) else x for x in v)

    def _wrap_call(self, *args, **kwargs) -> dict[str, tuple[FVArray, ...]]:
        r = self.call(*args, **kwargs)
        if isinstance(r, dict):
            r = {k: self._normalize_to_tuple(v) for k, v in r.items()}
        else:
            r = {'final': self._normalize_to_tuple(r)}
        return r

    def __call__(self, *args, **kwargs) -> dict[str, tuple[FVArray, ...]]:
        assert all(not isinstance(a, FVArray) for a in kwargs.values())

        layer: keras.layers.Layer = self.op
        assert kwargs.pop('training', False) is False, 'Training mode is not supported in mirror operation'

        trace: dict[str, tuple[FVArray, ...]] = {}
        outputs = self._wrap_call(*args, **kwargs)
        trace.update(outputs)
        trace['post_call'] = trace['final']

        if not self.__activation_handled__:
            activation = getattr(layer, 'activation', keras.activations.linear)
            if activation is not keras.activations.linear:
                if activation is keras.activations.relu:
                    assert len(trace['post_call']) == 1, 'ReLU activation is expected to have a single output'
                    trace['final'] = (relu(trace['post_call'][0]),)
                else:
                    raise NotImplementedError(
                        f'Activation {activation} is not allowed in activation= field for common layers.'
                        ' Use dedicated QUnaryFunctionLUT layer instead.'
                    )

        trace['final'] = trace.pop('final')

        return trace
