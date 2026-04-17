import typing
from typing import Any

import keras
import numpy as np
from keras.ops import convert_to_numpy

from alkaid.trace import FVariable, FVArray


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
    handles: tuple[type, ...] | type = ()
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
        from .activation import keras_unary_to_numpy

        assert all(not isinstance(a, FVArray) for a in kwargs.values())

        op: keras.Operation = self.op
        assert kwargs.pop('training', False) is False, 'Training mode is not supported in mirror operation'

        trace: dict[str, tuple[FVArray, ...]] = {}
        outputs = self._wrap_call(*args, **kwargs)
        trace.update(outputs)
        trace['post_call'] = trace['final']

        if not self.__activation_handled__:
            activation = keras_unary_to_numpy(getattr(op, 'activation', keras.activations.linear), allow_unknown=False)
            assert len(trace['post_call']) == 1
            trace['final'] = (activation(trace['post_call'][0]),)

        trace['final'] = trace.pop('final')

        return trace
