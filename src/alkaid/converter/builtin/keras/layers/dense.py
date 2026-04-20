import numpy as np
from keras.layers import (
    Dense,
    EinsumDense,
)

from alkaid.trace import FVArray

from ._base import ReplayOperationBase


class ReplayDense(ReplayOperationBase):
    handles = (Dense, EinsumDense)

    def call(self, inputs: FVArray) -> FVArray:
        op = self.op
        kernel = self._load_weight('kernel')
        bias = self._load_weight('bias')
        if isinstance(op, Dense):
            eq = '...c,cC->...C'
        elif isinstance(op, EinsumDense):
            eq = op.equation
        else:
            raise TypeError(f'Unsupported layer type: {type(op)}')
        return np.einsum(eq, inputs, kernel) + bias


__all__ = ['ReplayDense']
