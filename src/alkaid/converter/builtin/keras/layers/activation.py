import numpy as np
from keras.layers import LeakyReLU, PReLU, ReLU

from alkaid.trace import FVArray
from alkaid.trace.ops import relu

from ._base import ReplayOperationBase, to_np_arr


class ReplayReLU(ReplayOperationBase):
    handles = (ReLU, LeakyReLU, PReLU)

    def call(self, inputs: FVArray) -> FVArray:
        op = self.op
        if isinstance(op, ReLU):
            th, neg, maxv = op.threshold, op.negative_slope, op.max_value
        elif isinstance(op, LeakyReLU):
            th, neg, maxv = 0, op.negative_slope, None
        elif isinstance(op, PReLU):
            th, neg, maxv = 0, to_np_arr(op.alpha), None
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
