import numpy as np
from keras import ops
from keras.layers import BatchNormalization

from alkaid.trace import FVArray

from ._base import ReplayOperationBase, to_np_arr


class ReplayBatchNormalization(ReplayOperationBase):
    handles = (BatchNormalization,)

    def fused_scale_offset(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the fused ``(scale, offset)`` such that ``output = input * scale + offset``.

        Downstream packages may override this to supply bit-exact quantized
        values without having to separately express ``gamma``/``beta``/``mean``/
        ``variance``.
        """
        layer: BatchNormalization = self.op
        mean = to_np_arr(ops.cast(layer.moving_mean, layer.dtype))
        variance = to_np_arr(ops.cast(layer.moving_variance, layer.dtype))
        if layer.scale:
            gamma = to_np_arr(ops.cast(layer.gamma, layer.dtype))
        else:
            gamma = np.ones_like(mean)
        if layer.center:
            beta = to_np_arr(ops.cast(layer.beta, layer.dtype))
        else:
            beta = np.zeros_like(mean)
        scale = gamma / np.sqrt(variance + layer.epsilon)
        offset = beta - mean * scale
        return scale, offset

    def call(self, inputs: FVArray, mask=None) -> FVArray:
        layer: BatchNormalization = self.op
        scale, offset = self.fused_scale_offset()
        shape = [1] * inputs.ndim
        axis = layer.axis if isinstance(layer.axis, (list, tuple)) else [layer.axis]
        for a in axis:
            shape[a] = inputs.shape[a]
        return inputs * scale.reshape(shape) + offset.reshape(shape)  # type: ignore
