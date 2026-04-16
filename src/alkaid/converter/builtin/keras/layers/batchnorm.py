from keras import ops
from keras.layers import BatchNormalization

from alkaid.trace import FVArray

from ._base import ReplayOperationBase


def _scaler_and_offset(layer: BatchNormalization):
    mean = ops.cast(layer.moving_mean, layer.dtype)
    variance = ops.cast(layer.moving_variance, layer.dtype)

    if layer.scale:
        bn_gamma = ops.cast(layer.gamma, layer.dtype)
    else:
        bn_gamma = 1

    if layer.center:
        bn_beta = ops.cast(layer.beta, layer.dtype)
    else:
        bn_beta = 1

    scale = bn_gamma / ops.sqrt(variance + layer.epsilon)  # type: ignore
    offset = bn_beta - mean * scale  # type: ignore

    return scale, offset


class ReplayBatchNormalization(ReplayOperationBase):
    handles = (BatchNormalization,)

    def call(self, inputs: FVArray, mask=None) -> FVArray:
        layer: BatchNormalization = self.op
        scale, bias = map(ops.convert_to_numpy, _scaler_and_offset(layer))
        # Build broadcast shape matching the layer's axis
        shape = [1] * inputs.ndim
        axis = layer.axis if isinstance(layer.axis, (list, tuple)) else [layer.axis]
        for a in axis:
            shape[a] = inputs.shape[a]
        return inputs * scale.reshape(shape) + bias.reshape(shape)  # type: ignore
