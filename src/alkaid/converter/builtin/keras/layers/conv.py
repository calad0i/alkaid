import numpy as np
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.src.ops.image import ExtractPatches

from alkaid.trace import FVArray
from alkaid.trace.ops import extract_patches

from ._base import ReplayOperationBase


class ReplayExtractPatches(ReplayOperationBase):
    handles = (ExtractPatches,)

    def call(self, images: FVArray) -> FVArray:
        op: ExtractPatches = self.op
        return extract_patches(
            images,
            size=op.size,
            strides=op.strides,
            dilation_rate=op.dilation_rate,
            padding=op.padding,
            data_format=op.data_format,
        )


class ReplayConv(ReplayOperationBase):
    handles = (Conv1D, Conv2D, Conv3D)

    def call(self, inputs: FVArray) -> FVArray:
        layer: Conv1D | Conv2D | Conv3D = self.op
        kernel = self._load_weight('kernel')
        bias = self._load_weight('bias')
        groups = layer.groups

        x = extract_patches(
            inputs,
            size=layer.kernel_size,
            strides=layer.strides,
            dilation_rate=layer.dilation_rate,
            padding=layer.padding,
            data_format=layer.data_format,
        )
        # x.shape = (batch, out_spa..., kernel_volume * ch_in)

        ch_out = kernel.shape[-1]
        _ch_out = ch_out // groups
        x = x.reshape(*x.shape[:-1], -1, groups)  # type: ignore
        kernel = kernel.reshape(-1, groups, _ch_out)

        outputs = np.einsum('...ig,igo->...go', x, kernel)  # type: ignore
        outputs = outputs.reshape(*outputs.shape[:-2], -1) + bias  # type: ignore
        # outputs.shape = (batch, out_spa..., ch_out) in channels_last

        if layer.data_format == 'channels_first':
            outputs = np.moveaxis(outputs, -1, 1)  # type: ignore

        return outputs  # type: ignore
