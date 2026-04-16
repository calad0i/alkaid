from math import prod

import keras
import numpy as np
from keras.src.layers.pooling.base_global_pooling import BaseGlobalPooling
from keras.src.layers.pooling.base_pooling import BasePooling

from alkaid.trace import FVArray

from ._base import ReplayOperationBase
from .conv import symbolic_extract_patches


class ReplayPool(ReplayOperationBase):
    handles = (
        keras.layers.AveragePooling1D,
        keras.layers.AveragePooling2D,
        keras.layers.AveragePooling3D,
        keras.layers.MaxPooling1D,
        keras.layers.MaxPooling2D,
        keras.layers.MaxPooling3D,
        keras.layers.GlobalAveragePooling1D,
        keras.layers.GlobalMaxPooling1D,
        keras.layers.GlobalAveragePooling2D,
        keras.layers.GlobalMaxPooling2D,
        keras.layers.GlobalAveragePooling3D,
        keras.layers.GlobalMaxPooling3D,
    )

    def call(self, inputs: FVArray, mask: None = None) -> FVArray:
        assert mask is None, 'Masking is not supported in pooling layers.'
        cname = self.op.__class__.__name__
        if 'Max' in cname:
            op = 'max'
        else:
            assert 'Average' in cname, f'Unsupported global pooling layer: {cname}'
            op = 'avg'

        # Strip leading batch dim (FVArray always has batch=1 from get_input_shapes)
        has_batch = inputs.shape[0] == 1 and inputs.ndim > 2
        if has_batch:
            inputs = inputs[0]

        data_format = self.op.data_format
        if data_format == 'channels_first':
            inputs = np.moveaxis(inputs, 0, -1)  # type: ignore

        if isinstance(self.op, BaseGlobalPooling):
            # Reduce all spatial dims (everything except last = channels)
            axis = tuple(range(inputs.ndim - 1))
            keepdims = self.op.keepdims

            if op == 'max':
                out = np.amax(inputs, axis=axis, keepdims=keepdims)  # type: ignore
            elif op == 'avg':
                pool_size = prod(inputs.shape[:-1])
                out = np.sum(inputs, axis=axis, keepdims=keepdims) / pool_size  # type: ignore
        else:
            assert isinstance(self.op, BasePooling), f'Unknown pooling layer: {type(self.op)}'
            pool_size = self.op.pool_size
            strides = self.op.strides
            padding = self.op.padding
            ch = inputs.shape[-1]
            x = symbolic_extract_patches(
                inputs,
                pool_size,
                strides,
                dilation_rate=1,
                padding=padding,
                data_format='channels_last',
            )
            x = x.reshape(x.shape[:-1] + (-1, ch))

            if padding == 'same':
                _mask = symbolic_extract_patches(
                    np.ones(inputs.shape, dtype=np.int32),
                    pool_size,
                    strides,
                    dilation_rate=1,
                    padding=padding,
                    data_format='channels_last',
                ).reshape(x.shape)
            elif padding == 'valid':
                _mask = np.ones(x.shape, dtype=np.int32)
            else:
                raise ValueError(f'Unknown padding type: {padding}')

            if op == 'max':
                _vars = np.where(_mask, x, -2147483648)  # type: ignore
                x = FVArray(_vars, x.solver_options)
                out = np.max(x, axis=-2)  # type: ignore
            elif op == 'avg':
                out = np.sum(x, axis=-2) / np.sum(_mask, axis=-2)  # type: ignore
            else:
                raise ValueError(f'Unknown pooling operation: {op}')

        if data_format == 'channels_first':
            out = np.moveaxis(out, -1, 0)  # type: ignore

        # Re-add batch dim
        if has_batch:
            out = out[None]  # type: ignore

        return out  # type: ignore
