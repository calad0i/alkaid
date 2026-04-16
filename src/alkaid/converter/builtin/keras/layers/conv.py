from typing import TypeVar

import numpy as np
from keras import ops
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.src.ops.image import ExtractPatches, extract_patches_3d

from alkaid.trace import FVArray

from ._base import ReplayOperationBase

T = TypeVar('T', FVArray, np.ndarray)


def symbolic_extract_patches_3d(
    images: T,
    size: tuple[int, int, int],
    strides: tuple[int, int, int],
    dilation_rate: tuple[int, int, int],
    padding: str,
    data_format: str,
    pad_value: float = 0,
) -> T:
    img_tensor = ops.cast(ops.reshape(ops.arange(images.size), images.shape), dtype='float32')
    img_tensor = -img_tensor - 1
    out_tensor = extract_patches_3d(
        img_tensor[None],
        size=size,
        strides=strides,
        dilation_rate=dilation_rate,  # type: ignore
        padding=padding,
        data_format=data_format,
    )[0]
    out_index: np.ndarray = ops.convert_to_numpy(out_tensor).round().astype(np.int32)  # type: ignore
    mask = out_index == 0
    out_index = np.where(mask, 0, -out_index - 1)
    images = images.ravel()[out_index]  # type: ignore

    if isinstance(images, FVArray):
        _vars = np.asarray(images)
        _vars = np.where(mask, pad_value, _vars)
        images = FVArray(_vars, images.solver_options)  # type: ignore
    else:
        images = np.where(mask, pad_value, images)

    return images


def symbolic_extract_patches(
    images: T,
    size: tuple[int, ...] | int,
    strides: tuple[int, ...] | int,
    dilation_rate: tuple[int, ...] | int,
    padding: str,
    data_format: str,
    pad_value: float = 0,
) -> T:
    rank = images.ndim - 1
    size = (size,) * rank if isinstance(size, int) else size
    strides = (strides,) * rank if isinstance(strides, int) else strides
    dilation_rate = (dilation_rate,) * rank if isinstance(dilation_rate, int) else dilation_rate

    assert rank == len(size) == len(strides) == len(dilation_rate), (
        f'Invalid rank {rank} for size {size}, strides {strides}, dilation_rate {dilation_rate}'
    )

    pad_rank = 3 - rank
    _size: tuple[int, int, int] = (1,) * pad_rank + size  # type: ignore
    _strides: tuple[int, int, int] = (1,) * pad_rank + strides  # type: ignore
    _dilation_rate: tuple[int, int, int] = (1,) * pad_rank + dilation_rate  # type: ignore

    _pad = (1,) * pad_rank
    if data_format == 'channels_first':
        images = np.moveaxis(images, 0, -1)  # type: ignore

    *spa, ch = images.shape
    images = images.reshape(*_pad, *spa, ch)  # type: ignore

    r = symbolic_extract_patches_3d(
        images,
        size=_size,
        strides=_strides,
        dilation_rate=_dilation_rate,
        padding=padding,
        data_format='channels_last',
        pad_value=pad_value,
    )

    return r.reshape(r.shape[pad_rank:])


class ReplayExtractPatches(ReplayOperationBase):
    handles = (ExtractPatches,)

    def call(self, images: FVArray) -> FVArray:
        op: ExtractPatches = self.op
        pixel_shape = op.size
        strides = op.strides
        dilation_rate: int | tuple[int, int] = op.dilation_rate
        padding = op.padding
        data_format = op.data_format

        if strides is None:
            strides = 1

        return symbolic_extract_patches(images, pixel_shape, strides, dilation_rate, padding, data_format)


class ReplayConv(ReplayOperationBase):
    handles = (Conv1D, Conv2D, Conv3D)

    def call(self, inputs: FVArray) -> FVArray:
        layer: Conv1D | Conv2D | Conv3D = self.op
        kernel = self._load_weight('kernel')
        bias = self._load_weight('bias')
        strides = layer.strides
        padding = layer.padding
        dilation_rate = layer.dilation_rate
        groups = layer.groups

        # Strip leading batch dim (FVArray always has batch=1 from get_input_shapes)
        has_batch = inputs.shape[0] == 1 and inputs.ndim > len(layer.kernel_size) + 1
        if has_batch:
            inputs = inputs[0]

        x = symbolic_extract_patches(
            inputs,
            size=layer.kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            data_format=layer.data_format,
        )

        if layer.data_format == 'channels_first':
            inputs = np.moveaxis(inputs, 0, -1)  # type: ignore

        ch_out = kernel.shape[-1]

        _ch_out = ch_out // groups

        x = x.reshape(*x.shape[:-1], -1, groups)  # type: ignore
        kernel = kernel.reshape(-1, groups, _ch_out)

        outputs = np.einsum('...ig,igo->...go', x, kernel)  # type: ignore
        outputs = outputs.reshape(*outputs.shape[:-2], -1) + bias  # type: ignore

        if layer.data_format == 'channels_first':
            outputs: FVArray = np.moveaxis(outputs, -1, 0)  # type: ignore

        # Re-add batch dim
        if has_batch:
            outputs = outputs[None]  # type: ignore

        return outputs
