"""Pure-geometric image layers: padding, cropping, upsampling."""

import keras
import numpy as np

from alkaid.trace import FVArray

from ._base import ReplayOperationBase


def _spa_axes(rank: int, data_format: str) -> tuple[int, ...]:
    return tuple(range(1, 1 + rank)) if data_format == 'channels_last' else tuple(range(2, 2 + rank))


def _normalize_pad(p, rank: int) -> tuple[tuple[int, int], ...]:
    if isinstance(p, int):
        return ((p, p),) * rank
    if isinstance(p, tuple) and len(p) == 2 and all(isinstance(v, int) for v in p):
        # 1D case: (before, after)
        return (tuple(p),)  # type: ignore
    return tuple((int(a[0]), int(a[1])) for a in p)  # type: ignore


class ReplayZeroPadding(ReplayOperationBase):
    handles = (
        keras.layers.ZeroPadding1D,
        keras.layers.ZeroPadding2D,
        keras.layers.ZeroPadding3D,
    )

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        rank = inputs.ndim - 2
        data_format = getattr(layer, 'data_format', 'channels_last') or 'channels_last'
        spa = _spa_axes(rank, data_format)
        pads = _normalize_pad(layer.padding, rank)
        np_pad = [(0, 0)] * inputs.ndim
        for i, ax in enumerate(spa):
            np_pad[ax] = pads[i]
        return np.pad(inputs, np_pad, mode='constant', constant_values=0)  # type: ignore


class ReplayCropping(ReplayOperationBase):
    handles = (
        keras.layers.Cropping1D,
        keras.layers.Cropping2D,
        keras.layers.Cropping3D,
    )

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        rank = inputs.ndim - 2
        data_format = getattr(layer, 'data_format', 'channels_last') or 'channels_last'
        spa = _spa_axes(rank, data_format)
        crops = _normalize_pad(layer.cropping, rank)
        slicer: list[slice] = [slice(None)] * inputs.ndim
        for i, ax in enumerate(spa):
            before, after = crops[i]
            size = inputs.shape[ax]
            slicer[ax] = slice(before, size - after if after > 0 else size)
        return inputs[tuple(slicer)]  # type: ignore


class ReplayUpSampling(ReplayOperationBase):
    handles = (
        keras.layers.UpSampling1D,
        keras.layers.UpSampling2D,
        keras.layers.UpSampling3D,
    )

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        interp = getattr(layer, 'interpolation', 'nearest')
        assert interp == 'nearest', f'only nearest-neighbour upsampling is bit-exact; got {interp!r}'
        rank = inputs.ndim - 2
        size = layer.size
        if isinstance(size, int):
            size = (size,) * rank
        data_format = getattr(layer, 'data_format', 'channels_last') or 'channels_last'
        spa = _spa_axes(rank, data_format)
        out = inputs
        for i, s in enumerate(size):
            out = np.repeat(out, s, axis=spa[i])
        return out  # type: ignore
