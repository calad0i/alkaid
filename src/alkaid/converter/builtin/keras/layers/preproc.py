"""Preprocessing-style keras layers: Rescaling, Normalization, Permute, CategoryEncoding, Embedding."""

import keras
import numpy as np

from alkaid.trace import FVArray

from ._base import ReplayOperationBase


class ReplayRescaling(ReplayOperationBase):
    handles = (keras.layers.Rescaling,)

    def call(self, inputs: FVArray) -> FVArray:
        return inputs * self._load_weight('scale') + self._load_weight('offset')  # type: ignore


class ReplayNormalization(ReplayOperationBase):
    handles = (keras.layers.Normalization,)

    def fused_scale_offset(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the fused ``(scale, offset)`` such that ``output = input * scale + offset``.

        Keras' Normalization uses ``std = max(sqrt(variance), backend.epsilon())``
        and computes:
          * ``invert=False``: ``(x - mean) / std``
          * ``invert=True``:  ``mean + x * std``

        Both cases are reshaped here as ``x * scale + offset``.
        """
        layer = self.op
        mean = self._load_weight('mean').astype(np.float32)
        variance = self._load_weight('variance').astype(np.float32)
        eps = keras.backend.epsilon()
        std = np.maximum(np.sqrt(variance), eps)
        if layer.invert:
            return std, mean
        scale = np.float32(1.0) / std
        offset = -mean * scale
        return scale, offset

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        scale, offset = self.fused_scale_offset()
        axis = layer.axis if isinstance(layer.axis, (tuple, list)) else (layer.axis,)
        shape = [1] * inputs.ndim
        for ax in axis:
            real_ax = ax if ax >= 0 else inputs.ndim + ax
            shape[real_ax] = inputs.shape[real_ax]
        return inputs * np.asarray(scale).reshape(shape) + np.asarray(offset).reshape(shape)  # type: ignore


class ReplayPermute(ReplayOperationBase):
    handles = (keras.layers.Permute,)

    def call(self, inputs: FVArray) -> FVArray:
        # keras.Permute.dims excludes the batch dim; rebase to include it.
        axes = (0,) + tuple(d for d in self.op.dims)
        return np.transpose(inputs, axes)  # type: ignore


class ReplayEmbedding(ReplayOperationBase):
    handles = (keras.layers.Embedding,)

    def call(self, inputs: FVArray) -> FVArray:
        weight = self._load_weight('embeddings')
        D = weight.shape[1]
        cols: list[FVArray] = []
        for d in range(D):
            col_w = weight[:, d]

            def _lut(x, _w=col_w):
                idx = np.asarray(x).astype(np.int64)
                idx = np.clip(idx, 0, _w.shape[0] - 1)
                return _w[idx]

            col = inputs.apply(_lut).quantize()
            cols.append(col)
        return np.stack(cols, axis=-1)  # type: ignore


class ReplayCategoryEncoding(ReplayOperationBase):
    handles = (keras.layers.CategoryEncoding,)

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        mode = layer.output_mode
        V = int(layer.num_tokens)
        assert mode in ('one_hot', 'multi_hot', 'count'), f'unsupported output_mode: {mode}'
        if mode == 'one_hot':
            cols = [inputs == c for c in range(V)]
            return np.stack(cols, axis=-1)  # type: ignore
        cols = [inputs == c for c in range(V)]
        stacked = np.stack(cols, axis=-1)
        summed = np.sum(stacked, axis=-2)
        if mode == 'multi_hot':
            summed = (summed > 0).astype(inputs.dtype) if hasattr(inputs, 'dtype') else summed
        return summed  # type: ignore
