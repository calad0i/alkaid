from math import prod

import numpy as np
from keras.layers import (
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
    DepthwiseConv1D,
    DepthwiseConv2D,
)
from keras.src.ops.image import ExtractPatches

from alkaid.trace import FVArray
from alkaid.trace.ops import extract_patches, extract_patches_transposed

from ._base import ReplayOperationBase


def _conv(
    patches: FVArray,  # (batch, *out_spa, K_vol * C_in)
    kernel: np.ndarray,  # keras layout: (*K, C_in/groups, C_out)
    *,
    k_vol: int,
    groups: int,
    ch_in_per_g: int,
    out_per_g: int,
) -> FVArray:
    """conv operation on patches.

    Assumes ``patches`` layout is ``(batch, *out_spa, K_vol * groups * C_in_per_g)``
    with kernel-position outermost, groups next, ``C_in_per_g`` innermost — the layout
    produced by :func:`extract_patches` when input channels are split contiguously
    across groups.
    """
    x = patches.reshape(*patches.shape[:-1], k_vol, groups, ch_in_per_g)
    # keras kernel ``(*K, C_in/g, C_out)`` → ``(K_vol, C_in_per_g, groups, out_per_g)``
    # (keras stores C_out blocked by group: [g=0 outputs, g=1 outputs, ...]).
    w = kernel.reshape(k_vol, ch_in_per_g, groups, out_per_g)
    w = np.transpose(w, (0, 2, 1, 3))  # (K_vol, groups, C_in_per_g, out_per_g)
    out: FVArray = np.einsum('...kgc,kgco->...go', x, w)
    return out.reshape(*out.shape[:-2], -1)


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
    """Unified handler for regular and depthwise convolution.

    Depthwise kernel: ``(*K, C_in, dm) -> (*K, 1, C_in * dm)`` makes just a conv w/ groups.
    """

    handles = (Conv1D, Conv2D, Conv3D, DepthwiseConv1D, DepthwiseConv2D)

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        kernel = self._load_weight('kernel')
        bias = self._load_weight('bias')

        if isinstance(layer, (DepthwiseConv1D, DepthwiseConv2D)):
            ch_in, dm = kernel.shape[-2:]
            kernel = kernel.reshape(*kernel.shape[:-2], 1, ch_in * dm)
            groups = ch_in
        else:
            groups = layer.groups

        x = extract_patches(
            inputs,
            size=layer.kernel_size,
            strides=layer.strides,
            dilation_rate=layer.dilation_rate,
            padding=layer.padding,
            data_format=layer.data_format,
        )

        ch_out = kernel.shape[-1]
        ch_in_per_g = kernel.shape[-2]
        k_vol = int(prod(layer.kernel_size))
        out = _conv(
            x,
            kernel,
            k_vol=k_vol,
            groups=groups,
            ch_in_per_g=ch_in_per_g,
            out_per_g=ch_out // groups,
        )
        if bias.shape != ():
            out = out + bias
        if layer.data_format == 'channels_first':
            out = np.moveaxis(out, -1, 1)  # type: ignore
        return out  # type: ignore


class ReplayConvTranspose(ReplayOperationBase):
    handles = (Conv1DTranspose, Conv2DTranspose, Conv3DTranspose)

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        kernel = self._load_weight('kernel')  # (*K, filters, in_channels)
        bias = self._load_weight('bias')

        rank = len(layer.kernel_size)
        strides = layer.strides
        dilation = layer.dilation_rate
        k_eff = tuple((layer.kernel_size[i] - 1) * dilation[i] + 1 for i in range(rank))
        output_padding = layer.output_padding
        if output_padding is None:
            output_padding = (0,) * rank
        if layer.padding == 'same':
            pads = tuple(
                (max(0, k_eff[i] - strides[i]) // 2, max(0, k_eff[i] - strides[i]) - max(0, k_eff[i] - strides[i]) // 2)
                for i in range(rank)
            )
        elif layer.padding == 'valid':
            pads = ((0, 0),) * rank
        else:
            raise ValueError(f'unsupported padding: {layer.padding!r}')

        x = extract_patches_transposed(
            inputs,
            size=layer.kernel_size,
            strides=strides,
            dilation_rate=dilation,
            padding=pads,
            output_padding=output_padding,
            data_format=layer.data_format,
            pad_value=0,
        )
        # x: (batch, *out_spa, K_vol * C_in). ConvTranspose is ungrouped in keras.
        k_vol = int(prod(layer.kernel_size))
        # Kernel layout: (*K, C_out, C_in). Reshape+transpose to match the shared einsum's
        # expectation of (*K, C_in, C_out).
        kernel = kernel.reshape(k_vol, layer.filters, -1)
        kernel = np.transpose(kernel, (0, 2, 1))  # (K_vol, C_in, C_out)
        x = x.reshape(*x.shape[:-1], k_vol, -1)
        out = np.einsum('...kc,kco->...o', x, kernel)
        if bias.shape != ():
            out = out + bias
        if layer.data_format == 'channels_first':
            out = np.moveaxis(out, -1, 1)  # type: ignore
        return out  # type: ignore
