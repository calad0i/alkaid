"""Framework-neutral sliding-window patch extraction.

Pure numpy, works on both ``np.ndarray`` and ``alkaid.trace.FVArray``. Used by
both the keras and torch converter wrappers.

``extract_patches`` and ``extract_patches_transposed`` share everything except
the per-dim index formula and the output spatial size; the common tail is in
``_gather_patches``.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from ..fixed_variable_array import FVArray

T = TypeVar('T', 'FVArray', np.ndarray)

Padding = int | str | Sequence[int] | Sequence[tuple[int, int]]


def _as_tuple(v: int | Sequence[int], rank: int) -> tuple[int, ...]:
    if isinstance(v, int):
        return (v,) * rank
    v = tuple(v)
    assert len(v) == rank, f'expected rank {rank}, got {v}'
    return v


def _resolve_pad_per_dim(
    padding: Padding,
    in_sizes: tuple[int, ...],
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Translate any of the accepted ``padding`` forms into per-dim ``(before, after)`` pads."""
    rank = len(in_sizes)
    if isinstance(padding, str):
        if padding == 'valid':
            return ((0, 0),) * rank
        if padding == 'same':
            # tf / keras "same" semantics: out_size = ceil(in_size / stride).
            out = []
            for n, k, s, d in zip(in_sizes, sizes, strides, dilations):
                out_dim = (n + s - 1) // s
                total = max(0, (out_dim - 1) * s + (k - 1) * d + 1 - n)
                out.append((total // 2, total - total // 2))
            return tuple(out)
        raise ValueError(f'unknown padding mode: {padding!r}')
    if isinstance(padding, int):
        return ((padding, padding),) * rank
    return tuple((int(p), int(p)) if isinstance(p, int) else (int(p[0]), int(p[1])) for p in padding)


def _gather_patches(
    images: T,
    spa_in: tuple[int, ...],
    out_spa: tuple[int, ...],
    size_t: tuple[int, ...],
    idx_1d_per_dim: list[np.ndarray],
    mask_1d_per_dim: list[np.ndarray],
    pad_value: float,
) -> T:
    """Common back-half: broadcast per-dim indices, gather, mask, reshape.

    ``idx_1d_per_dim[i]`` and ``mask_1d_per_dim[i]`` are both ``(out_spa[i], size_t[i])``.
    """
    rank = len(spa_in)
    idx_per_dim: list[np.ndarray] = []
    mask_per_dim: list[np.ndarray] = []
    for i in range(rank):
        shape = [1] * (2 * rank)
        shape[i] = out_spa[i]
        shape[rank + i] = size_t[i]
        idx_per_dim.append(idx_1d_per_dim[i].reshape(shape))
        mask_per_dim.append(mask_1d_per_dim[i].reshape(shape))

    mask: np.ndarray = mask_per_dim[0]
    for m in mask_per_dim[1:]:
        mask = mask & m

    linear_idx = np.zeros_like(idx_per_dim[0])
    stride_acc = 1
    for i in reversed(range(rank)):
        linear_idx = linear_idx + idx_per_dim[i] * stride_acc
        stride_acc *= spa_in[i]

    batch = images.shape[0]
    ch = images.shape[-1]
    flat = images.reshape((batch, int(np.prod(spa_in)), ch))
    gathered = np.take(flat, linear_idx, axis=1)  # (batch, *out_spa, *size, ch)

    from ..fixed_variable_array import FVArray

    mask_b: np.ndarray = np.broadcast_to(mask[..., None], gathered.shape[1:])
    if isinstance(gathered, FVArray):
        _vars = np.asarray(gathered)
        _vars = np.where(mask_b, _vars, pad_value)
        gathered = FVArray(_vars, gathered.solver_options)
    else:
        gathered = np.where(mask_b, gathered, pad_value)  # type: ignore

    return gathered.reshape((batch,) + out_spa + (-1,))  # type: ignore


def extract_patches(
    images: T,
    size: int | Sequence[int],
    strides: int | Sequence[int] | None = 1,
    dilation_rate: int | Sequence[int] = 1,
    padding: Padding = 'valid',
    data_format: str = 'channels_last',
    pad_value: float = 0,
) -> T:
    """Forward sliding-window patch extraction.

    For each output position ``o`` and kernel index ``k``, gathers
    ``images[..., o*stride + k*dilation - pad_before, ...]``, masking positions
    that fall outside the original spatial range.

    Parameters
    ----------
    images
        Array of shape ``(batch, *spatial, ch)`` for ``channels_last`` or
        ``(batch, ch, *spatial)`` for ``channels_first``.
    size
        Per-spatial-dim kernel size. Scalar broadcasts to all spatial dims.
    strides
        Per-spatial-dim stride. ``None`` defaults to ``1``.
    dilation_rate
        Per-spatial-dim dilation.
    padding
        One of: ``'valid'``, ``'same'`` (keras ``ceil(in/stride)`` semantics),
        a single ``int`` (symmetric on every dim), a sequence of ints (symmetric
        per dim), or a sequence of ``(before, after)`` pairs.
    data_format
        ``'channels_last'`` or ``'channels_first'``. Output is always
        channels-last form: ``(batch, *out_spatial, prod(size) * ch)``.
    pad_value
        Fill value for out-of-bounds positions.
    """
    if data_format == 'channels_first':
        images = np.moveaxis(images, 1, -1)  # type: ignore
    elif data_format != 'channels_last':
        raise ValueError(f'unknown data_format: {data_format!r}')

    rank = images.ndim - 2
    size_t = _as_tuple(size, rank)
    stride_t = _as_tuple(strides if strides is not None else 1, rank)
    dilation_t = _as_tuple(dilation_rate, rank)
    spa_in = tuple(images.shape[1:-1])
    pads = _resolve_pad_per_dim(padding, spa_in, size_t, stride_t, dilation_t)

    out_spa = tuple(
        (spa_in[i] + pads[i][0] + pads[i][1] - (size_t[i] - 1) * dilation_t[i] - 1) // stride_t[i] + 1 for i in range(rank)
    )

    idx_1d_per_dim: list[np.ndarray] = []
    mask_1d_per_dim: list[np.ndarray] = []
    for i in range(rank):
        o = np.arange(out_spa[i])[:, None] * stride_t[i]
        k = np.arange(size_t[i])[None, :] * dilation_t[i]
        idx = o + k - pads[i][0]
        mask = (idx >= 0) & (idx < spa_in[i])
        idx_1d_per_dim.append(np.clip(idx, 0, spa_in[i] - 1))
        mask_1d_per_dim.append(mask)

    return _gather_patches(images, spa_in, out_spa, size_t, idx_1d_per_dim, mask_1d_per_dim, pad_value)


def extract_patches_transposed(
    images: T,
    size: int | Sequence[int],
    strides: int | Sequence[int] = 1,
    dilation_rate: int | Sequence[int] = 1,
    padding: Padding = 0,
    output_padding: int | Sequence[int] = 0,
    data_format: str = 'channels_last',
    pad_value: float = 0,
) -> T:
    """Gather-based patch extraction for transposed convolution.

    Output spatial size per dim:
        ``(in - 1) * stride - pad_before - pad_after + dilation * (size - 1) + output_padding + 1``.
    """
    if data_format == 'channels_first':
        images = np.moveaxis(images, 1, -1)  # type: ignore
    elif data_format != 'channels_last':
        raise ValueError(f'unknown data_format: {data_format!r}')

    rank = images.ndim - 2
    size_t = _as_tuple(size, rank)
    stride_t = _as_tuple(strides, rank)
    dilation_t = _as_tuple(dilation_rate, rank)
    output_pad_t = _as_tuple(output_padding, rank)
    spa_in = tuple(images.shape[1:-1])
    pads = _resolve_pad_per_dim(padding, spa_in, size_t, stride_t, dilation_t)

    out_spa = tuple(
        (spa_in[i] - 1) * stride_t[i] - pads[i][0] - pads[i][1] + dilation_t[i] * (size_t[i] - 1) + output_pad_t[i] + 1
        for i in range(rank)
    )

    idx_1d_per_dim: list[np.ndarray] = []
    mask_1d_per_dim: list[np.ndarray] = []
    for i in range(rank):
        o = np.arange(out_spa[i])[:, None] + pads[i][0]
        k = np.arange(size_t[i])[None, :] * dilation_t[i]
        numer = o - k
        l_in = numer // stride_t[i]
        mask = (numer % stride_t[i] == 0) & (l_in >= 0) & (l_in < spa_in[i])
        idx_1d_per_dim.append(np.where(mask, l_in, 0))
        mask_1d_per_dim.append(mask)

    return _gather_patches(images, spa_in, out_spa, size_t, idx_1d_per_dim, mask_1d_per_dim, pad_value)
