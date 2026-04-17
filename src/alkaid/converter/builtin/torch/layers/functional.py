import operator
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import torch
from torch import no_grad
from torch.nn import functional as F

from alkaid.trace import FVariable, FVArray
from alkaid.trace.ops import einsum as _einsum
from alkaid.trace.ops import extract_patches, extract_patches_transposed


def to_np_arr(x: Any) -> np.ndarray:
    """Convert a torch tensor (or anything numpy-coercible) to a numpy array."""
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            x = x.detach()
        return x.cpu().numpy()
    return np.asarray(x)


_functional_map: dict[Callable, Callable] = {}


def _makesure_ret_arr(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        r = fn(*args, **kwargs)
        if isinstance(r, FVariable):
            return FVArray(np.array(r))
        return r

    return wrapper


_IGNORED_KWARGS = ('inplace', 'out', 'memory_format')


def _strip_ignored_kwargs(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        for k in _IGNORED_KWARGS:
            kwargs.pop(k, None)
        return fn(*args, **kwargs)

    return wrapper


def _functional(*torch_func: Callable):
    def decorator(np_func: Callable):
        np_func_wrap = wraps(np_func)(no_grad()(_strip_ignored_kwargs(_makesure_ret_arr(np_func))))
        for _torch_func in torch_func:
            _functional_map[_torch_func] = np_func_wrap
        return np_func_wrap

    return decorator


# ---------------------------------------------------------------------------
# Helpers shared with modules.py
# ---------------------------------------------------------------------------


def conv_nd_replay(
    input: FVArray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    stride: int | tuple[int, ...] = 1,
    padding: int | str | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> FVArray:
    """Channels-first conv replay; matches torch.nn.functional.conv{1,2,3}d semantics."""
    x = np.moveaxis(input, 1, -1)  # NCHW -> NHWC
    # weight shape: (Cout, Cin/g, *K)
    ch_out = weight.shape[0]
    ksize = weight.shape[2:]
    k_vol = int(np.prod(ksize))
    ch_in_per_g = weight.shape[1]
    out_per_g = ch_out // groups

    x = extract_patches(x, ksize, stride, dilation, padding, pad_value=0)
    # x: (batch, *out_spa, K_vol * Cin) where Cin = groups * ch_in_per_g
    # Within the last dim: kernel position varies slowest, then channels. Within channels,
    # groups varies slower than ch_in_per_g. Reshape accordingly.
    x = x.reshape(*x.shape[:-1], k_vol, groups, ch_in_per_g)

    # weight: (groups * out_per_g, ch_in_per_g, *K)
    # → (groups, out_per_g, ch_in_per_g, K_vol) → (K_vol, groups, ch_in_per_g, out_per_g)
    w = weight.reshape(groups, out_per_g, ch_in_per_g, k_vol)
    w = np.transpose(w, (3, 0, 2, 1))

    out = np.einsum('...kgc,kgco->...go', x, w)
    # out: (batch, *out_spa, groups, out_per_g)
    out = out.reshape(*out.shape[:-2], -1)
    if bias is not None and bias.shape != ():
        out = out + bias
    out = np.moveaxis(out, -1, 1)  # NHWC -> NCHW
    return out  # type: ignore


def batch_norm_replay(
    input: FVArray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> FVArray:
    assert not training, 'training-mode batch norm is not supported'
    gamma = weight if weight is not None else np.ones_like(running_mean)
    beta = bias if bias is not None else np.zeros_like(running_mean)
    scale = gamma / np.sqrt(running_var + eps)
    offset = beta - running_mean * scale
    shape = [1] * input.ndim
    shape[1] = input.shape[1]  # channel dim
    return input * scale.reshape(shape) + offset.reshape(shape)  # type: ignore


def pool_nd_replay(
    input: FVArray,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    mode: str = 'max',
) -> FVArray:
    if stride is None:
        stride = kernel_size
    ksize = kernel_size
    x = np.moveaxis(input, 1, -1)  # NCHW -> NHWC
    ch = x.shape[-1]
    patches = extract_patches(x, ksize, stride, dilation, padding, pad_value=0)
    patches = patches.reshape(patches.shape[:-1] + (-1, ch))  # (batch, *out, K_vol, C)

    if mode == 'max':
        if isinstance(padding, int):
            padding_t = (padding,) * (input.ndim - 2)
        else:
            padding_t = tuple(padding)
        has_pad = any(p != 0 for p in padding_t)
        if has_pad:
            mask = extract_patches(
                np.ones(x.shape, dtype=np.int32),
                ksize,
                stride,
                dilation,
                padding,
                pad_value=0,
            ).reshape(patches.shape)
            _vars = np.where(mask, np.asarray(patches), -2147483648)
            patches = FVArray(_vars, patches.solver_options)
        out = np.max(patches, axis=-2)
    elif mode == 'avg':
        if isinstance(padding, int):
            padding_t = (padding,) * (input.ndim - 2)
        else:
            padding_t = tuple(padding)
        has_pad = any(p != 0 for p in padding_t)
        if has_pad:
            mask = extract_patches(
                np.ones(x.shape, dtype=np.int32),
                ksize,
                stride,
                dilation,
                padding,
                pad_value=0,
            ).reshape(patches.shape)
            out = np.sum(patches, axis=-2) / np.sum(mask, axis=-2)
        else:
            out = np.mean(patches, axis=-2)
    else:
        raise ValueError(f'unknown pool mode: {mode}')
    return np.moveaxis(out, -1, 1)  # type: ignore


def conv_transpose_nd_replay(
    input: FVArray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
    dilation: int | tuple[int, ...] = 1,
) -> FVArray:
    """Channels-first transposed convolution replay."""
    x = np.moveaxis(input, 1, -1)  # NCHW -> NHWC
    # weight shape: (C_in, C_out/g, *K)
    ksize = weight.shape[2:]
    k_vol = int(np.prod(ksize))
    ch_in = x.shape[-1]
    ch_in_per_g = ch_in // groups
    out_per_g = weight.shape[1]

    patches = extract_patches_transposed(
        x,
        size=ksize,
        strides=stride,
        dilation_rate=dilation,
        padding=padding,
        output_padding=output_padding,
        pad_value=0,
    )
    # patches: (batch, *out_spa, K_vol * ch_in) with kernel outermost, groups inside channels
    patches = patches.reshape(*patches.shape[:-1], k_vol, groups, ch_in_per_g)

    # weight torch layout: (groups * ch_in_per_g, C_out/g, *K)
    # -> (groups, ch_in_per_g, C_out/g, K_vol) -> (K_vol, groups, ch_in_per_g, C_out/g)
    w = weight.reshape(groups, ch_in_per_g, out_per_g, k_vol)
    w = np.transpose(w, (3, 0, 1, 2))

    out = np.einsum('...kgc,kgco->...go', patches, w)
    out = out.reshape(*out.shape[:-2], -1)
    if bias is not None and bias.shape != ():
        out = out + bias
    out = np.moveaxis(out, -1, 1)
    return out  # type: ignore


def upsample_nearest_replay(input: FVArray, scale: int | tuple[int, ...]) -> FVArray:
    """Channels-first nearest-neighbour upsampling via np.repeat along each spatial dim."""
    rank = input.ndim - 2
    if isinstance(scale, int):
        scale = (scale,) * rank
    out = input
    for i, s in enumerate(scale):
        out = np.repeat(out, s, axis=2 + i)
    return out  # type: ignore


def pixel_shuffle_replay(input: FVArray, upscale_factor: int) -> FVArray:
    """Channels-first: (N, C*r^2, H, W) -> (N, C, H*r, W*r) via reshape+transpose."""
    r = upscale_factor
    shape = input.shape
    assert len(shape) >= 3, 'pixel_shuffle needs at least a rank-3 tensor'
    rank = len(shape) - 2
    batch = shape[0]
    c_in = shape[1]
    spa = shape[2:]
    assert c_in % (r**rank) == 0, f'channels {c_in} not divisible by r^{rank}={r**rank}'
    c_out = c_in // (r**rank)
    # (N, C_out, r, r, ..., r, H, W, ..., L)
    x = input.reshape((batch, c_out) + (r,) * rank + spa)
    # Interleave r factors into spatial dims: (N, C_out, H, r, W, r, ..., L, r)
    perm = [0, 1]
    for i in range(rank):
        perm.append(2 + rank + i)  # spatial_i
        perm.append(2 + i)  # r_i
    x = np.transpose(x, tuple(perm))
    new_spa = tuple(spa[i] * r for i in range(rank))
    return x.reshape((batch, c_out) + new_spa)  # type: ignore


def pixel_unshuffle_replay(input: FVArray, downscale_factor: int) -> FVArray:
    r = downscale_factor
    shape = input.shape
    assert len(shape) >= 3, 'pixel_unshuffle needs at least a rank-3 tensor'
    rank = len(shape) - 2
    batch = shape[0]
    c_in = shape[1]
    spa = shape[2:]
    assert all(s % r == 0 for s in spa), f'spatial dims {spa} not divisible by {r}'
    new_spa = tuple(s // r for s in spa)
    # (N, C_in, out_0, r, out_1, r, ..., out_{r-1}, r)
    split_shape = [batch, c_in]
    for i in range(rank):
        split_shape.extend([new_spa[i], r])
    x = input.reshape(tuple(split_shape))
    # Permute -> (N, C_in, r, r, ..., r, out_0, out_1, ..., out_{r-1})
    perm = [0, 1]
    for i in range(rank):
        perm.append(3 + 2 * i)  # r_i
    for i in range(rank):
        perm.append(2 + 2 * i)  # out_i
    x = np.transpose(x, tuple(perm))
    return x.reshape((batch, c_in * (r**rank)) + new_spa)  # type: ignore


def embedding_replay(indices: FVArray, weight: np.ndarray, padding_idx: int | None = None) -> FVArray:
    """Scalar-LUT per embed-dim column; stacks D columns into a new trailing axis."""
    assert isinstance(indices, FVArray), 'embedding indices must be FVArray'
    assert weight.ndim == 2, f'weight must be (V, D), got shape {weight.shape}'
    if padding_idx is not None:
        weight = weight.copy()
        weight[padding_idx] = 0
    D = weight.shape[1]
    cols: list[FVArray] = []
    for d in range(D):
        col_w = weight[:, d]

        def _lut(x, _w=col_w):
            idx = np.asarray(x).astype(np.int64)
            idx = np.clip(idx, 0, _w.shape[0] - 1)
            return _w[idx]

        col = indices.apply(_lut).quantize()
        cols.append(col)
    return np.stack(cols, axis=-1)  # type: ignore


def adaptive_pool_replay(input: FVArray, output_size, mode: str) -> FVArray:
    rank = input.ndim - 2
    if isinstance(output_size, int):
        output_size = (output_size,) * rank
    output_size = tuple(output_size)
    assert all(s == 1 for s in output_size), 'only adaptive pooling with output_size==1 is supported'
    axes = tuple(range(2, input.ndim))
    if mode == 'max':
        return np.amax(input, axis=axes, keepdims=True)  # type: ignore
    elif mode == 'avg':
        return np.mean(input, axis=axes, keepdims=True)  # type: ignore
    raise ValueError(f'unknown pool mode: {mode}')


# ---------------------------------------------------------------------------
# torch.nn.functional handlers
# ---------------------------------------------------------------------------


@_functional(F.linear)
def replay_linear(input: FVArray, weight, bias=None) -> FVArray:
    weight = to_np_arr(weight)
    out = input @ weight.T
    if bias is not None:
        out = out + to_np_arr(bias)
    return out


@_functional(F.conv1d, F.conv2d, F.conv3d)
def replay_conv(
    input: FVArray,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> FVArray:
    weight_np = to_np_arr(weight)
    bias_np = to_np_arr(bias) if bias is not None else None
    return conv_nd_replay(input, weight_np, bias_np, stride, padding, dilation, groups)


@_functional(F.batch_norm)
def replay_batch_norm(
    input: FVArray,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> FVArray:
    return batch_norm_replay(
        input,
        to_np_arr(running_mean),
        to_np_arr(running_var),
        to_np_arr(weight) if weight is not None else None,
        to_np_arr(bias) if bias is not None else None,
        training,
        momentum,
        eps,
    )


@_functional(F.max_pool1d, F.max_pool2d, F.max_pool3d)
def replay_max_pool(
    input: FVArray,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> FVArray:
    assert not ceil_mode, 'ceil_mode is not supported'
    assert not return_indices, 'return_indices with indices output is not supported'
    return pool_nd_replay(input, kernel_size, stride, padding, dilation, mode='max')


@_functional(F.adaptive_max_pool1d, F.adaptive_max_pool2d, F.adaptive_max_pool3d)
def replay_adaptive_max(input: FVArray, output_size, return_indices: bool = False) -> FVArray:
    assert not return_indices, 'return_indices with indices output is not supported'
    return adaptive_pool_replay(input, output_size, mode='max')


@_functional(F.avg_pool1d, F.avg_pool2d, F.avg_pool3d)
def replay_avg_pool(
    input: FVArray,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override=None,
) -> FVArray:
    assert not ceil_mode, 'ceil_mode is not supported'
    assert divisor_override is None, 'divisor_override is not supported'
    # count_include_pad controls whether zeros from padding count in avg denominator.
    # Our extract_patches uses a mask of valid positions; if count_include_pad we need
    # to divide by fixed kernel size; else by valid count. Implement simplest form:
    if count_include_pad:
        # numerator = sum over patches (padded with zeros), denom = full kernel volume
        x = np.moveaxis(input, 1, -1)
        ch = x.shape[-1]
        patches = extract_patches(x, kernel_size, stride if stride is not None else kernel_size, 1, padding, pad_value=0)
        patches = patches.reshape(patches.shape[:-1] + (-1, ch))
        out = np.mean(patches, axis=-2)
        return np.moveaxis(out, -1, 1)  # type: ignore
    return pool_nd_replay(input, kernel_size, stride, padding, 1, mode='avg')


@_functional(F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d)
def replay_adaptive_avg(input: FVArray, output_size) -> FVArray:
    return adaptive_pool_replay(input, output_size, mode='avg')


@_functional(F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)
def replay_conv_transpose(
    input: FVArray,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups: int = 1,
    dilation=1,
) -> FVArray:
    weight_np = to_np_arr(weight)
    bias_np = to_np_arr(bias) if bias is not None else None
    return conv_transpose_nd_replay(input, weight_np, bias_np, stride, padding, output_padding, groups, dilation)


@_functional(F.pixel_shuffle)
def replay_pixel_shuffle(input: FVArray, upscale_factor: int) -> FVArray:
    return pixel_shuffle_replay(input, upscale_factor)


@_functional(F.pixel_unshuffle)
def replay_pixel_unshuffle(input: FVArray, downscale_factor: int) -> FVArray:
    return pixel_unshuffle_replay(input, downscale_factor)


@_functional(F.interpolate, F.upsample, F.upsample_nearest)
def replay_interpolate(
    input: FVArray,
    size=None,
    scale_factor=None,
    mode: str = 'nearest',
    align_corners=None,
    recompute_scale_factor=None,
    antialias: bool = False,
) -> FVArray:
    assert mode == 'nearest', f'only nearest-neighbour interpolation is bit-exact; got {mode!r}'
    assert not antialias, 'antialias is not supported'
    rank = input.ndim - 2
    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale = (int(scale_factor),) * rank
        else:
            scale = tuple(int(s) for s in scale_factor)
    else:
        assert size is not None
        if isinstance(size, int):
            size = (size,) * rank
        spa_in = input.shape[2:]
        scale = tuple(s // si for s, si in zip(size, spa_in))
        assert all(s * si == sz for s, si, sz in zip(scale, spa_in, size)), (
            'only integer-scale nearest-neighbour upsampling is bit-exact'
        )
    return upsample_nearest_replay(input, scale)


@_functional(F.embedding)
def replay_embedding_fn(
    input: FVArray,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> FVArray:
    assert max_norm is None, 'max_norm is not supported'
    return embedding_replay(input, to_np_arr(weight), padding_idx)


@_functional(F.pad)
def replay_pad(input: FVArray, pad, mode: str = 'constant', value=None) -> FVArray:
    # torch's pad tuple is (last_dim_before, last_dim_after, prev_dim_before, ...)
    # convert to numpy's ((before, after),) * ndim
    assert len(pad) % 2 == 0
    np_pad = [(0, 0)] * input.ndim
    for i in range(len(pad) // 2):
        np_pad[-(i + 1)] = (int(pad[2 * i]), int(pad[2 * i + 1]))
    if mode == 'constant':
        return np.pad(input, np_pad, mode=mode, constant_values=value if value is not None else 0)  # type: ignore
    if mode == 'replicate':
        return np.pad(input, np_pad, mode='edge')  # type: ignore
    if mode == 'reflect':
        return np.pad(input, np_pad, mode='reflect')  # type: ignore
    if mode == 'circular':
        return np.pad(input, np_pad, mode='wrap')  # type: ignore
    raise ValueError(f'unknown pad mode: {mode}')


@_functional(F.dropout, F.dropout1d, F.dropout2d, F.dropout3d, F.alpha_dropout)
def replay_dropout(input, p: float = 0.5, training: bool = False, inplace: bool = False):
    assert not training, 'training-mode dropout is not supported'
    return input


@_functional(F.softmax)
def _softmax_unsupported(*args, **kwargs):
    raise NotImplementedError('softmax is not supported')


# ---------------------------------------------------------------------------
# torch tensor-level ops (call_function targets)
# ---------------------------------------------------------------------------


def _dim_to_axis(func):
    def wrapper(arr, dim=None, **kwargs):
        if 'keepdim' in kwargs:
            kwargs['keepdims'] = kwargs.pop('keepdim')
        return func(arr, axis=dim, **kwargs)

    return wrapper


@_functional(torch.cat, torch.concat, torch.concatenate)
def replay_cat(tensors, dim: int = 0):
    return np.concatenate(list(tensors), axis=dim)


@_functional(torch.stack)
def replay_stack(tensors, dim: int = 0):
    return np.stack(list(tensors), axis=dim)


def _split_by_indices(tensor, indices: list[int], dim: int) -> tuple:
    """Slice ``tensor`` along ``dim`` at the given split indices.

    Avoids ``np.split`` because FVArray's ``__array_function__`` re-wraps the
    returned list into a single FVArray, which fails when the slices are not
    uniformly shaped.
    """
    pieces = []
    start = 0
    for end in [*indices, tensor.shape[dim]]:
        slicer = [slice(None)] * tensor.ndim
        slicer[dim] = slice(start, end)
        pieces.append(tensor[tuple(slicer)])
        start = end
    return tuple(pieces)


@_functional(torch.split)
def replay_split(tensor, split_size_or_sections, dim: int = 0):
    if isinstance(split_size_or_sections, int):
        n = tensor.shape[dim]
        indices = list(range(split_size_or_sections, n, split_size_or_sections))
    else:
        indices = list(np.cumsum(split_size_or_sections)[:-1])
    return _split_by_indices(tensor, indices, dim)


@_functional(torch.chunk)
def replay_chunk(tensor, chunks: int, dim: int = 0):
    n = tensor.shape[dim]
    base, rem = divmod(n, chunks)
    sizes = [base + 1] * rem + [base] * (chunks - rem)
    indices = list(np.cumsum(sizes)[:-1])
    return _split_by_indices(tensor, indices, dim)


@_functional(torch.matmul)
def replay_matmul(x1, x2):
    return _einsum('...ij,...jk->...ik', x1, x2)


@_functional(torch.einsum)
def replay_einsum(equation, *operands):
    return _einsum(equation, *operands)


@_functional(torch.clamp, torch.clip)
def replay_clamp(input, min=None, max=None):
    return np.clip(input, min, max)


@_functional(torch.where)
def replay_where(condition, x=None, y=None):
    if x is None and y is None:
        return np.nonzero(condition)
    return np.where(condition, x, y)


@_functional(torch.reshape)
def replay_reshape_fn(input, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return input.reshape(shape)


@_functional(torch.flatten)
def replay_flatten_fn(input, start_dim: int = 0, end_dim: int = -1):
    end = end_dim if end_dim >= 0 else input.ndim + end_dim
    new_shape = input.shape[:start_dim] + (-1,) + input.shape[end + 1 :]
    return input.reshape(new_shape)


@_functional(torch.unsqueeze)
def replay_unsqueeze(input, dim: int):
    return np.expand_dims(input, axis=dim)


@_functional(torch.squeeze)
def replay_squeeze(input, dim=None):
    if dim is None:
        return np.squeeze(input)
    return np.squeeze(input, axis=dim)


@_functional(torch.transpose)
def replay_transpose_fn(input, dim0: int, dim1: int):
    return np.swapaxes(input, dim0, dim1)


@_functional(torch.permute)
def replay_permute_fn(input, dims):
    return np.transpose(input, tuple(dims))


@_functional(torch.moveaxis, torch.movedim)
def replay_moveaxis(input, source, destination):
    return np.moveaxis(input, source, destination)


@_functional(torch.repeat_interleave)
def replay_repeat(input, repeats, dim=None):
    return np.repeat(input, repeats, axis=dim)


@_functional(torch.tile)
def replay_tile(input, dims):
    return np.tile(input, dims)


# comparison / elementwise min-max via torch.* callables

for _t, _np in [
    (torch.maximum, np.maximum),
    (torch.minimum, np.minimum),
    (torch.eq, np.equal),
    (torch.equal, np.equal),
    (torch.ne, np.not_equal),
    (torch.not_equal, np.not_equal),
    (torch.gt, np.greater),
    (torch.greater, np.greater),
    (torch.ge, np.greater_equal),
    (torch.greater_equal, np.greater_equal),
    (torch.lt, np.less),
    (torch.less, np.less),
    (torch.le, np.less_equal),
    (torch.less_equal, np.less_equal),
]:
    _functional(_t)(_np)


# Use Python operators for arithmetic so RetardedFVArray (from .apply) dispatches
# through its own __mul__/__add__/etc. Numpy ufuncs would bypass that and lose the
# delayed lookup-table operation.

_functional(operator.add, torch.add)(lambda a, b: a + b)
_functional(operator.sub, torch.sub, torch.subtract)(lambda a, b: a - b)
_functional(operator.mul, torch.mul, torch.multiply)(lambda a, b: a * b)
_functional(operator.truediv, torch.div, torch.divide, torch.true_divide)(lambda a, b: a / b)
_functional(operator.floordiv, torch.floor_divide)(lambda a, b: a // b)
_functional(operator.mod, torch.remainder)(lambda a, b: a % b)
_functional(operator.matmul)(lambda a, b: _einsum('...ij,...jk->...ik', a, b))
_functional(operator.neg)(lambda x: -x)
_functional(operator.pos)(lambda x: +x)
_functional(operator.eq)(np.equal)
_functional(operator.ne)(np.not_equal)
_functional(operator.lt)(np.less)
_functional(operator.le)(np.less_equal)
_functional(operator.gt)(np.greater)
_functional(operator.ge)(np.greater_equal)
_functional(operator.getitem)(lambda x, key: x[key])


# reductions (dim + keepdim kwargs)

_functional(torch.sum)(_dim_to_axis(np.sum))
_functional(torch.mean)(_dim_to_axis(np.mean))
_functional(torch.prod)(_dim_to_axis(np.prod))
_functional(torch.amax)(_dim_to_axis(np.amax))
_functional(torch.amin)(_dim_to_axis(np.amin))


def _reduce_scalar_or_values(np_reducer):
    """torch.max(x, dim=...) returns (values, indices); ours returns just values."""

    def wrapper(arr, dim=None, keepdim: bool = False):
        if dim is None:
            return np_reducer(arr)
        return np_reducer(arr, axis=dim, keepdims=keepdim)

    return wrapper


_functional(torch.max)(_reduce_scalar_or_values(np.amax))
_functional(torch.min)(_reduce_scalar_or_values(np.amin))


_functional(torch.all)(_dim_to_axis(np.all))
_functional(torch.any)(_dim_to_axis(np.any))


@_functional(torch.count_nonzero)
def replay_count_nonzero(input, dim=None):
    return np.count_nonzero(input, axis=dim)


@_functional(torch.argmax)
def replay_argmax(input, dim=None, keepdim: bool = False):
    if dim is None:
        return np.argmax(input)
    return np.argmax(input, axis=dim, keepdims=keepdim)


@_functional(torch.argmin)
def replay_argmin(input, dim=None, keepdim: bool = False):
    if dim is None:
        return np.argmin(input)
    return np.argmin(input, axis=dim, keepdims=keepdim)


@_functional(torch.sort)
def replay_sort(input, dim: int = -1, descending: bool = False, stable: bool = False):
    out = np.sort(input, axis=dim)
    if descending:
        out = np.flip(out, axis=dim)
    return out


# Activations exposed via torch.nn.functional that take shape-specific kwargs
# (signatures differ from the unary map). Handle explicitly.


@_functional(F.elu)
def replay_elu(input, alpha: float = 1.0):
    if isinstance(input, FVArray):
        return input.apply(lambda x: np.where(x > 0, x, alpha * (np.exp(x) - 1)))
    return np.where(input > 0, input, alpha * (np.exp(input) - 1))


@_functional(F.hardtanh)
def replay_hardtanh(input, min_val: float = -1.0, max_val: float = 1.0):
    return np.clip(input, min_val, max_val)


@_functional(F.leaky_relu)
def replay_leaky_relu(input, negative_slope: float = 0.01):
    return np.where(input < 0, input * negative_slope, input)  # type: ignore


@_functional(F.prelu)
def replay_prelu(input, weight):
    alpha = to_np_arr(weight)
    if alpha.ndim == 1 and alpha.size != 1:
        shape = [1] * input.ndim
        shape[1] = input.shape[1]
        alpha = alpha.reshape(shape)
    return np.where(input < 0, input * alpha, input)  # type: ignore


# getattr for .T / .mT / .shape


def _replay_getattr(obj, name: str):
    if name == 'T':
        return np.swapaxes(obj, -1, -2) if obj.ndim >= 2 else obj
    if name == 'mT':
        return np.swapaxes(obj, -1, -2)
    if name == 'shape':
        return obj.shape
    if name == 'ndim':
        return obj.ndim
    # torch.max / torch.min / torch.sort return a namedtuple; our replay returns the
    # values tensor directly, so both accessors are identity.
    if name == 'values':
        return obj
    if name == 'indices':
        raise NotImplementedError('accessing .indices of a torch reduction/sort is not supported')
    raise AttributeError(f'unsupported getattr for replay: {name!r}')


_functional_map[getattr] = _replay_getattr
