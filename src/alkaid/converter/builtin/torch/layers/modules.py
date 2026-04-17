from typing import Any

import numpy as np
import torch

from alkaid.trace import FVariable, FVArray
from alkaid.trace.ops import relu as _relu

from .direct import torch_numpy_unary_map
from .functional import (
    adaptive_pool_replay,
    conv_nd_replay,
    conv_transpose_nd_replay,
    embedding_replay,
    pixel_shuffle_replay,
    pixel_unshuffle_replay,
    pool_nd_replay,
    replay_pad,
    to_np_arr,
    upsample_nearest_replay,
)

# ---------------------------------------------------------------------------
# Module-handler registry, base class, and metaclass
# ---------------------------------------------------------------------------

_modules_map: 'dict[type, type[ReplayModuleBase]]' = {}


class HandlerRegMeta(type):
    """Metaclass: auto-register subclasses keyed by their ``handles`` tuple."""

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name == 'ReplayModuleBase':
            return cls
        handles: type | tuple[type, ...] = namespace.get('handles', ())
        if not isinstance(handles, tuple):
            handles = (handles,)
        for handle in handles:
            _modules_map[handle] = cls  # type: ignore
        return cls


ARR_or_tuple_ARR = tuple[FVArray, ...] | FVArray


class ReplayModuleBase(metaclass=HandlerRegMeta):
    handles: tuple[type, ...] | type = ()

    def __init__(self, module: torch.nn.Module):
        assert isinstance(module, self.handles)
        self.module: Any = module  # type: ignore

    def _load_weight(self, name: str) -> np.ndarray:
        w = getattr(self.module, name)
        if w is None:
            return np.array(0.0)
        return to_np_arr(w)

    def call(self, *args, **kwargs) -> ARR_or_tuple_ARR | dict[str, ARR_or_tuple_ARR]: ...

    @staticmethod
    def _normalize_to_tuple(
        v: FVArray | FVariable | tuple[FVArray | FVariable, ...],
    ) -> tuple[FVArray, ...]:
        if isinstance(v, FVArray):
            return (v,)
        elif isinstance(v, FVariable):
            return (FVArray(np.array([v])),)
        return tuple(FVArray(np.array(x)) if isinstance(x, FVariable) else x for x in v)

    def _wrap_call(self, *args, **kwargs) -> dict[str, tuple[FVArray, ...]]:
        r = self.call(*args, **kwargs)
        if isinstance(r, dict):
            r = {k: self._normalize_to_tuple(v) for k, v in r.items()}
        else:
            r = {'final': self._normalize_to_tuple(r)}
        return r

    def __call__(self, *args, **kwargs) -> dict[str, tuple[FVArray, ...]]:
        assert all(not isinstance(a, FVArray) for a in kwargs.values())
        return self._wrap_call(*args, **kwargs)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


class ReplayLinear(ReplayModuleBase):
    handles = (torch.nn.Linear,)

    def call(self, input: FVArray) -> FVArray:
        out = input @ self._load_weight('weight').T
        bias = self._load_weight('bias')
        if bias.shape != ():
            out = out + bias
        return out


class ReplayConv(ReplayModuleBase):
    handles = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)

    def call(self, input: FVArray) -> FVArray:
        module = self.module
        return conv_nd_replay(
            input,
            self._load_weight('weight'),
            self._load_weight('bias'),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )


class ReplayBatchNorm(ReplayModuleBase):
    handles = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

    def fused_scale_offset(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the fused ``(scale, offset)`` such that ``output = input * scale + offset``.

        Downstream packages may override this to supply bit-exact quantized
        values without separately expressing gamma/beta/running_mean/running_var.
        """
        module = self.module
        mean = self._load_weight('running_mean')
        variance = self._load_weight('running_var')
        gamma = self._load_weight('weight') if module.affine else np.ones_like(mean)
        beta = self._load_weight('bias') if module.affine else np.zeros_like(mean)
        scale = gamma / np.sqrt(variance + module.eps)
        offset = beta - mean * scale
        return scale, offset

    def call(self, input: FVArray) -> FVArray:
        scale, offset = self.fused_scale_offset()
        shape = [1] * input.ndim
        shape[1] = input.shape[1]
        return input * scale.reshape(shape) + offset.reshape(shape)  # type: ignore


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class ReplayReLU(ReplayModuleBase):
    handles = (torch.nn.ReLU,)

    def call(self, input: FVArray) -> FVArray:
        return _relu(input)


class ReplayLeakyReLU(ReplayModuleBase):
    handles = (torch.nn.LeakyReLU,)

    def call(self, input: FVArray) -> FVArray:
        slope = self.module.negative_slope
        return np.where(input < 0, input * slope, input)  # type: ignore


class ReplayPReLU(ReplayModuleBase):
    handles = (torch.nn.PReLU,)

    def call(self, input: FVArray) -> FVArray:
        alpha = self._load_weight('weight')
        # alpha is per-channel (dim=1) or scalar; broadcast over channel axis
        if alpha.ndim == 1 and alpha.size != 1:
            shape = [1] * input.ndim
            shape[1] = input.shape[1]
            alpha = alpha.reshape(shape)
        return np.where(input < 0, input * alpha, input)  # type: ignore


class ReplaySimpleActivation(ReplayModuleBase):
    """Pure functions of a single input — dispatched by class name."""

    handles = (
        torch.nn.Sigmoid,
        torch.nn.Tanh,
        torch.nn.SiLU,
        torch.nn.GELU,
        torch.nn.ELU,
        torch.nn.SELU,
        torch.nn.Softplus,
        torch.nn.Softsign,
        torch.nn.Hardsigmoid,
        torch.nn.Hardswish,
        torch.nn.Hardtanh,
        torch.nn.ReLU6,
        torch.nn.LogSigmoid,
    )

    def call(self, input: FVArray) -> FVArray:
        name = self.module.__class__.__name__.lower()
        if name not in torch_numpy_unary_map:
            raise NotImplementedError(f'unsupported activation: {name}')
        return torch_numpy_unary_map[name](input)


# ---------------------------------------------------------------------------
# Shape / structural
# ---------------------------------------------------------------------------


class ReplayFlatten(ReplayModuleBase):
    handles = (torch.nn.Flatten,)

    def call(self, input: FVArray) -> FVArray:
        start, end = self.module.start_dim, self.module.end_dim
        ndim = input.ndim
        start = start if start >= 0 else ndim + start
        end = end if end >= 0 else ndim + end
        new_shape = input.shape[:start] + (-1,) + input.shape[end + 1 :]
        return input.reshape(new_shape)


class ReplayUnflatten(ReplayModuleBase):
    handles = (torch.nn.Unflatten,)

    def call(self, input: FVArray) -> FVArray:
        dim = self.module.dim
        unflat = tuple(self.module.unflattened_size)
        ndim = input.ndim
        dim = dim if dim >= 0 else ndim + dim
        new_shape = input.shape[:dim] + unflat + input.shape[dim + 1 :]
        return input.reshape(new_shape)


class ReplayNoOp(ReplayModuleBase):
    handles = (
        torch.nn.Identity,
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )

    def call(self, input: FVArray) -> FVArray:
        return input


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


class ReplayPool(ReplayModuleBase):
    handles = (
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
    )

    def call(self, input: FVArray) -> FVArray:
        module = self.module
        mode = 'max' if isinstance(module, (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)) else 'avg'
        return pool_nd_replay(
            input,
            module.kernel_size,
            getattr(module, 'stride', None),
            getattr(module, 'padding', 0),
            getattr(module, 'dilation', 1) if mode == 'max' else 1,
            mode=mode,
        )


_PAD_MODE_MAP = {
    'Zero': ('constant', 0.0),
    'Constant': ('constant', None),  # value comes from module
    'Reflection': ('reflect', None),
    'Replication': ('replicate', None),
    'Circular': ('circular', None),
}


class ReplayPaddingModule(ReplayModuleBase):
    """All nn.{Zero,Constant,Reflection,Replication,Circular}Pad{1,2,3}d modules."""

    handles = (
        torch.nn.ZeroPad1d,
        torch.nn.ZeroPad2d,
        torch.nn.ZeroPad3d,
        torch.nn.ConstantPad1d,
        torch.nn.ConstantPad2d,
        torch.nn.ConstantPad3d,
        torch.nn.ReflectionPad1d,
        torch.nn.ReflectionPad2d,
        torch.nn.ReflectionPad3d,
        torch.nn.ReplicationPad1d,
        torch.nn.ReplicationPad2d,
        torch.nn.ReplicationPad3d,
        torch.nn.CircularPad1d,
        torch.nn.CircularPad2d,
        torch.nn.CircularPad3d,
    )

    def call(self, input: FVArray) -> FVArray:
        name = self.module.__class__.__name__
        kind = name.replace('Pad1d', '').replace('Pad2d', '').replace('Pad3d', '')
        mode, default_val = _PAD_MODE_MAP[kind]
        value = getattr(self.module, 'value', default_val)
        pad = tuple(self.module.padding)
        return replay_pad(input, pad, mode=mode, value=value)


class ReplayUpsample(ReplayModuleBase):
    handles = (
        torch.nn.Upsample,
        torch.nn.UpsamplingNearest2d,
    )

    def call(self, input: FVArray) -> FVArray:
        module = self.module
        mode = getattr(module, 'mode', 'nearest')
        assert mode == 'nearest', f'only nearest-neighbour upsampling is bit-exact; got {mode!r}'
        rank = input.ndim - 2
        if module.scale_factor is not None:
            sf = module.scale_factor
            if isinstance(sf, (int, float)):
                scale = (int(sf),) * rank
            else:
                scale = tuple(int(s) for s in sf)
        else:
            spa_in = input.shape[2:]
            size = module.size
            if isinstance(size, int):
                size = (size,) * rank
            scale = tuple(s // si for s, si in zip(size, spa_in))
            assert all(s * si == sz for s, si, sz in zip(scale, spa_in, size)), (
                'only integer-scale nearest-neighbour upsampling is bit-exact'
            )
        return upsample_nearest_replay(input, scale)


class ReplayPixelShuffle(ReplayModuleBase):
    handles = (torch.nn.PixelShuffle,)

    def call(self, input: FVArray) -> FVArray:
        return pixel_shuffle_replay(input, self.module.upscale_factor)


class ReplayPixelUnshuffle(ReplayModuleBase):
    handles = (torch.nn.PixelUnshuffle,)

    def call(self, input: FVArray) -> FVArray:
        return pixel_unshuffle_replay(input, self.module.downscale_factor)


class ReplayConvTranspose(ReplayModuleBase):
    handles = (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)

    def call(self, input: FVArray) -> FVArray:
        module = self.module
        return conv_transpose_nd_replay(
            input,
            self._load_weight('weight'),
            self._load_weight('bias'),
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
            groups=module.groups,
            dilation=module.dilation,
        )


class ReplayEmbedding(ReplayModuleBase):
    handles = (torch.nn.Embedding,)

    def call(self, input: FVArray) -> FVArray:
        return embedding_replay(input, self._load_weight('weight'), self.module.padding_idx)


class ReplayAdaptivePool(ReplayModuleBase):
    handles = (
        torch.nn.AdaptiveMaxPool1d,
        torch.nn.AdaptiveMaxPool2d,
        torch.nn.AdaptiveMaxPool3d,
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
    )

    def call(self, input: FVArray) -> FVArray:
        module = self.module
        mode = 'max' if 'Max' in module.__class__.__name__ else 'avg'
        return adaptive_pool_replay(input, module.output_size, mode)
