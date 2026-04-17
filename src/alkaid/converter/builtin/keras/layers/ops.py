import re
from collections.abc import Sequence

import keras
import numpy as np
from keras.src.ops.nn import Elu, Gelu, HardSigmoid, HardSilu, Selu, Sigmoid, Silu
from keras.src.ops.numpy import (
    Abs,
    Absolute,
    Add,
    All,
    Amax,
    Amin,
    Any,
    Arccos,
    Arcsin,
    Arcsinh,
    Arctanh,
    Argmax,
    Argmin,
    Average,
    Ceil,
    Clip,
    Concatenate,
    Cos,
    Cosh,
    CountNonzero,
    Divide,
    Dot,
    Einsum,
    Equal,
    Exp,
    Expm1,
    Floor,
    GetItem,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Log,
    Log1p,
    Matmul,
    Max,
    Maximum,
    Mean,
    Min,
    Minimum,
    Moveaxis,
    Multiply,
    Pad,
    Prod,
    Ravel,
    Repeat,
    Reshape,
    Round,
    Sign,
    Signbit,
    Sin,
    Sinh,
    Sort,
    Sqrt,
    Subtract,
    Sum,
    Tan,
    Tanh,
    Transpose,
    TrueDivide,
)

from alkaid.trace import FVArray
from alkaid.trace.ops import einsum

from ._base import ReplayOperationBase
from .activation import keras_numpy_unary_map


class ReplayReshape(ReplayOperationBase):
    handles = (keras.layers.Reshape, keras.layers.Flatten, Reshape, Ravel)

    def call(self, inputs: FVArray) -> FVArray:
        if isinstance(self.op, (keras.layers.Flatten, Ravel)):
            return inputs.ravel()
        elif isinstance(self.op, keras.layers.Reshape):
            return inputs.reshape(self.op.target_shape)
        elif isinstance(self.op, Reshape):
            return inputs.reshape(self.op.newshape)
        else:
            raise TypeError(f'Unsupported layer type: {type(self.op)}')


class ReplayMerge(ReplayOperationBase):
    handles = (
        keras.layers.Add,
        keras.layers.Concatenate,
        keras.layers.Multiply,
        keras.layers.Subtract,
        keras.layers.Maximum,
        keras.layers.Minimum,
        keras.layers.Average,
    )

    def call(self, inputs: tuple[FVArray, ...]) -> FVArray:
        op = self.op
        name = op.__class__.__name__
        _inputs: FVArray = np.stack(np.broadcast_arrays(*inputs), axis=0)  # type: ignore
        match name:
            case 'Add':
                return np.sum(_inputs, axis=0)
            case 'Average':
                return np.mean(_inputs, axis=0)
            case 'Subtract':
                assert len(_inputs) == 2, 'Subtract operation requires exactly two inputs'
                return _inputs[0] - _inputs[1]
            case 'Multiply':
                return np.prod(_inputs, axis=0)
            case 'Maximum':
                return np.amax(_inputs, axis=0)
            case 'Minimum':
                return np.amin(_inputs, axis=0)
            case 'Concatenate':
                return np.concatenate(_inputs, axis=op.axis)  # type: ignore

            case _:
                raise TypeError(f'Unsupported layer type: {type(op)}')


class ReplayRepeatVector(ReplayOperationBase):
    handles = (keras.layers.RepeatVector,)

    def call(self, inputs: FVArray) -> FVArray:
        op: keras.layers.RepeatVector = self.op
        return np.repeat(inputs, op.n, axis=0)  # type: ignore


class ReplayGetItem(ReplayOperationBase):
    handles = (GetItem,)

    def call(self, x: FVArray, key) -> FVArray:
        return x[key]


class ReplayReduction(ReplayOperationBase):
    handles = (Sum, Max, Min, CountNonzero, All, Any, Prod, Mean)

    def call(self, x: FVArray, axis=None, keepdims=False) -> FVArray:
        match self.op.__class__.__name__:
            case 'Sum':
                op = np.sum
            case 'Max':
                op = np.amax
            case 'Min':
                op = np.amin
            case 'CountNonzero':
                op = np.count_nonzero
            case 'All':
                op = np.all
            case 'Any':
                op = np.any
            case 'Prod':
                op = np.prod
            case 'Mean':
                op = np.mean
            case _:
                raise TypeError(f'Unsupported reduction operation: {type(self.op)}')

        # axis/keepdims are stored as op attributes, not passed as kwargs
        axis = self.op.axis if hasattr(self.op, 'axis') else axis
        keepdims = self.op.keepdims if hasattr(self.op, 'keepdims') else keepdims
        return op(x, axis=axis, keepdims=keepdims)  # type: ignore


class ReplayAverage(ReplayOperationBase):
    handles = Average

    def call(self, x, weights=None) -> FVArray:
        axis = self.op.axis
        return np.average(x, axis=axis, weights=weights)  # type: ignore


class ReplayArithmetic(ReplayOperationBase):
    handles = (Add, Subtract, Multiply, TrueDivide, Divide, Maximum, Minimum)

    def call(self, x1: FVArray, x2: FVArray) -> FVArray:
        name = self.op.__class__.__name__
        match name:
            case 'Add':
                return x1 + x2
            case 'Subtract':
                return x1 - x2
            case 'Multiply':
                return x1 * x2
            case 'TrueDivide' | 'Divide':
                return x1 / x2
            case 'Maximum':
                return np.maximum(x1, x2)  # type: ignore
            case 'Minimum':
                return np.minimum(x1, x2)  # type: ignore
            case _:
                raise TypeError(f'Unsupported arithmetic operation: {type(self.op)}')


class ReplayConcatenate(ReplayOperationBase):
    handles = (Concatenate,)

    def call(self, xs: Sequence[FVArray]) -> FVArray:
        return np.concatenate(list(xs), axis=self.op.axis)  # type: ignore


class ReplayRepeat(ReplayOperationBase):
    handles = (Repeat,)

    def call(self, x: FVArray) -> FVArray:
        return np.repeat(x, self.op.repeats, axis=self.op.axis)  # type: ignore


class ReplayTranspose(ReplayOperationBase):
    handles = (Transpose,)

    def call(self, x: FVArray) -> FVArray:
        axes = self.op.axes
        return np.transpose(x, axes)  # type: ignore


class ReplayMoveaxis(ReplayOperationBase):
    handles = (Moveaxis,)

    def call(self, x: FVArray):
        return np.moveaxis(x, self.op.source, self.op.destination)  # type: ignore


class ReplayNoOp(ReplayOperationBase):
    __noop_layers = []
    for k, v in keras.layers.__dict__.items():
        name = k.lower()
        if 'dropout' in name or 'random' in name or 'noise' in name:
            __noop_layers.append(v)

    handles = tuple(__noop_layers)

    def call(self, x: FVArray, training=False) -> FVArray:
        assert not training, 'Training mode is not supported in mirror operation'
        return x


class ReplayEinsum(ReplayOperationBase):
    handles = (Einsum, keras.layers.Dot)

    def call(self, *_inputs: tuple[FVArray, FVArray] | FVArray) -> FVArray:
        op = self.op
        inputs: tuple[FVArray, FVArray]
        if len(_inputs) == 1 and isinstance(_inputs[0], (tuple, list)):
            inputs = tuple(_inputs[0])  # type: ignore
        else:
            inputs = _inputs  # type: ignore
        assert len(inputs) == 2, 'Only (Q)Einsum operations with exactly two inputs are supported'

        if isinstance(op, Einsum):
            eq = op.subscripts
        else:  # keras.layers.Dot
            dim0, dim1 = inputs[0].ndim, inputs[1].ndim
            letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[: dim0 + dim1 - 1]
            sub0 = letters[:dim0]
            _sub1 = list(sub0[0] + letters[dim0:])  # share batch idx
            axes = list(op.axes) if not isinstance(op.axes, int) else [op.axes, op.axes]
            idx0, idx1 = axes[0] % dim0, axes[1] % dim1
            contracted = sub0[idx0]
            _sub1[idx1] = contracted
            sub1 = ''.join(_sub1)
            sub_out = ''.join(c for c in sub0 if c != contracted) + ''.join(c for c in sub1[1:] if c != contracted)
            eq = f'{sub0},{sub1}->{sub_out}'
        return einsum(eq, inputs[0], inputs[1])  # type: ignore


class ReplayMatmul(ReplayOperationBase):
    handles = (Matmul, Dot)

    def call(self, x1: FVArray, x2: FVArray) -> FVArray:
        return einsum('...ij,...jk->...ik', x1, x2)  # type: ignore


class ReplayAbs(ReplayOperationBase):
    handles = (Absolute, Abs)

    def call(self, x: FVArray) -> FVArray:
        return np.abs(x)  # type: ignore


class ReplayClip(ReplayOperationBase):
    handles = (Clip,)

    def call(self, x: FVArray) -> FVArray:
        x_min = getattr(self.op, 'x_min', getattr(self.op, 'a_min', None))
        x_max = getattr(self.op, 'x_max', getattr(self.op, 'a_max', None))
        return np.clip(x, x_min, x_max)  # type: ignore


class ReplayRound(ReplayOperationBase):
    handles = (Round,)

    def call(self, x: FVArray) -> FVArray:
        return np.round(x)  # type: ignore


class ReplayFloor(ReplayOperationBase):
    handles = (Floor,)

    def call(self, x: FVArray) -> FVArray:
        return np.floor(x)  # type: ignore


class ReplayCeil(ReplayOperationBase):
    handles = (Ceil,)

    def call(self, x: FVArray) -> FVArray:
        return np.ceil(x)  # type: ignore


class ReplaySortLike(ReplayOperationBase):
    handles = (Argmax, Argmin, Amax, Amin, Sort)

    def call(self, x: FVArray) -> FVArray:
        fn = getattr(np, self.op.__class__.__name__.lower())
        kwargs = {}
        if hasattr(self.op, 'axis'):
            kwargs['axis'] = self.op.axis
        if hasattr(self.op, 'keepdims'):
            kwargs['keepdims'] = self.op.keepdims
        return fn(x, **kwargs)  # type: ignore


class ReplayUnary(ReplayOperationBase):
    handles = (
        Sin,
        Cos,
        Tan,
        Exp,
        Log,
        Sqrt,
        Sign,
        Signbit,
        Sinh,
        Cosh,
        Tanh,
        Arccos,
        Arcsin,
        Arctanh,
        Arcsinh,
        Expm1,
        Log1p,
    )

    def call(self, x: FVArray) -> FVArray:
        name = self.op.__class__.__name__
        return getattr(np, name.lower())(x)


class ReplayKerasNNActivation(ReplayOperationBase):
    handles = (Sigmoid, Silu, HardSigmoid, HardSilu, Gelu, Elu, Selu)

    def call(self, x: FVArray) -> FVArray:
        snake_name = re.sub(r'(?<!^)(?=[A-Z])', '_', self.op.__class__.__name__).lower()
        return keras_numpy_unary_map[snake_name](x)  # type: ignore


class ReplayPad(ReplayOperationBase):
    handles = (Pad,)

    def call(self, x: FVArray, constant_values=None) -> FVArray:
        self.op: Pad
        pad_width = self.op.pad_width
        mode = self.op.mode
        if mode == 'constant':
            return np.pad(x, pad_width, mode=mode, constant_values=constant_values)  # type: ignore
        else:
            return np.pad(x, pad_width, mode=mode)  # type: ignore


class ReplayCmp(ReplayOperationBase):
    handles = (Equal, Greater, GreaterEqual, Less, LessEqual)

    def call(self, x1: FVArray, x2: FVArray) -> FVArray:
        name = self.op.__class__.__name__
        match name:
            case 'Equal':
                return x1 == x2
            case 'Greater':
                return x1 > x2
            case 'GreaterEqual':
                return x1 >= x2
            case 'Less':
                return x1 < x2
            case 'LessEqual':
                return x1 <= x2
            case _:
                raise TypeError(f'Unsupported comparison operation: {type(self.op)}')
