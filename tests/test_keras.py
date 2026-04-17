from functools import partial

import keras
import numpy as np
import pytest
from keras import layers, ops

from alkaid.converter import trace_model
from alkaid.trace import FVArray, trace

_TORCH = keras.backend.backend() == 'torch'
_torch_skip = partial(pytest.mark.skipif, _TORCH)


def _qdata(shape, kif, seed=0) -> np.ndarray:
    k, i, f = kif
    rng = np.random.default_rng(seed)
    hi = 2.0**i - 2.0**-f
    lo = -(2.0**i) * k
    raw = rng.uniform(lo, hi, size=shape).astype(np.float32)
    step = 2.0**f
    return np.floor(raw * step).astype(np.float32) / step


def _perturb_weights(model, fbits=2, seed=42):
    rng = np.random.default_rng(seed)
    step = 2.0**fbits
    new_weights = []
    for w in model.get_weights():
        if w.ndim == 0:
            new_weights.append(w)
            continue
        r = rng.standard_normal(w.shape).astype(np.float32)
        new_weights.append(np.round(r * step) / step)
    model.set_weights(new_weights)


def _run(op, shapes, kif=(1, 4, 4), n=65536, perturb_weights=True, hook_model=None, hook_data=None, wrap_model=True, **kwargs):
    inps = [layers.Input(shape=s) for s in shapes]
    out = op(inps[0]) if len(inps) == 1 else op(inps)
    if wrap_model:
        model = keras.Model(inps[0] if len(inps) == 1 else inps, out)
    else:
        model = op
    if perturb_weights:
        _perturb_weights(model)
    if hook_model is not None:
        hook_model(model)
    datas = [_qdata((n,) + s, kif, seed=i) for i, s in enumerate(shapes)]
    if hook_data is not None:
        datas = hook_data(datas)
    trace_inp, trace_out = trace_model(model, inputs_kif=kif, **kwargs)
    comb = trace(trace_inp, trace_out)
    data_keras = datas[0] if len(datas) == 1 else datas
    expected: np.ndarray = ops.convert_to_numpy(model(data_keras))  # type: ignore
    data_comb = np.concatenate([x.reshape((n, -1)) for x in datas], axis=1)
    actual = comb.predict(data_comb).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDense:
    @pytest.fixture(
        params=[
            (layers.Dense(4, use_bias=True, activation='relu'), (8,)),
            (layers.Dense(4, use_bias=False, activation=keras.activations.hard_tanh), (8,)),
            (layers.EinsumDense('...c,cd->...d', output_shape=(4,)), (4, 8)),
        ],
        ids=['Dense[bias]', 'Dense[nobias]', 'EinsumDense'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class TestConv:
    @pytest.fixture(
        params=[
            (layers.Conv1D(4, 3, padding='valid'), (16, 3)),
            (layers.Conv1D(4, 3, padding='same'), (16, 3)),
            (layers.Conv2D(4, (4, 3), padding='valid'), (8, 8, 3)),
            (layers.Conv2D(4, (3, 4), padding='same'), (8, 8, 3)),
        ],
        ids=['Conv1D[valid]', 'Conv1D[same]', 'Conv2D[valid]', 'Conv2D[same]'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class TestBatchNorm:
    def test(self):
        # epsilon=0 so scale = gamma / sqrt(variance) is exact when variance is a perfect square
        # variance=0.0625 -> sqrt=0.25 -> scale = gamma*4
        # gamma=0.25 -> scale=1.0; beta=0.5, mean=0.25 -> offset = 0.5 - 0.25*1.0 = 0.25
        def hook_model(model):
            model.layers[1].set_weights(
                [
                    np.full(8, 0.25, dtype=np.float32),  # gamma
                    np.full(8, 0.5, dtype=np.float32),  # beta
                    np.full(8, 0.25, dtype=np.float32),  # moving_mean
                    np.full(8, 0.0625, dtype=np.float32),  # moving_variance
                ]
            )

        _run(layers.BatchNormalization(epsilon=0), [(8,)], kif=(1, 2, 4), hook_model=hook_model)


class TestReLU:
    @pytest.fixture(
        params=[
            layers.ReLU(),
            layers.LeakyReLU(negative_slope=0.25),
            layers.PReLU(),
        ],
        ids=['ReLU', 'LeakyReLU', 'PReLU'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,)])


def custom(x):
    return ops.sin(x) + ops.cos(x)


class TestActivation:
    @pytest.fixture(
        params=[
            keras.activations.linear,
            keras.activations.relu,
            keras.activations.tanh,
            keras.activations.sigmoid,
            keras.activations.swish,
            keras.activations.gelu,
            keras.activations.elu,
            keras.activations.selu,
            keras.activations.silu,
            keras.activations.softplus,
            keras.activations.softsign,
            keras.activations.exponential,
            keras.activations.hard_silu,
            keras.activations.hard_sigmoid,
            keras.activations.hard_swish,
            keras.activations.log_sigmoid,
            keras.activations.hard_tanh,
            keras.activations.relu6,
            custom,
        ]
    )
    def activation(self, request):
        return request.param

    def test(self, activation) -> None:
        _run(lambda x: ops.round(layers.Activation(activation)(x) * 16), [(8,)], kif=(1, 2, 4))


class TestUnaryNonlinear:
    """LUT-backed unary functions: follow with *n then floor for non-trivial output."""

    @pytest.fixture(
        params=[
            ops.sin,
            ops.cos,
            ops.tanh,
            ops.sinh,
            ops.cosh,
            ops.arccos,
            ops.arcsin,
            ops.arctanh,
            ops.arcsinh,
            ops.log,
            ops.exp,
            ops.expm1,
            ops.log1p,
            ops.sigmoid,
            ops.silu,
            ops.hard_sigmoid,
            ops.hard_swish,
            ops.gelu,
            ops.elu,
            ops.selu,
            ops.sign,
        ]
    )
    def fn(self, request):
        return request.param

    @pytest.fixture
    def op(self, fn):
        return lambda x: ops.round(fn(x) * 16)

    def test(self, op, fn):
        # Use smaller range to avoid float32 precision issues at tanh/etc boundaries
        hook_data = lambda x: x if fn is not ops.log else [np.maximum(2.0**-5, d) for d in x]
        _run(op, [(8,)], kif=(1, -1, 5), hook_data=hook_data)


class TestPooling:
    @pytest.fixture(
        params=[
            (layers.MaxPooling1D(pool_size=3, padding='same'), (8, 4)),
            (layers.AveragePooling1D(pool_size=2), (8, 4)),
            (layers.MaxPooling2D(pool_size=(2, 3), padding='same'), (8, 8, 4)),
            (layers.AveragePooling2D(pool_size=2), (8, 8, 4)),
            (layers.GlobalMaxPooling1D(), (4, 8)),
            (layers.GlobalAveragePooling1D(), (4, 8)),
            (layers.GlobalMaxPooling2D(), (4, 4, 8)),
            (layers.GlobalAveragePooling2D(), (4, 4, 8)),
        ],
        ids=['Max1D', 'Avg1D', 'Max2D', 'Avg2D', 'GlobalMax1D', 'GlobalAvg1D', 'GlobalMax2D', 'GlobalAvg2D'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class TestReshape:
    @pytest.fixture(
        params=[
            layers.Flatten(),
            layers.Reshape((8, 4)),
            partial(ops.reshape, newshape=(-1, 4)),
        ],
        ids=['Flatten', 'Reshape', 'ops.reshape'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(4, 8)])


class TestBinaryOp:
    @pytest.fixture(
        params=[
            lambda xs: ops.add(*xs),
            lambda xs: ops.subtract(*xs),
            lambda xs: ops.multiply(*xs),
            lambda xs: ops.round(10 / ops.maximum(xs[0], 0.125)),
            lambda xs: ops.round(xs[0] / 4),
            lambda xs: ops.maximum(*xs),
            lambda xs: ops.minimum(*xs),
            lambda xs: ops.concatenate(xs, axis=-1),
        ],
        ids=['add', 'sub', 'mul', 'div_const', 'div', 'max', 'min', 'concat'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,), (8,)])


class TestReduction:
    @pytest.fixture(
        params=[
            pytest.param(partial(ops.sum, axis=-1), id='sum'),
            pytest.param(partial(ops.max, axis=-1), id='max'),
            pytest.param(partial(ops.min, axis=-1), id='min'),
            pytest.param(partial(ops.mean, axis=1), id='mean'),
            pytest.param(partial(ops.prod, axis=1), id='prod'),
            pytest.param(lambda x: ops.all(x < 0, axis=1), id='all'),
            pytest.param(lambda x: ops.any(x < 0, axis=1), id='any'),
            pytest.param(
                partial(ops.count_nonzero, axis=1),
                id='count_nonzero',
                marks=_torch_skip(reason='behavior mismatch: keras/torch transposes count_nonzero across all dims (spurious.T)'),
            ),
        ],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(4, 8)], kif=(1, 2, 2))


class TestAverage:
    @pytest.mark.parametrize(
        'weighted',
        [
            False,
            pytest.param(True, marks=_torch_skip(reason='behavior mismatch: keras/torch ignores axis arg')),
        ],
        ids=['unweighted', 'weighted'],
    )
    def test(self, weighted):
        weights = np.arange(4) + 0.5 if weighted else None
        _run(lambda a: ops.average(a, axis=1, weights=weights), [(4, 8)])


class TestUnaryOp:
    @pytest.fixture(
        params=[
            ops.abs,
            partial(ops.clip, x_min=-4.0, x_max=4.0),
            ops.sign,
            ops.floor,
            ops.ceil,
            ops.round,
        ],
        ids=['abs', 'clip', 'sign', 'floor', 'ceil', 'round'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,)])


class TestContraction:
    @pytest.fixture(
        params=[
            lambda xs: ops.matmul(*xs),
            lambda xs: ops.einsum('...ij,...jk->...ik', *xs),
        ],
        ids=['matmul', 'einsum'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(4, 3), (3, 2)], kif=(1, 2, 4))


class TestDot:
    @pytest.fixture(
        params=[
            (lambda xs: layers.Dot(axes=-1)(xs), [(4, 3), (3,)]),
            (lambda xs: layers.Dot(axes=1)(xs), [(4, 3), (4, 5)]),
            (lambda xs: layers.Dot(axes=(1, 2))(xs), [(4, 3), (5, 4)]),
        ],
        ids=['axes=-1', 'axes=1', 'axes=(1,2)'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shapes = case
        _run(op, shapes, kif=(1, 2, 4))


class TestNoOp:
    def test(self):
        _run(layers.Dropout(0.5), [(8,)])


class TestPad:
    @pytest.fixture(
        params=[
            'constant',
            pytest.param('edge', marks=_torch_skip(reason='keras torch backend does not support edge pad')),
            'reflect',
            pytest.param(
                'symmetric',
                marks=_torch_skip(reason='behavior mismatch: keras/torch maps symmetric -> replicate (edge)'),
            ),
            pytest.param('wrap', marks=_torch_skip(reason='keras torch backend does not support wrap pad')),
        ],
    )
    def op(self, request):
        constant_values = 0 if request.param == 'constant' else None
        return partial(ops.pad, pad_width=((0, 0), (2, 3)), mode=request.param, constant_values=constant_values)

    def test(self, op):
        _run(op, [(5,)])


class TestGetItem:
    @pytest.fixture(
        params=[
            pytest.param(lambda x: x[:, :4], id='slice'),
            pytest.param(
                lambda x: x[:, 2::2, ::-1],
                id='slice_step',
                marks=_torch_skip(reason='torch slicing requires step > 0'),
            ),
            pytest.param(
                lambda x: x[:, ::2, 8:2:-2],
                id='index_array',
                marks=_torch_skip(reason='torch slicing requires step > 0'),
            ),
        ],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8, 8)])


class TestRepeat:
    @pytest.fixture(
        params=[
            layers.RepeatVector(4),
            partial(ops.repeat, repeats=4, axis=1),
        ],
        ids=['RepeatVector', 'repeat'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,)], kif=(1, 2, 4))


class TestNested:
    @pytest.fixture
    def model(self):
        inp = layers.Input(shape=(8,))
        out = layers.Dense(8, activation='relu')(inp)
        model0 = keras.Model(inp, out)
        inp1 = layers.Input(shape=(8,))
        out1 = model0(inp1)
        out1 = model0(out1)
        model1 = keras.Model(inp1, out1)
        return model1

    def test(self, model):
        _run(lambda x: model(x), [(8,)], verbose=True)


class TestUnknownShape:
    @pytest.fixture
    def model(self):
        inp = keras.Input(shape=(None, None))
        out = keras.layers.MaxPool1D(2)(inp)
        model = keras.Model(inp, out)
        return model

    @pytest.mark.parametrize('inp_shape', [(8, 2), (12, 3)])
    def test(self, model, inp_shape):
        kif = (1, 4, 4)
        inputs = FVArray.new((1,) + inp_shape).quantize(*kif).as_new()
        _run(model, [inp_shape], kif=(1, 2, 4), wrap_model=False, inputs=inputs)


class TestSortLike:
    @pytest.fixture(
        params=[
            partial(ops.argmax, axis=-1),
            partial(ops.argmin, axis=-1),
            partial(ops.amax, axis=-1),
            partial(ops.amin, axis=-1),
            partial(ops.sort, axis=-1),
        ],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,)])


class TestMoveAxis:
    def test(self):
        _run(lambda x: ops.moveaxis(x, 1, 2), [(8, 4, 2)])


class TestTranspose:
    def test(self):
        _run(lambda x: ops.transpose(x, (0, 2, 1)), [(8, 4)])


class TestMerge:
    @pytest.fixture(
        params=[
            layers.Add(),
            layers.Concatenate(),
            layers.Multiply(),
            layers.Subtract(),
            layers.Maximum(),
            layers.Minimum(),
            layers.Average(),
        ],
        ids=['Add', 'Concatenate', 'Multiply', 'Subtract', 'Maximum', 'Minimum', 'Average'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        shapes = [(3, 4), (3, 1), (1, 4)]
        if isinstance(op, layers.Concatenate):
            shapes = [(3, 4), (3, 4)]
        elif isinstance(op, layers.Subtract):
            shapes = [(3, 4), (3, 1)]
        elif isinstance(op, layers.Average):
            shapes = [(3, 4), (3, 1), (1, 4), (1, 1)]
        _run(op, shapes, kif=(1, 2, 3))


class TestExtractPatches:
    @pytest.fixture(
        params=[
            partial(ops.image.extract_patches, size=(2, 2), strides=(1, 1), padding='valid'),
            partial(ops.image.extract_patches, size=(2, 2), strides=(1, 1), padding='same'),
            partial(ops.image.extract_patches, size=(2, 1), strides=(1, 2), padding='valid'),
            partial(ops.image.extract_patches, size=(1, 3), strides=(2, 2), padding='same'),
        ],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(3, 4, 5)])


class TestCmp:
    @pytest.fixture(
        params=[
            lambda xs: ops.equal(*xs),
            lambda xs: ops.greater(*xs),
            lambda xs: ops.greater_equal(*xs),
            lambda xs: ops.less(*xs),
            lambda xs: ops.less_equal(*xs),
        ],
        ids=['equal', 'greater', 'greater_equal', 'less', 'less_equal'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,), (8,)], kif=(1, 2, 4))
