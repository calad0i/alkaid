import numpy as np
import pytest

from alkaid.trace import FVArray, FVArrayInput, trace
from alkaid.trace.ops import quantize, relu
from alkaid.types import CombLogic

from .test_ops import OperationTest


@pytest.fixture(autouse=True)
def w8x8():
    return (np.random.randn(8, 8).astype(np.float32) * 32).round() / 32


functions = {
    'einsum0': lambda x, w: np.einsum('...i,...i->...i', x[..., :4], x[..., 4:]),
    'einsum1': lambda x, w: np.einsum('...ij,...jk->...ik', x.reshape(-1, 4, 2), x.reshape(-1, 2, 4)),
    'power': lambda x, w: x**2,
    'cmvm0': lambda x, w: np.einsum('...i,ij->...j', x, w),
    'cmvm1': lambda x, w: np.einsum('...i,ij->...', x, w),
    'cmvm2': lambda x, w: x @ w,
    'cmvm3': lambda x, w: np.einsum('ij,...j->...i', w, x),
    'cmvm_collapsed_left': lambda x, w: np.einsum('ij,...j->...i', w, x * 0 + 1),
    'cmvm_collapsed_right': lambda x, w: (x * 0 + 2) @ w,
    'mvm_collapsed_left': lambda x, w: np.einsum('...i,...i->...i', x * 0 + 3, x),
    'mvm_collapsed_right': lambda x, w: np.einsum('...i,...i->...i', x, x * 0 + 4),
    'mvm_collapsed_all': lambda x, w: np.einsum('...i,...i->...i', x * 0 + 5, x * 0 + 6),
    'maximum': lambda x, w: np.maximum(x[..., None, :], w),
    'minimum': lambda x, w: np.minimum(x[..., None, :], w),
    'amax': lambda x, w: np.amax(x, axis=-1, keepdims=True),
    'amin': lambda x, w: np.amin(x, axis=-1, keepdims=True),
    'umax': lambda x, w: x.max(axis=-1, keepdims=True),
    'umin': lambda x, w: x.min(axis=-1, keepdims=True),
    'relu0': lambda x, w: relu(x),
    'relu1': lambda x, w: relu(x, i=np.array(1)),
    'relu2': lambda x, w: relu(x, f=np.array(1), round_mode='RND'),
    'multi_cadd': lambda x, w: x * 2.0 ** np.arange(-8, 8, 2) + 2 + 3.75,
    'mux0': lambda x, w: np.where(x[..., None] > w, x[..., None], w),
    'lut': lambda x, w: (
        quantize(np.cos(np.sin(x)), 1, 2, 3) if not isinstance(x, FVArray) else quantize(x.apply(np.sin).apply(np.cos), 1, 2, 3)
    ),
    'prod': lambda x, w: np.prod(x[..., :3], axis=-1, keepdims=True),
    'mean': lambda x, w: np.mean(x, axis=-1, keepdims=True),
    'sum': lambda x, w: np.sum(x, axis=-1, keepdims=True),
    'usum': lambda x, w: x.sum(axis=-1, keepdims=True),
    'clip0': lambda x, w: np.clip(x, -1.0, 2.0),
    'clip1': lambda x, w: np.clip(x[..., :4], x[..., 4:8], 1.5),
    'dot0': lambda x, w: np.dot(x, w),
    'dot2': lambda x, w: np.dot(np.array(1.5), np.mean(x, axis=-1, keepdims=True)),
    'where1': lambda x, w: np.where(x - 3 == 0, x * 2, x / 2),
    'where2': lambda x, w: np.where(x != 0, x, -1),
    'where3': lambda x, w: np.where(x >= 1.375, -1, x),
    'where4': lambda x, w: np.where(x[..., :4] <= x[..., 4:], x[..., 4:] + 1, x[..., 4:] - 1),
    'where5': lambda x, w: np.where(x == x * 2, 3, 4),
    'where6': lambda x, w: np.where(w[0] > 0, x * 2, x / 2),
    'any0': lambda x, w: np.any(x, axis=-1, keepdims=True),
    'any1': lambda x, w: np.any((x > 0).reshape(x.shape[:-1] + (2, 4)), axis=-2, keepdims=True),
    'all0': lambda x, w: np.all(x, axis=-1, keepdims=True),
    'all1': lambda x, w: np.all((x > 0).reshape(x.shape[:-1] + (2, 4)), axis=-2, keepdims=True),
    'floor': lambda x, w: np.floor(x),
    'sign': lambda x, w: np.sign(x),
    'signbit': lambda x, w: np.signbit(x),
    'const_propagation': lambda x, w: x * 3 - 2 + 1.5 - 0.5 * x - 0.25 * x + 0.5,
    'argmax': lambda x, w: np.argmax(x, axis=-1, keepdims=False),
    'argmin': lambda x, w: np.argmin(x, axis=-1, keepdims=True),
    'argmax_4': lambda x, w: np.argmax(x[..., :4], axis=-1, keepdims=False),
    'argmin_4': lambda x, w: np.argmin(x[..., :4], axis=-1, keepdims=True),
    'round': lambda x, w: np.round(x),
    'ceil': lambda x, w: np.ceil(x),
    'average0': lambda x, w: np.average(x, axis=-1, keepdims=True),
    'average1': lambda x, w: np.average(x, axis=-1, weights=np.arange(0.5, 8.5), keepdims=True),
    'div': lambda x, w: np.round(10 / np.maximum(x, 0.125)),
    'count_nonzero': lambda x, w: np.count_nonzero(x, axis=-1, keepdims=True),
    'tanh_floor': lambda x, w: np.floor(np.tanh(x) * 32),
    'tanh_ceil': lambda x, w: np.ceil(np.tanh(x) * 32),
    'tanh_rnd': lambda x, w: np.round(np.tanh(x) * 32),
    'exp_floor': lambda x, w: np.floor(np.exp(x * 0.03125)),
    'rnd_pwr': lambda x, w: np.round(np.abs(x) ** 3.14),
    'square': lambda x, w: np.square(x),
    'ufunc_any': lambda x, w: (x[..., :2] > 0).any(axis=-1),
    'ufunc_all': lambda x, w: (x[..., :2] > 0).all(axis=-1),
    'retard_op_fuse': lambda x, w: np.round(1.14514 / ((3.14 * (np.pi + (1 - np.pi * (-np.sin(x)) ** 2 + 1)) - 2.7) / 1.919810)),
}


class TestOperations(OperationTest):
    @pytest.fixture(params=list(functions.keys()))
    def op_func(self, request, w8x8: np.ndarray, test_data, inp):
        return lambda x: functions[request.param](x, w8x8)


class TestSort(OperationTest):
    @pytest.fixture(params=['batcher', 'bitonic'])
    def kind(self, request):
        return request.param

    @pytest.fixture(params=[8, 7, 4, 3])
    def size(self, request):
        return request.param

    @pytest.fixture()
    def op_func(self, kind, size):
        def sort_fn(x):
            _kind = kind
            if not isinstance(x, FVArray):
                _kind = 'quicksort'
            if size >= 4:
                return np.sort(x[..., :size], axis=-1, kind=_kind)  # type: ignore
            else:
                x = x.reshape(x.shape[:-1] + (4, 2))
                x = np.sort(x, axis=-2, kind=_kind)[..., :size, :]
                return x

        return sort_fn


class TestArgsort(OperationTest):
    @pytest.fixture()
    def op_func(self):
        def argsort_fn(x):
            if isinstance(x, FVArray):
                return x[..., :4][np.argsort(x[..., 4:])]  # type: ignore
            else:
                return np.apply_along_axis(lambda v: v[:4][np.argsort(v[4:])], -1, x)

        return argsort_fn

    def test_eq(self, op_func, test_data: np.ndarray, comb: CombLogic, n_samples: int):
        traced_out = comb.predict(test_data, n_threads=1)
        test_data = quantize(test_data, *comb.inp_kifs)
        expected_out = quantize(op_func(test_data).reshape(n_samples, -1), 1, 12, 12)

        # bitonic sort is unstable - check with skipping tied keys
        keys = test_data[:, 4:]

        sorted_keys = np.sort(keys, axis=-1)
        mask_has_tie = np.any(np.diff(sorted_keys, axis=-1) == 0, axis=-1)
        np.testing.assert_equal(traced_out[~mask_has_tie], expected_out[~mask_has_tie])

        traced_out_tied = traced_out[mask_has_tie]
        expected_out_tied = traced_out[mask_has_tie]
        for i in range(len(traced_out_tied)):
            hw = traced_out_tied[i]
            exp = expected_out_tied[i]

            for k in np.unique(keys[i]):
                pos = np.where(sorted_keys[i] == k)[0]
                if len(pos) == 1:  # unique
                    np.testing.assert_equal(hw[pos], exp[pos], err_msg=f'sample {i}, key={k}')
                else:
                    # Tied
                    np.testing.assert_array_equal(
                        np.sort(hw[pos]),
                        np.sort(exp[pos]),
                        err_msg=f'sample {i}: tied-key group key={k} at positions {pos}',
                    )

        symbolic_out = []
        for x in test_data[:100]:
            x = list(map(float, x))
            r = comb(x, quantize=True)
            symbolic_out.append(r)
        symbolic_out = np.array(symbolic_out, dtype=np.float64)
        np.testing.assert_equal(symbolic_out, traced_out[:100])


@pytest.mark.parametrize('thres', [0.0, 0.5, 1.0])
def test_offload(thres):
    w = (np.random.randn(8, 8).astype(np.float32) * 10).round() / 16

    def offload_fn(weights, vector):
        return np.random.rand(*np.shape(weights)) > thres

    inp = FVArrayInput((2, 8), solver_options={'offload_fn': offload_fn}).quantize(1, 4, 3)
    out = inp @ w
    comb = trace(inp, out)

    data_in = np.random.rand(10000, 2, 8).astype(np.float32) * 64 - 32
    traced_out = comb.predict(data_in, n_threads=1)
    expected_out = (quantize(data_in, *inp.kif) @ w).reshape(10000, -1)
    np.testing.assert_equal(traced_out, expected_out)


class TestHistogram(OperationTest):
    @pytest.fixture(
        params=[
            {'bins': 4, 'range': (-2.0, 2.0)},
            {'bins': np.array([-2.0, -1.0, 0.0, 1.0, 2.0])},
            {'bins': 1, 'range': (-1.0, 1.0)},
            {'bins': 2, 'range': (-1.0, 1.0)},
            {'bins': np.array([-3.0, -0.5, 0.5, 3.0])},
        ]
    )
    def edge_config(self, request):
        return request.param

    @pytest.fixture(params=[False, True])
    def weighting(self, request):
        return request.param

    @pytest.fixture()
    def op_func(self, edge_config, weighting):

        if weighting:
            w = np.array([1.0, 0.5, 2.0, 1.5, 0.25, 1.0, 3.0, 0.75])
        else:
            w = None

        kwargs = {**edge_config, 'weights': w}

        def hist_fn(x):
            if isinstance(x, FVArray):
                return np.histogram(x, **kwargs)[0]
            return np.apply_along_axis(lambda v: np.histogram(v, **kwargs)[0], -1, x)

        return hist_fn


def test_empty_histogram():
    from alkaid.trace.fixed_variable import HWConfig

    hwconf = HWConfig(1, 1, -1)
    fva = FVArray(np.array([], dtype=object), hwconf=hwconf)
    counts, edges = np.histogram(fva, bins=3, range=(0.0, 3.0))
    assert counts.shape == (3,)
    assert all(float(c.low) == 0.0 for c in np.asarray(counts).ravel())


class TestSearchsorted(OperationTest):
    """Searchsorted with parametrized sorter, edge type, and side."""

    @pytest.fixture(params=['thermometer', 'bsearch'])
    def sorter(self, request):
        return request.param

    @pytest.fixture(params=['const-fva', 'fva-fva', 'fva-const'])
    def _type(self, request):
        return request.param

    @pytest.fixture(params=['left', 'right'])
    def side(self, request):
        return request.param

    @pytest.fixture()
    def inp(self, _type) -> FVArray:
        n = 16
        b = np.random.randint(0, 9, size=n)
        i = np.random.randint(-8, 8, size=n)
        k = np.random.randint(0, 2, size=n)
        return FVArray.from_kif(k, i, b - i)

    @pytest.fixture()
    def op_func(self, sorter, _type, side, w8x8):
        if _type == 'const-fva':
            edges = np.sort(w8x8.ravel())

            def fn(x):  # type: ignore
                if isinstance(x, FVArray):
                    return np.searchsorted(edges, x, side=side, sorter=sorter)
                return np.array([np.searchsorted(edges, v, side=side) for v in x])

        elif _type == 'fva-fva':
            # FVA edges: first 8 elements are edges, last 8 are values
            def fn(x):
                a, v = np.sort(x[..., :8], axis=-1), x[..., 8:]
                if isinstance(x, FVArray):
                    return np.searchsorted(a, v, side=side, sorter=sorter)
                sx = np.sort(x[:, :8], axis=-1)
                return np.stack([np.searchsorted(_sx, _x, side=side) for _sx, _x in zip(sx, x[:, 8:])])
        else:
            assert _type == 'fva-const'
            v = w8x8.ravel()

            def fn(x):
                a = np.sort(x, axis=-1)
                if isinstance(x, FVArray):
                    return np.searchsorted(a, v, side=side, sorter=sorter)
                sx = np.sort(x, axis=-1)
                return np.stack([np.searchsorted(_sx, v, side=side) for _sx in sx])

        return fn
