from functools import lru_cache, partial

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from alkaid.converter import trace_model
from alkaid.trace import FVArray, trace


def _qdata(shape, kif, seed=0) -> np.ndarray:
    k, i, f = kif
    rng = np.random.default_rng(seed)
    hi = 2.0**i - 2.0**-f
    lo = -(2.0**i) * k
    raw = rng.uniform(lo, hi, size=shape).astype(np.float32)
    step = 2.0**f
    return np.floor(raw * step).astype(np.float32) / step


def _perturb_weights(model: nn.Module, fbits: int = 2, seed: int = 42):
    rng = np.random.default_rng(seed)
    step = 2.0**fbits
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 0:
                continue
            r = rng.standard_normal(tuple(p.shape)).astype(np.float32)
            p.copy_(torch.tensor(np.round(r * step) / step))
        for name, b in model.named_buffers():
            if 'num_batches_tracked' in name:
                continue
            if b.ndim == 0:
                continue
            r = rng.standard_normal(tuple(b.shape)).astype(np.float32)
            val = np.round(r * step) / step
            if name.endswith('running_var'):
                val = np.maximum(val, 0.25).astype(np.float32)
            b.copy_(torch.tensor(val))


def _make_input(shapes, kif):
    k, i, f = kif
    out = []
    for s in shapes:
        shp = (1,) + tuple(s)
        out.append(
            FVArray.from_kif(
                np.full(shp, k, dtype=np.int8),
                np.full(shp, i, dtype=np.int8),
                np.full(shp, f, dtype=np.int8),
            )
        )
    return out[0] if len(out) == 1 else tuple(out)


class _WrapBase(nn.Module):
    """Arity-generic wrapper around a stateless callable or a single submodule.

    ``_Wrap(n_inputs)`` returns a subclass with ``forward(self, x0, ..., x_{n-1})``
    so fx sees exactly ``n_inputs`` placeholders; the call becomes ``self._fn(*args)``.
    An nn.Module ``fn`` is also stored under ``self._fn`` as a submodule so fx can
    dispatch through its ``call_module`` path.

    This wrapper is NOT usable for tests that need to hold a submodule AND apply
    pre-/post-processing around it (e.g. ``torch.round(activation(x) * 16)``) —
    fx rejects a closed-over module with "not installed as a submodule", so those
    tests keep dedicated ``nn.Module`` subclasses (see ``_Act``, ``_EmbeddingWrap``).
    Tests that need ``nn.Parameter``s similarly keep their own subclasses.
    """

    def __init__(self, fn):
        super().__init__()
        self._fn = fn  # auto-registers as submodule if fn is an nn.Module


@lru_cache(maxsize=None)
def _Wrap(n_inputs: int) -> type[_WrapBase]:
    argnames = [f'x{i}' for i in range(n_inputs)]
    src = f'def forward(self, {", ".join(argnames)}):\n    return self._fn({", ".join(argnames)})\n'
    ns: dict = {}
    exec(src, ns)
    return type(f'_Wrap{n_inputs}', (_WrapBase,), {'forward': ns['forward']})


@torch.no_grad()
def _run(
    op,
    shapes,
    kif=(1, 4, 4),
    n=2048,
    perturb_weights=True,
    hook_model=None,
    hook_data=None,
    wrap_module=True,
    **kwargs,
):
    if wrap_module:
        model = _Wrap(len(shapes))(op).eval()
    else:
        model = op.eval() if isinstance(op, nn.Module) else op

    if perturb_weights:
        _perturb_weights(model)
    if hook_model is not None:
        hook_model(model)

    datas = [_qdata((n,) + tuple(s), kif, seed=i) for i, s in enumerate(shapes)]
    if hook_data is not None:
        datas = hook_data(datas)

    inputs = _make_input(shapes, kif)
    _kwargs = dict(kwargs)
    if not isinstance(inputs, tuple):
        _kwargs['inputs'] = inputs
    else:
        _kwargs['inputs'] = inputs
    trace_inp, trace_out = trace_model(model, inputs_kif=kif, framework='torch', **_kwargs)
    comb = trace(trace_inp, trace_out)

    torch_inputs = [torch.tensor(d) for d in datas]
    expected = model(*torch_inputs).detach().cpu().numpy()

    data_comb = np.concatenate([x.reshape((n, -1)) for x in datas], axis=1)
    actual = comb.predict(data_comb).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLinear:
    @pytest.fixture(
        params=[
            (nn.Linear(8, 4, bias=True), (8,)),
            (nn.Linear(8, 4, bias=False), (8,)),
        ],
        ids=['bias', 'nobias'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class TestConv:
    @pytest.fixture(
        params=[
            (nn.Conv1d(3, 4, 3, padding=0), (3, 16)),
            (nn.Conv1d(3, 4, 3, padding='same'), (3, 16)),
            (nn.Conv2d(3, 4, 3, padding=0), (3, 8, 8)),
            (nn.Conv2d(3, 4, 3, padding='same'), (3, 8, 8)),
            (nn.Conv2d(4, 4, 3, padding=0, groups=2), (4, 8, 8)),
            (nn.Conv3d(3, 4, 3, padding=0), (3, 4, 4, 4)),
        ],
        ids=['Conv1D', 'Conv1D[same]', 'Conv2D', 'Conv2D[same]', 'Conv2D[groups]', 'Conv3D'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class TestBatchNorm:
    @pytest.fixture(
        params=[
            (nn.BatchNorm1d(8), (8,)),
            (nn.BatchNorm1d(4), (4, 6)),
            (nn.BatchNorm2d(4), (4, 6, 6)),
        ],
        ids=['BN1d[rank2]', 'BN1d[rank3]', 'BN2d'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case

        def hook_model(model):
            # scale = gamma / sqrt(var + eps) == 1 when gamma = sqrt(var + eps) (float32-exact)
            # offset = beta - mean * scale = beta - mean
            with torch.no_grad():
                ch = model._fn.num_features
                eps = np.float32(model._fn.eps)
                var = np.float32(1.0)
                gamma = float(np.sqrt(var + eps))
                model._fn.weight.copy_(torch.full((ch,), gamma))
                model._fn.bias.copy_(torch.full((ch,), 0.5))
                model._fn.running_mean.copy_(torch.full((ch,), 0.25))
                model._fn.running_var.copy_(torch.full((ch,), float(var)))

        _run(op, [shape], kif=(1, 2, 4), hook_model=hook_model, perturb_weights=False)


class TestReLU:
    def test(self):
        _run(nn.ReLU(), [(8,)])


class TestLeakyReLU:
    def test(self):
        _run(nn.LeakyReLU(0.25), [(8,)])


class TestPReLU:
    @pytest.fixture(params=[None, 4], ids=['scalar', 'per-channel'])
    def case(self, request):
        n_ch = request.param
        if n_ch is None:
            return nn.PReLU(), (8,)
        return nn.PReLU(n_ch), (n_ch, 8)

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class TestSimpleActivation:
    @pytest.fixture(
        params=[
            nn.Sigmoid,
            nn.Tanh,
            nn.SiLU,
            nn.GELU,
            nn.ELU,
            nn.SELU,
            nn.Softplus,
            nn.Softsign,
            nn.Hardsigmoid,
            nn.Hardswish,
            nn.Hardtanh,
            nn.ReLU6,
            nn.LogSigmoid,
        ]
    )
    def cls(self, request):
        return request.param

    def test(self, cls):
        # quantize after activation to allow exact equality
        class _Act(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.a = c()

            def forward(self, x):
                return torch.round(self.a(x) * 16)

        _run(_Act(cls), [(8,)], kif=(1, 2, 4), perturb_weights=False)


class TestFunctionalActivation:
    @pytest.fixture(
        params=[
            F.relu,
            F.sigmoid,
            F.tanh,
            F.silu,
            F.gelu,
            F.elu,
            F.selu,
            F.softplus,
            F.hardsigmoid,
            F.hardswish,
            F.hardtanh,
            F.relu6,
        ]
    )
    def fn(self, request):
        return request.param

    def test(self, fn):
        _run(lambda x: torch.round(fn(x) * 16), [(8,)], kif=(1, 2, 4), perturb_weights=False)


class TestFlatten:
    def test(self):
        _run(nn.Flatten(), [(3, 4)])


class TestUnflatten:
    def test(self):
        _run(nn.Unflatten(dim=1, unflattened_size=(2, 4)), [(8,)])


class TestReshapeMethod:
    def test(self):
        _run(lambda x: x.reshape(1, -1), [(3, 4)])


class TestPermute:
    def test(self):
        _run(lambda x: torch.permute(x, (0, 2, 1)), [(4, 3)])


class TestMethodPermute:
    def test(self):
        _run(lambda x: x.permute(0, 2, 1), [(4, 3)])


class TestTranspose:
    def test(self):
        _run(lambda x: torch.transpose(x, 1, 2), [(4, 3, 2)])


class TestSqueezeUnsqueeze:
    def test(self):
        _run(lambda x: torch.squeeze(torch.unsqueeze(x, 1), 1), [(4,)])


class TestPool1D:
    @pytest.fixture(
        params=[
            nn.MaxPool1d(2),
            nn.AvgPool1d(2),
        ],
        ids=['max', 'avg'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(4, 8)])


class TestPool2D:
    @pytest.fixture(
        params=[
            nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            nn.MaxPool2d(2, padding=1),
            nn.AvgPool2d(2, padding=1, count_include_pad=False),
        ],
        ids=['max', 'avg', 'max[pad]', 'avg[pad,novalidcnt]'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(4, 6, 6)])


class TestPool3D:
    def test(self):
        _run(nn.MaxPool3d(2), [(2, 4, 4, 4)])


class TestGlobalPool:
    @pytest.fixture(
        params=[
            nn.AdaptiveMaxPool1d(1),
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool3d(1),
            nn.AdaptiveAvgPool3d(1),
        ],
        ids=['max1d', 'avg1d', 'max2d', 'avg2d', 'max3d', 'avg3d'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        # Use shapes whose spatial volume is a power of 2 so avg=sum/2^k stays bit-exact.
        if '3d' in op.__class__.__name__.lower():
            _run(op, [(2, 2, 2, 2)])
        elif '2d' in op.__class__.__name__.lower():
            _run(op, [(2, 4, 4)])
        else:
            _run(op, [(2, 4)])


class TestBinaryOp:
    @pytest.fixture(
        params=[
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: torch.round(x / 4),
            lambda x, y: torch.maximum(x, y),
            lambda x, y: torch.minimum(x, y),
            lambda x, y: torch.cat([x, y], dim=-1),
            lambda x, y: torch.stack([x, y], dim=0).sum(0),
        ],
        ids=['add', 'sub', 'mul', 'div_const', 'max', 'min', 'cat', 'stack'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,), (8,)])


class TestReduction:
    @pytest.fixture(
        params=[
            ('sum', lambda x: torch.sum(x, dim=-1), (4, 8), (1, 2, 2)),
            ('mean', lambda x: torch.mean(x, dim=-1), (4, 8), (1, 2, 2)),
            # prod magnitudes grow geometrically; keep the reduced axis short.
            ('prod', lambda x: torch.prod(x, dim=-1), (4, 3), (1, 1, 2)),
            ('amax', lambda x: torch.amax(x, dim=-1), (4, 8), (1, 2, 2)),
            ('amin', lambda x: torch.amin(x, dim=-1), (4, 8), (1, 2, 2)),
            ('all', lambda x: torch.all(x < 0, dim=-1), (4, 8), (1, 2, 2)),
            ('any', lambda x: torch.any(x < 0, dim=-1), (4, 8), (1, 2, 2)),
        ],
        ids=lambda v: v[0],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        _, op, shape, kif = case
        _run(op, [shape], kif=kif)


class TestReductionMethod:
    @pytest.fixture(
        params=[
            lambda x: x.sum(dim=-1),
            lambda x: x.mean(dim=-1),
            lambda x: x.amax(dim=-1),
        ],
        ids=['sum', 'mean', 'amax'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(4, 8)], kif=(1, 2, 2))


class TestUnaryOp:
    @pytest.fixture(
        params=[
            torch.abs,
            partial(torch.clamp, min=-4.0, max=4.0),
            torch.sign,
            torch.floor,
            torch.ceil,
            torch.round,
        ],
        ids=['abs', 'clip', 'sign', 'floor', 'ceil', 'round'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,)])


class TestMatmul:
    @pytest.fixture(
        params=[
            lambda x, y: torch.matmul(x, y),
            lambda x, y: x @ y,
        ],
        ids=['matmul', 'operator'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(4, 3), (3, 2)], kif=(1, 2, 4))


class TestEinsum:
    def test(self):
        _run(lambda x, y: torch.einsum('...ij,...jk->...ik', x, y), [(4, 3), (3, 2)], kif=(1, 2, 4))


class TestSortLike:
    @pytest.fixture(
        params=[
            lambda x: torch.argmax(x, dim=-1),
            lambda x: torch.argmin(x, dim=-1),
            lambda x: torch.amax(x, dim=-1),
            lambda x: torch.amin(x, dim=-1),
        ],
        ids=['argmax', 'argmin', 'amax', 'amin'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,)])


class TestPad:
    def test(self):
        _run(lambda x: F.pad(x, (2, 3), mode='constant', value=0), [(5,)])


class TestGetItem:
    def test(self):
        _run(lambda x: x[:, :4], [(8, 8)])


class TestNoOp:
    @pytest.fixture(
        params=[nn.Dropout(0.5), nn.Identity()],
        ids=['dropout', 'identity'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,)])


class TestNested:
    @pytest.fixture
    def model(self):
        return nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def test(self, model):
        _run(model, [(8,)], wrap_module=True)


class TestCmp:
    @pytest.fixture(
        params=[
            lambda x, y: x == y,
            lambda x, y: x > y,
            lambda x, y: x >= y,
            lambda x, y: x < y,
            lambda x, y: x <= y,
        ],
        ids=['eq', 'gt', 'ge', 'lt', 'le'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,), (8,)], kif=(1, 2, 4))


# ---------------------------------------------------------------------------
# Functional-API modules: exercise torch.nn.functional call_function nodes,
# get_attr nodes (for weights), and the _replay_getattr for tensor attributes.
# ---------------------------------------------------------------------------


class _FLinear(nn.Module):
    def __init__(self, in_f, out_f, with_bias=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(out_f, in_f))
        self.b = nn.Parameter(torch.randn(out_f)) if with_bias else None

    def forward(self, x):
        return F.linear(x, self.w, self.b)


class TestFunctionalLinear:
    @pytest.mark.parametrize('bias', [True, False], ids=['bias', 'nobias'])
    def test(self, bias):
        _run(_FLinear(8, 4, bias), [(8,)])


class _FConv(nn.Module):
    def __init__(self, rank, ch_in, ch_out, k, stride=1, padding=0, groups=1, dilation=1):
        super().__init__()
        shape = (ch_out, ch_in // groups) + (k,) * rank
        self.w = nn.Parameter(torch.randn(*shape))
        self.b = nn.Parameter(torch.randn(ch_out))
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

    def forward(self, x):
        fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[self.rank]
        return fn(x, self.w, self.b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


class TestFunctionalConv:
    @pytest.fixture(
        params=[
            (_FConv(1, 3, 4, 3, padding=1), (3, 10)),
            (_FConv(2, 3, 4, 3, padding=1), (3, 6, 6)),
            (_FConv(2, 4, 4, 3, groups=2), (4, 6, 6)),
            (_FConv(3, 2, 4, 3, padding=1), (2, 4, 4, 4)),
        ],
        ids=['F.conv1d', 'F.conv2d', 'F.conv2d[groups]', 'F.conv3d'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class _FBatchNorm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w = nn.Parameter(torch.ones(ch))
        self.b = nn.Parameter(torch.zeros(ch))
        self.register_buffer('running_mean', torch.zeros(ch))
        self.register_buffer('running_var', torch.ones(ch))

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.w, self.b, eps=1e-5)  # type: ignore


class TestFunctionalBatchNorm:
    def test(self):
        ch = 4
        model = _FBatchNorm(ch)
        eps = np.float32(1e-5)
        var = np.float32(1.0)
        # Pick gamma so scale = gamma / sqrt(var + eps) is exactly 1.0 in float32.
        gamma = float(np.sqrt(var + eps))
        with torch.no_grad():
            model.w.copy_(torch.full((ch,), gamma))
            model.b.copy_(torch.full((ch,), 0.5))
            model.running_mean.copy_(torch.full((ch,), 0.25))  # type: ignore
            model.running_var.copy_(torch.full((ch,), float(var)))  # type: ignore
        _run(model, [(ch,)], kif=(1, 2, 4), perturb_weights=False)


class TestFunctionalPool:
    @pytest.fixture(
        params=[
            (partial(F.max_pool1d, kernel_size=2), (3, 8)),
            (partial(F.max_pool2d, kernel_size=2), (3, 4, 4)),
            (partial(F.max_pool3d, kernel_size=2), (2, 4, 4, 4)),
            (partial(F.avg_pool1d, kernel_size=2), (3, 8)),
            (partial(F.avg_pool2d, kernel_size=2), (3, 4, 4)),
            (partial(F.avg_pool3d, kernel_size=2), (2, 4, 4, 4)),
            (partial(F.avg_pool2d, kernel_size=2, padding=1, count_include_pad=False), (3, 4, 4)),
            (partial(F.adaptive_max_pool1d, output_size=1), (3, 8)),
            (partial(F.adaptive_avg_pool2d, output_size=1), (3, 4, 4)),
            (partial(F.adaptive_max_pool3d, output_size=1), (2, 2, 2, 2)),
            (partial(F.adaptive_avg_pool3d, output_size=1), (2, 2, 2, 2)),
        ],
        ids=[
            'F.max_pool1d',
            'F.max_pool2d',
            'F.max_pool3d',
            'F.avg_pool1d',
            'F.avg_pool2d',
            'F.avg_pool3d',
            'F.avg_pool2d[pad,valcnt]',
            'F.adaptive_max_pool1d',
            'F.adaptive_avg_pool2d',
            'F.adaptive_max_pool3d',
            'F.adaptive_avg_pool3d',
        ],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class TestFunctionalDropout:
    @pytest.fixture(
        params=[F.dropout, F.dropout1d, F.alpha_dropout],
        ids=['dropout', 'dropout1d', 'alpha_dropout'],
    )
    def fn(self, request):
        return request.param

    def test(self, fn):
        _run(lambda x: fn(x, p=0.5, training=False), [(8,)])


class TestFunctionalLeakyReLU:
    def test(self):
        _run(lambda x: F.leaky_relu(x, negative_slope=0.125), [(8,)])


class _FPReLU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(ch) * 0.25)

    def forward(self, x):
        return F.prelu(x, self.alpha)


class TestFunctionalPReLU:
    @pytest.mark.parametrize('ch', [1, 4], ids=['scalar', 'per-channel'])
    def test(self, ch):
        if ch == 1:
            _run(_FPReLU(1), [(8,)])
        else:
            _run(_FPReLU(ch), [(ch, 6)])


class TestFunctionalElu:
    def test(self):
        # non-default alpha exercises the explicit kwargs path
        _run(lambda x: torch.round(F.elu(x, alpha=0.5) * 16), [(8,)], kif=(1, 2, 4), perturb_weights=False)


class TestFunctionalHardtanh:
    def test(self):
        _run(lambda x: F.hardtanh(x, min_val=-2.0, max_val=2.0), [(8,)])


class TestPadModes:
    @pytest.fixture(
        params=[
            ('constant', partial(F.pad, pad=(1, 2), mode='constant', value=0)),
            ('replicate', partial(F.pad, pad=(1, 2), mode='replicate')),
            ('reflect', partial(F.pad, pad=(1, 2), mode='reflect')),
            ('circular', partial(F.pad, pad=(1, 2), mode='circular')),
        ],
        ids=lambda v: v[0],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        _, op = case
        _run(op, [(4, 6)])


# ---------------------------------------------------------------------------
# Tensor-level ops with multi-value returns or odd kwargs
# ---------------------------------------------------------------------------


class TestSplit:
    def test(self):
        # torch.split returns a tuple; combine them back to a single FVArray.
        _run(lambda x: torch.split(x, 3, dim=-1)[0] + torch.split(x, 3, dim=-1)[1], [(6,)])


class TestChunk:
    def test(self):
        _run(lambda x: torch.chunk(x, 2, dim=-1)[0] - torch.chunk(x, 2, dim=-1)[1], [(6,)])


class TestWhere:
    def test(self):
        _run(lambda x, y: torch.where(x > 0, x, y), [(8,), (8,)], kif=(1, 2, 4))


class TestFunctionalFlatten:
    def test(self):
        _run(lambda x: torch.flatten(x, start_dim=1), [(3, 4)])


class TestFunctionalSqueeze:
    def test(self):
        # no-dim squeeze
        _run(lambda x: torch.squeeze(torch.unsqueeze(x, 1)), [(4,)])


class TestMoveaxisFn:
    def test(self):
        _run(lambda x: torch.moveaxis(x, 1, 2), [(2, 3, 4)])


class TestRepeatInterleave:
    def test(self):
        _run(lambda x: torch.repeat_interleave(x, 2, dim=-1), [(8,)], kif=(1, 2, 2))


class TestTile:
    def test(self):
        _run(lambda x: torch.tile(x, (1, 2)), [(4,)], kif=(1, 2, 2))


class TestMaxWithDim:
    def test(self):
        # torch.max(x, dim=...) returns a namedtuple; .values pulls the tensor.
        _run(lambda x: torch.max(x, dim=-1).values, [(4, 8)], kif=(1, 2, 2))


class TestMinWithDim:
    def test(self):
        _run(lambda x: torch.min(x, dim=-1).values, [(4, 8)], kif=(1, 2, 2))


class TestSortDescending:
    def test(self):
        _run(lambda x: torch.sort(x, dim=-1, descending=True).values, [(6,)])


# ---------------------------------------------------------------------------
# call_method nodes not covered by the shared _functional_map
# ---------------------------------------------------------------------------


class TestMethodArgmax:
    def test(self):
        _run(lambda x: x.argmax(dim=-1), [(8,)])


class TestMethodArgmin:
    def test(self):
        _run(lambda x: x.argmin(dim=-1), [(8,)])


class TestMethodClamp:
    def test(self):
        _run(lambda x: x.clamp(min=-2.0, max=2.0), [(8,)])


class TestMethodClip:
    def test(self):
        _run(lambda x: x.clip(-1.0, 1.0), [(8,)])


class TestMethodExpand:
    def test(self):
        # input is batched to (1, 1, 4); expand to (1, 3, 4)
        _run(lambda x: x.unsqueeze(1).expand(-1, 3, -1), [(4,)])


class TestMethodRepeat:
    def test(self):
        # batched input is (1, 3, 4); tile both channel and spatial dims
        _run(lambda x: x.repeat(1, 2, 1), [(3, 4)])


class TestMethodT:
    def test(self):
        # .T via getattr builtin. Use .mT to avoid the torch-deprecation warning
        # when ndim != 2 (batched input is 3D).
        _run(lambda x: x.mT, [(3, 4)])


class TestMethodSigmoid:
    def test(self):
        _run(lambda x: torch.round(x.sigmoid() * 16), [(8,)], kif=(1, 2, 4), perturb_weights=False)


class TestMethodView:
    def test(self):
        _run(lambda x: x.view(-1, 2), [(2, 4)])


class TestMethodContiguous:
    def test(self):
        _run(lambda x: x.contiguous().reshape(-1), [(2, 4)])


class TestMethodPermuteArgs:
    def test(self):
        # variadic permute (positional dims, not a tuple)
        _run(lambda x: x.permute(0, 2, 1), [(4, 3)])


class TestMethodTransposeArgs:
    def test(self):
        _run(lambda x: x.transpose(1, 2), [(4, 3, 2)])


class TestMethodSqueezeDim:
    def test(self):
        _run(lambda x: x.unsqueeze(1).squeeze(1), [(4,)])


class TestMethodFlatten:
    def test(self):
        _run(lambda x: x.flatten(start_dim=1), [(3, 4)])


class TestMethodClone:
    def test(self):
        _run(lambda x: x.clone() + x.detach(), [(8,)])


class TestMethodTo:
    def test(self):
        # .to / .float are no-ops in replay (dtype metadata only)
        _run(lambda x: x.to(torch.float32).float(), [(8,)])


class TestMethodSum:
    def test(self):
        _run(lambda x: x.sum(dim=-1, keepdim=True), [(4, 8)], kif=(1, 2, 2))


class TestMethodMax:
    def test(self):
        # per-sample max over the non-batch dim
        _run(lambda x: x.max(dim=-1).values, [(8,)])


class TestCountNonzero:
    def test(self):
        # kif=(0, ...) means unsigned; negative values never appear so count_nonzero
        # stays well-defined and bit-exact.
        _run(lambda x: torch.count_nonzero(x, dim=-1), [(4, 6)], kif=(1, 2, 0))


# ---------------------------------------------------------------------------
# Binary-op tensor-callable paths (torch.add / torch.sub / torch.mul / torch.div)
# ---------------------------------------------------------------------------


class TestTorchBinary:
    @pytest.fixture(
        params=[
            lambda x, y: torch.add(x, y),
            lambda x, y: torch.sub(x, y),
            lambda x, y: torch.mul(x, y),
            lambda x, y: torch.round(torch.div(x, 4)),
        ],
        ids=['add', 'sub', 'mul', 'div_const'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(8,), (8,)])


# ---------------------------------------------------------------------------
# More GetItem patterns
# ---------------------------------------------------------------------------


class TestGetItemVariants:
    @pytest.fixture(
        params=[
            lambda x: x[:, 1],
            lambda x: x[..., :4],
            lambda x: x[:, None, :],
        ],
        ids=['int', 'ellipsis', 'newaxis'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(4, 8)])


# ---------------------------------------------------------------------------
# Stress: shared submodule reused twice (exercises MaybeRename collision path)
# ---------------------------------------------------------------------------


class _Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.ReLU()

    def forward(self, x):
        return self.shared(self.shared(x))


class TestSharedSubmodule:
    def test(self):
        _run(_Shared(), [(8,)])


# ---------------------------------------------------------------------------
# Padding modules (nn.{Zero,Constant,Reflection,Replication,Circular}Pad*d)
# ---------------------------------------------------------------------------


class TestPadModules:
    @pytest.fixture(
        params=[
            (nn.ZeroPad1d(2), (3, 8)),
            (nn.ZeroPad2d((1, 2, 2, 1)), (3, 6, 6)),
            (nn.ZeroPad3d(1), (2, 4, 4, 4)),
            (nn.ConstantPad1d(2, 0.5), (3, 8)),
            (nn.ConstantPad2d((1, 2, 2, 1), -0.25), (3, 6, 6)),
            (nn.ReflectionPad1d(2), (3, 8)),
            (nn.ReflectionPad2d(1), (3, 6, 6)),
            (nn.ReplicationPad1d(2), (3, 8)),
            (nn.ReplicationPad2d(1), (3, 6, 6)),
            (nn.ReplicationPad3d(1), (2, 4, 4, 4)),
            (nn.CircularPad1d(2), (3, 8)),
            (nn.CircularPad2d(1), (3, 6, 6)),
        ],
        ids=[
            'Zero1d',
            'Zero2d',
            'Zero3d',
            'Const1d',
            'Const2d',
            'Refl1d',
            'Refl2d',
            'Repl1d',
            'Repl2d',
            'Repl3d',
            'Circ1d',
            'Circ2d',
        ],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


# ---------------------------------------------------------------------------
# Upsampling (nearest) and pixel shuffle
# ---------------------------------------------------------------------------


class TestUpsample:
    @pytest.fixture(
        params=[
            (nn.Upsample(scale_factor=2, mode='nearest'), (3, 4, 4)),
            (nn.Upsample(size=(8, 8), mode='nearest'), (3, 4, 4)),
            (nn.UpsamplingNearest2d(scale_factor=(2, 3)), (3, 4, 4)),
        ],
        ids=['scale', 'size', 'nearest2d'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class TestFunctionalInterpolate:
    @pytest.fixture(
        params=[
            partial(F.interpolate, scale_factor=2, mode='nearest'),
            partial(F.interpolate, scale_factor=(2, 3), mode='nearest'),
            partial(F.interpolate, size=(8, 8), mode='nearest'),
            partial(F.interpolate, size=8, mode='nearest'),
        ],
        ids=['scale-int', 'scale-tuple', 'size-tuple', 'size-int'],
    )
    def op(self, request):
        return request.param

    def test(self, op):
        _run(op, [(3, 4, 4)])


class TestPixelShuffle:
    def test(self):
        _run(nn.PixelShuffle(2), [(12, 4, 4)])


class TestPixelUnshuffle:
    def test(self):
        _run(nn.PixelUnshuffle(2), [(3, 4, 4)])


class TestFunctionalPixelShuffle:
    def test(self):
        _run(lambda x: F.pixel_shuffle(x, 2), [(12, 4, 4)])


class TestFunctionalPixelUnshuffle:
    def test(self):
        _run(lambda x: F.pixel_unshuffle(x, 2), [(3, 4, 4)])


# ---------------------------------------------------------------------------
# ConvTranspose
# ---------------------------------------------------------------------------


class TestConvTranspose:
    @pytest.fixture(
        params=[
            (nn.ConvTranspose1d(3, 4, 3, stride=2, padding=1, output_padding=1), (3, 6)),
            (nn.ConvTranspose2d(3, 4, 3, stride=2, padding=1, output_padding=1), (3, 4, 4)),
            (nn.ConvTranspose2d(3, 4, 3, stride=1, padding=0), (3, 4, 4)),
            (nn.ConvTranspose2d(4, 4, 3, stride=2, padding=1, output_padding=1, groups=2), (4, 4, 4)),
            (nn.ConvTranspose3d(2, 4, 3, stride=2, padding=1, output_padding=1), (2, 3, 3, 3)),
        ],
        ids=['CT1d', 'CT2d', 'CT2d[stride1]', 'CT2d[groups]', 'CT3d'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


class _FConvTranspose(nn.Module):
    def __init__(self, rank, ch_in, ch_out, k, stride=1, padding=0, output_padding=0):
        super().__init__()
        shape = (ch_in, ch_out) + (k,) * rank
        self.w = nn.Parameter(torch.randn(*shape))
        self.b = nn.Parameter(torch.randn(ch_out))
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        fn = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[self.rank]
        return fn(x, self.w, self.b, stride=self.stride, padding=self.padding, output_padding=self.output_padding)


class TestFunctionalConvTranspose:
    @pytest.fixture(
        params=[
            (_FConvTranspose(1, 3, 4, 3, stride=2, padding=1, output_padding=1), (3, 6)),
            (_FConvTranspose(2, 3, 4, 3, stride=2, padding=1, output_padding=1), (3, 4, 4)),
            (_FConvTranspose(3, 2, 4, 3, stride=2, padding=1, output_padding=1), (2, 3, 3, 3)),
        ],
        ids=['F.CT1d', 'F.CT2d', 'F.CT3d'],
    )
    def case(self, request):
        return request.param

    def test(self, case):
        op, shape = case
        _run(op, [shape])


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class _EmbeddingWrap(nn.Module):
    """nn.Embedding takes integer indices; wrap so _run can feed integer-quantized inputs."""

    def __init__(self, V, D):
        super().__init__()
        self.emb = nn.Embedding(V, D)

    def forward(self, x):
        # fx sees int-like float input; cast to long for the actual torch call.
        return self.emb(x.to(torch.long))


def _run_embedding(V=8, D=4, shape=(3,)):
    # kif=(0, 3, 0) → unsigned integers in [0, 8)
    kif = (0, 3, 0)
    model = _EmbeddingWrap(V, D).eval()
    # Set embedding weights to fractional quantized values
    with torch.no_grad():
        w = torch.tensor(np.random.default_rng(0).standard_normal((V, D)).astype(np.float32))
        w = torch.round(w * 4) / 4
        model.emb.weight.copy_(w)
    inputs = _make_input([shape], kif)
    trace_inp, trace_out = trace_model(model, inputs=inputs, inputs_kif=kif, framework='torch')
    comb = trace(trace_inp, trace_out)
    n = 1024
    data = np.random.default_rng(1).integers(0, V, size=(n,) + shape).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.tensor(data)).detach().cpu().numpy()
    data_comb = data.reshape(n, -1)
    actual = comb.predict(data_comb).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


class TestEmbedding:
    def test(self):
        _run_embedding()


# ---------------------------------------------------------------------------
# Embedding via F.embedding (get_attr + call_function path)
# ---------------------------------------------------------------------------


class _FEmbedding(nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.w = nn.Parameter(torch.randn(V, D))

    def forward(self, x):
        return F.embedding(x.to(torch.long), self.w)


class TestEmbeddingPaddingIdx:
    def test(self):
        # padding_idx forces the chosen row to act as zeros.
        V, D, shape = 8, 4, (3,)
        kif = (0, 3, 0)

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(V, D, padding_idx=2)

            def forward(self, x):
                return self.emb(x.to(torch.long))

        model = _M().eval()
        with torch.no_grad():
            w = torch.tensor(np.random.default_rng(0).standard_normal((V, D)).astype(np.float32))
            w = torch.round(w * 4) / 4
            model.emb.weight.copy_(w)
            # torch zeros the padding row internally
            model.emb.weight[2].zero_()
        inputs = _make_input([shape], kif)
        trace_inp, trace_out = trace_model(model, inputs=inputs, inputs_kif=kif, framework='torch')
        comb = trace(trace_inp, trace_out)
        n = 256
        data = np.random.default_rng(2).integers(0, V, size=(n,) + shape).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.tensor(data)).detach().cpu().numpy()
        actual = comb.predict(data.reshape(n, -1)).reshape(expected.shape)
        np.testing.assert_array_equal(actual, expected)


# ----------------------------------------
# Tensor-attribute and shape-related paths
# ----------------------------------------


class TestGetattrShape:
    """``x.shape`` traces as ``getattr(x, 'shape')``; result is then indexed."""

    def test(self):
        _run(lambda x: x.reshape(x.shape[0], -1), [(3, 4)])


class TestGetattrNdim:
    """``x.ndim`` returns an int that we then add to x for a non-trivial output."""

    def test(self):
        _run(lambda x: x + x.ndim, [(8,)])


class TestMethodSize:
    """``x.size(-1)`` returns a constant int used in slicing."""

    def test(self):
        _run(lambda x: x[..., : x.size(-1) // 2], [(8,)])


class TestMethodSqueezeNoArg:
    def test(self):
        _run(lambda x: x.unsqueeze(1).squeeze(), [(4,)])


class TestGetattrIndicesUnsupported:
    """``torch.max(...).indices`` is not supported — trace must raise."""

    def test(self):
        kif = (1, 4, 4)
        inputs = _make_input([(8,)], kif)
        model = _Wrap(1)(lambda x: torch.max(x, dim=-1).indices).eval()
        with pytest.raises(NotImplementedError, match='indices'):
            trace_model(model, inputs=inputs, framework='torch')


class TestGetattrUnknownAttribute:
    """Arbitrary attribute access fails with AttributeError at replay time."""

    def test(self):
        kif = (1, 4, 4)
        inputs = _make_input([(8,)], kif)
        model = _Wrap(1)(lambda x: x.requires_grad).eval()
        with pytest.raises(AttributeError, match='requires_grad'):
            trace_model(model, inputs=inputs, framework='torch')


class TestUnsupportedSoftmax:
    """F.softmax is registered but immediately raises — used downstream as a sentinel."""

    def test(self):
        kif = (1, 4, 4)
        inputs = _make_input([(8,)], kif)
        model = _Wrap(1)(lambda x: F.softmax(x, dim=-1)).eval()
        with pytest.raises(NotImplementedError, match='softmax'):
            trace_model(model, inputs=inputs, framework='torch')


class TestSplitSections:
    """torch.split with a list of section sizes (not an int)."""

    def test(self):
        _run(lambda x: torch.split(x, [2, 4], dim=-1)[0].sum(-1) + torch.split(x, [2, 4], dim=-1)[1].sum(-1), [(6,)])


class TestReshapeFnUnpacked:
    """torch.reshape(x, *shape) (positional, not as a tuple)."""

    def test(self):
        _run(lambda x: torch.reshape(x, (-1, 2)), [(3, 4)])


class TestFunctionalEmbedding:
    def test(self):
        V, D, shape = 8, 4, (3,)
        kif = (0, 3, 0)
        model = _FEmbedding(V, D).eval()
        with torch.no_grad():
            w = torch.tensor(np.random.default_rng(2).standard_normal((V, D)).astype(np.float32))
            w = torch.round(w * 4) / 4
            model.w.copy_(w)
        inputs = _make_input([shape], kif)
        trace_inp, trace_out = trace_model(model, inputs=inputs, inputs_kif=kif, framework='torch')
        comb = trace(trace_inp, trace_out)
        n = 1024
        data = np.random.default_rng(3).integers(0, V, size=(n,) + shape).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.tensor(data)).detach().cpu().numpy()
        data_comb = data.reshape(n, -1)
        actual = comb.predict(data_comb).reshape(expected.shape)
        np.testing.assert_array_equal(actual, expected)
