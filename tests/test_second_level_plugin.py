"""End-to-end test for the ``alkaid_keras`` second-level plugin loader.

Monkey-patches ``importlib.metadata`` access inside ``alkaid.converter._plugin_loader``
so the test does not need to write a real .dist-info into site-packages.
"""

import keras
import numpy as np
import pytest

import alkaid.converter._plugin_loader as pl
from alkaid.converter import trace_model
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase, _registry
from alkaid.trace import FVArrayInput
from alkaid.trace.ops import quantize


class _FakeLayer(keras.layers.Dense):
    """Stand-in for a layer provided by a third-party package."""

    __module__ = 'my_fake_pkg.layers'


_REGISTER_CALLS: list[int] = []


def _plugin_register() -> None:
    """Second-level entry point: executed once, registers a handler for ``_FakeLayer``."""
    _REGISTER_CALLS.append(1)

    class _FakeLayerHandler(ReplayOperationBase):
        handles = (_FakeLayer,)

        def call(self, inputs):
            inputs = quantize(inputs, 1, 3, 2)
            kernel = self._load_weight('kernel')
            bias = self._load_weight('bias')
            return inputs @ kernel + bias


class _FakeDist:
    def __init__(self, name: str):
        self.name = name


class _FakeEP:
    def __init__(self, dist_name: str, value: str, func):
        self.dist = _FakeDist(dist_name)
        self.value = value
        self._func = func

    def load(self):
        return self._func

    @property
    def name(self):
        return self.dist.name


class _FakeEPs:
    def __init__(self, eps_by_group: dict[str, list[_FakeEP]]):
        self._eps = eps_by_group

    def select(self, group: str):
        return list(self._eps.get(group, ()))


@pytest.fixture
def fake_plugin_env(monkeypatch: pytest.MonkeyPatch):
    ep = _FakeEP('my_fake_pkg', __name__ + ':_plugin_register', _plugin_register)
    monkeypatch.setattr(pl, 'entry_points', lambda: _FakeEPs({'alkaid_keras': [ep]}))
    monkeypatch.setattr(pl, '_LOADED', set())
    _registry.pop(_FakeLayer, None)
    _REGISTER_CALLS.clear()
    yield
    _registry.pop(_FakeLayer, None)


_KERNEL = np.array(
    [
        [1, -1, 2],
        [0, 1, -1],
        [-2, 1, 0],
        [1, 0, 1],
    ],
    dtype=np.float32,
)
_BIAS = np.array([0, 1, -1], dtype=np.float32)


def _build_model():
    inp = keras.Input(shape=(4,))
    layer = _FakeLayer(3, use_bias=True, kernel_initializer='zeros', bias_initializer='zeros')
    out = layer(inp)
    layer.kernel.assign(_KERNEL)  # type: ignore
    layer.bias.assign(_BIAS)  # type: ignore
    return keras.Model(inp, out)


def test_plugin_is_loaded_on_demand(fake_plugin_env):
    assert _FakeLayer not in _registry
    assert _REGISTER_CALLS == []

    model = _build_model()
    trace_model(model, inputs=FVArrayInput((1, 4)))

    assert _REGISTER_CALLS == [1]
    assert _FakeLayer in _registry


def test_plugin_loaded_only_once(fake_plugin_env):
    model_a = _build_model()
    model_b = _build_model()
    trace_model(model_a, inputs=FVArrayInput((1, 4)))
    trace_model(model_b, inputs=FVArrayInput((1, 4)))

    assert _REGISTER_CALLS == [1]


def test_plugin_numerical_correctness(fake_plugin_env):
    rng = np.random.default_rng(0)
    model = _build_model()
    inp, out = trace_model(model, inputs=FVArrayInput((1, 4)))

    from alkaid.trace import trace as _trace

    comb = _trace(inp, out)
    xs = (rng.integers(-4, 5, size=(64, 4)) / 4.0).astype(np.float32)
    keras_out = np.asarray(keras.ops.convert_to_numpy(model(xs)))
    comb_out = comb.predict(xs).reshape(keras_out.shape).astype(np.float32)
    np.testing.assert_array_equal(keras_out, comb_out)
