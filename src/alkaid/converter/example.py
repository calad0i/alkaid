import numpy as np

from ..trace import FVArray
from ..trace.ops import quantize
from .plugin import ALIRTracerPluginBase


def operation(x):  # x[16,4]
    x = quantize(x, 1, 8, 4)
    a = np.maximum(x, 0)
    b = x[:, 1::2].T
    c = np.round(np.sin(b) * np.pi)
    d = np.repeat(c, 2, axis=0) * 3 + 4
    e = np.max(np.stack([d, -d * 2], axis=0), axis=0)
    f = c @ e.T
    g = np.einsum('ij,kj->ik', c, d)
    h = np.sum(g, axis=0)
    idx = np.argsort(h)[:2]
    j = h[idx].ravel()
    return j[None] + f[..., None] + a[-2::-4, :2] + ((a[0, -2:] > 0) & (a[-1, :2] > 0))


class ExampleModel:
    """A simple example model class for showcasing ALIR tracer plugin usage."""

    def __init__(self, input_shape: tuple[int, ...] | None = None):
        self.input_shape = input_shape

    def __call__(self, x):
        return operation(x)


class ExampleALIRTracer(ALIRTracerPluginBase):
    """Example top-level tracer for `ExampleModel`.

    The package registers this class under the `alkaid` key in the
    `alir_tracer.plugins` entry-point group. It implements the two methods
    required by `ALIRTracerPluginBase`: `get_input_shapes()` and
    `apply_model()`.
    """

    model: ExampleModel

    def get_input_shapes(self):
        return [self.model.input_shape] if self.model.input_shape is not None else None

    def apply_model(
        self,
        verbose: bool,
        inputs: tuple[FVArray, ...],
    ) -> tuple[dict[str, FVArray], list[str]]:
        assert len(inputs) == 1, 'ExampleModel expects a single input.'
        x = inputs[0]
        out = operation(x)
        return {'output_name': out}, ['output_name']
