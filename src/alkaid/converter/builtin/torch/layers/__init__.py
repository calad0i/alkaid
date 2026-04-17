from . import direct  # noqa: F401  (registers unary activations into _functional_map)
from .direct import torch_numpy_unary_map
from .functional import _functional_map
from .methods import _method_map
from .modules import _modules_map  # also imports modules (registers handlers via metaclass)

__all__ = ['_functional_map', '_method_map', '_modules_map', 'torch_numpy_unary_map']
