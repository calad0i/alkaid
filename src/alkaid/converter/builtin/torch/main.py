from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.fx import Node, Tracer

from alkaid.converter.plugin import ALIRTracerPluginBase
from alkaid.trace import FVArray

from .layers import _functional_map, _method_map, _modules_map, torch_numpy_unary_map


class GraphTracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str):
        if type(m) in _modules_map:
            return True
        return super().is_leaf_module(m, module_qualified_name)


class MaybeRename:
    def __init__(self):
        self.counter: dict[str, int] = {}

    def __call__(self, name: str) -> str:
        if name not in self.counter:
            self.counter[name] = 0
            return name
        else:
            self.counter[name] += 1
            return f'{name}#{self.counter[name]}'


def _resolve(obj: Any, env: dict[str, Any]) -> Any:
    """Recursively replace fx Node references with env values."""
    if isinstance(obj, Node):
        return env[obj.name]
    if isinstance(obj, (list, tuple)):
        rebuilt = [_resolve(v, env) for v in obj]
        return type(obj)(rebuilt) if isinstance(obj, tuple) else rebuilt
    if isinstance(obj, dict):
        return {k: _resolve(v, env) for k, v in obj.items()}
    if isinstance(obj, slice):
        return slice(_resolve(obj.start, env), _resolve(obj.stop, env), _resolve(obj.step, env))
    return obj


def _dispatch_method(name: str, receiver: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
    """Resolve a fx ``call_method`` to a replay implementation.

    Lookup order: explicit handlers in ``_method_map``, then unary activations
    exposed as tensor methods (``x.sigmoid()`` etc.) via ``torch_numpy_unary_map``.
    """
    if name in _method_map:
        return _method_map[name](receiver, *args, **kwargs)
    if name in torch_numpy_unary_map:
        return torch_numpy_unary_map[name](receiver, *args, **kwargs)
    raise NotImplementedError(f'call_method not supported: {name}')


def _resolve_attr(root: torch.nn.Module, target: str):
    obj: Any = root
    for part in target.split('.'):
        obj = getattr(obj, part)
    return obj


class TorchALIRTracer(ALIRTracerPluginBase):
    def _get_inputs(
        self,
        inputs: tuple[FVArray, ...] | FVArray | None,
        inputs_kif: tuple[int, int, int] | Sequence[tuple[int, int, int]] | None,
    ) -> tuple[FVArray, ...]:
        if inputs is not None:
            return inputs if isinstance(inputs, tuple) else (inputs,)

        raise ValueError('Inputs must be provided: cannot determine input shapes automatically.')

    def apply_model(
        self,
        verbose: bool,
        inputs: tuple[FVArray, ...],
    ):
        assert inputs is not None
        if isinstance(inputs, FVArray):
            inputs = (inputs,)
        self.model: torch.nn.Module
        graph = GraphTracer().trace(self.model)
        modules = dict(self.model.named_modules())
        inp_nodes = [n for n in graph.nodes if n.op == 'placeholder']
        out_nodes = [n for n in graph.nodes if n.op == 'output']
        assert len(out_nodes) == 1, f'only one output node is supported, but found {len(out_nodes)}'
        assert len(inputs) == len(inp_nodes), (
            f'inputs length {len(inputs)} does not match with graph input length {len(inp_nodes)}'
        )
        maybe_rename = MaybeRename()

        env: dict[str, Any] = {}
        for node, inp in zip(inp_nodes, inputs):
            env[node.name] = inp

        trace: dict[str, tuple[FVArray, ...]] = {'inputs': tuple(inputs)}

        for node in graph.nodes:
            args = _resolve(node.args, env)
            kwargs = _resolve(node.kwargs, env)
            match node.op:
                case 'placeholder':
                    continue
                case 'output':
                    continue
                case 'call_module':
                    target: str = node.target  # type: ignore
                    module = modules[target]
                    assert type(module) in _modules_map, f'{type(module)} is not supported'
                    replay_cls = _modules_map[type(module)]
                    replay = replay_cls(module)
                    _dump = replay(*args, **kwargs)
                    env[node.name] = _dump['final'][0]
                    name = maybe_rename(target.replace('.', '/'))
                    for k, v in _dump.items():
                        trace[f'{name}/{k}'] = v
                case 'call_function':
                    torch_fn: Callable = node.target  # type: ignore
                    assert torch_fn in _functional_map, f'{torch_fn} is not registered in functional map'
                    result = _functional_map[torch_fn](*args, **kwargs)
                    env[node.name] = result
                    name = maybe_rename(node.name)
                    if isinstance(result, FVArray):
                        trace[f'{name}/final'] = (result,)
                case 'call_method':
                    method: str = node.target  # type: ignore
                    receiver, *rest = args
                    result = _dispatch_method(method, receiver, rest, kwargs)
                    env[node.name] = result
                    name = maybe_rename(node.name)
                    if isinstance(result, FVArray):
                        trace[f'{name}/final'] = (result,)
                case 'get_attr':
                    target: str = node.target  # type: ignore
                    env[node.name] = _resolve_attr(self.model, target)
                case _:
                    raise NotImplementedError(f'unknown node op: {node.op}')

        out_arg = out_nodes[0].args[0]
        if isinstance(out_arg, Node):
            finals = (env[out_arg.name],)
        else:
            finals = tuple(env[n.name] if isinstance(n, Node) else n for n in out_arg)  # type: ignore
        trace['final'] = finals  # type: ignore
        return trace, ['final']
