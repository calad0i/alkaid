from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..cmvm import solver_options_t
from ..trace import FVariable, FVArray, FVArrayInput, HWConfig


def _flatten_arr(args: Any) -> FVArray:
    if isinstance(args, FVArray):
        return np.ravel(args)  # type: ignore
    if isinstance(args, FVariable):
        return FVArray(np.array([args]))
    if not isinstance(args, Sequence):
        raise ValueError(f'Expected a sequence or FVariable, got {type(args)}')
    args = [_flatten_arr(a) for a in args]
    args = [a for a in args if a is not None]
    return np.concatenate(args)  # type: ignore


class ALIRTracerPluginBase:
    """
    Base class for ALIR tracer plugins.

    Methods to be implemented by subclasses:
    - `apply_model`
    - `get_input_shapes`
    """

    def __init__(
        self,
        model: Callable,
        hwconf: HWConfig,
        solver_options: solver_options_t | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.hwconf = hwconf
        self.solver_options = solver_options
        assert not kwargs, f'Unexpected keyword arguments: {kwargs}'

    def apply_model(
        self,
        verbose: bool,
        inputs: tuple[FVArray, ...],
    ) -> tuple[dict[str, FVArray], list[str]]:
        """Apply the model and return all intermediate traces.

        Parameters
        ==========
        model: The model to be traced.
        verbose: Whether to print verbose output.
        inputs: Optional inputs to the model.

        Returns
        =======
        A tuple containing:
        - dict[str, FVArray]: A dictionary of intermediate names -> FVArray
        - list[str]: A list of output names.
        """
        ...

    def get_input_shapes(
        self,
    ) -> Sequence[tuple[int, ...]] | None:
        """Get the input shapes for the model. Only used if get_input_kifs returns None.

        Returns
        =======
        A list of input shapes, or None if not applicable. If cannot be determined, return None.
        """
        ...

    def _get_inputs(
        self,
        inputs: tuple[FVArray, ...] | FVArray | None,
        inputs_kif: tuple[int, int, int] | Sequence[tuple[int, int, int]] | None,
    ) -> tuple[FVArray, ...]:
        if inputs is not None:
            return inputs if isinstance(inputs, tuple) else (inputs,)

        shapes = self.get_input_shapes()
        assert shapes is not None, 'Inputs must be provided: cannot determine input shapes automatically.'

        if inputs_kif is None:
            return tuple(FVArrayInput(shape, self.hwconf, self.solver_options) for shape in shapes)

        _kifs: Sequence[tuple[int, int, int]] = inputs_kif  # type: ignore
        if not isinstance(inputs_kif[0], Sequence):
            _kifs = (inputs_kif,) * len(shapes)  # type: ignore
        else:
            _kifs = inputs_kif  # type: ignore
        assert len(_kifs) == len(shapes), 'Length of inputs_kif must match number of inputs'

        kifs = tuple(tuple(np.full(shape, v, dtype=np.int8) for v in _kif) for _kif, shape in zip(_kifs, shapes))
        return tuple(FVArray.from_kif(k, i, f, self.hwconf, 0, self.solver_options) for k, i, f in kifs)

    def trace(
        self,
        verbose: bool = False,
        inputs: tuple[FVArray, ...] | FVArray | None = None,
        inputs_kif: tuple[int, int, int] | None = None,
        dump: bool = False,
    ) -> dict[str, FVArray] | tuple[FVArray, FVArray]:
        """Trace the model.

        Parameters
        ==========
        verbose: Whether to print verbose output.
        inputs: Optional inputs to the model.
        inputs_kif: Optional input kif values, only used if inputs is None.
        dump: Whether to dump all intermediate traces.

        Returns
        =======
        If dump is True, returns a dictionary of all intermediate names -> FVArray.
        If dump is False, returns a list of output FVArray.
        """

        inputs = self._get_inputs(inputs, inputs_kif)

        all_traces, output_names = self.apply_model(
            verbose=verbose,
            inputs=inputs,
        )

        if dump:
            return all_traces

        outputs = _flatten_arr([all_traces[name] for name in output_names])
        inputs = _flatten_arr(inputs)
        return inputs, outputs
