# Conversion Plugin

alkaid always lowers computations through its NumPy-style symbolic tracing API. Conversion plugins bridge external model objects into that tracing API, so framework models can be replayed on `FVArray` inputs and then converted to ALIR.

## Top-Level Tracer Plugins

Top-level plugins are registered in the `alir_tracer.plugins` entry-point group. Each entry point maps a framework key to an `ALIRTracerPluginBase` subclass:

- `get_input_shapes()` returns input shapes when they can be inferred from the model, or `None` when callers must provide symbolic inputs.
- `apply_model(verbose, inputs)` replays the model on `FVArray` inputs and returns a trace dictionary plus output names.

When `alkaid.converter.trace_model()` is called without an explicit `framework`, the framework key is inferred from `type(model).__module__.split('.', 1)[0]`. For example, a Keras model resolves to `keras`, so the built-in Keras tracer is selected.

The current package declares these top-level entries in `pyproject.toml`:

| Entry key | Tracer                                                |
| --------- | ----------------------------------------------------- |
| `alkaid`  | `alkaid.converter.example:ExampleALIRTracer`          |
| `keras`   | `alkaid.converter.builtin.keras.main:KerasALIRTracer` |
| `torch`   | `alkaid.converter.builtin.torch.main:TorchALIRTracer` |

## Second-Level Operation Plugins

The built-in Keras and Torch tracers can be extended without replacing the top-level tracer. Third-party packages can register a zero-argument callable in one of these groups:

| Entry-point group | Purpose                                                            |
| ----------------- | ------------------------------------------------------------------ |
| `alkaid_keras`    | Import/register Keras layer or operation replay handlers.          |
| `alkaid_torch`    | Import/register Torch module, function, or method replay handlers. |

The callable should import code that registers handlers in the corresponding built-in registry. alkaid loads such plugins on demand based on the base module of the layer, operation, or module being replayed.

## Known Integrations

| Integration   | Mechanism                            | Description                                                                     |
| ------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| Keras 3       | Built-in `keras` top-level tracer    | Replays supported Keras graph operations.                                       |
| Torch         | Built-in `torch` top-level tracer    | Replays supported `torch.fx` graph nodes; callers must provide symbolic inputs. |
| HGQ2          | `alkaid_keras` second-level handlers | Adds HGQ-specific Keras layer support when HGQ2 is installed.                   |
| Example model | Built-in `alkaid` top-level tracer   | Minimal tracer used as an implementation reference and in tests.                |

Additional frontends can still register their own top-level key under `alir_tracer.plugins` when they are not naturally expressed as Keras or Torch operation handlers.

## Example Plugin

An example plugin class is provided in `alkaid.converter.example`. The package metadata registers it as:

```toml
entry-points."alir_tracer.plugins".alkaid = "alkaid.converter.example:ExampleALIRTracer"
```

The plugin is used in `tests/test_plugin.py`, which is the smallest complete example of the top-level tracer interface.
