# ALIR Graph Optimization

`alkaid.trace.passes.optimize()` applies general graph rewrites after tracing. Note that these does not include the CMVM optimization itself as it is applied during tracing.

Currently, the main optimization loop does the following:

1. canonicalization
2. dead-code elimination
3. common-subexpression elimination
4. no-op quantization op elimination
5. code retracing
6. surrogate model (cost/latency estimation)
7. topological live range minimization (for fast ALIR interpretation)


## Ternary Adder Fusion

Ternary adder fusion is applied after pipelining/before code emission as it is very hardware dependent. `RTLModel` and `HLSModel` apply ternary fusion by default. Pass `ternary_fuse=False` to either constructor, or use `alkaid convert --no-ternary`, to retain binary adders.
