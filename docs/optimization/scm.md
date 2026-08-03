# Single Constant Multiplication

Single Constant Multiplication (SCM) lowers $y=Cx$ to shifts, additions, and subtractions. During symbolic tracing, multiplication by a power of two remains a scale shift. Multiplication by another finite scalar invokes the SCM solver automatically, unless disabled with `scm_opt=False` in the `solver_options` argument to `FVArray`.

The algorithm used is the $H(k)+ODP$ algorithm described in [Time-efficient single constant multiplication based on overlapping digit patterns](https://ieeexplore.ieee.org/document/4799221)
