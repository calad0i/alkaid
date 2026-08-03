# Affine Interval Estimation

Ordinary interval arithmetic as used in da4ml loses correlations between values (e.g., $x \in [a,b]$, $x-x$ shall be 0 but will be interpreted as $[a-b, b-a]$). With affine arithmetic tracking the correlations between values, the bound estimations will be less conservative yet still safe for overflow.

Consider an affine interval of the form $$ b + \sum_i c_i \epsilon_i $$, where each atomic symbol $\epsilon_i$ retains its own quantization interval. Addition, negation, and scalar multiplication combine coefficients of the same symbols, so correlated terms can cancel before the output interval is evaluated.

Operations that cannot be represented as affine ops (such as quantization or relu), fallback to interval arithmetic and are considered as new interval source.
