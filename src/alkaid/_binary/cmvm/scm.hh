#pragma once

#include "types.hh"

// H(k)+ODP single-constant multiplication. Decompose multiplication by
// `constant` into a chain of shift-add/sub operations.
//
//   constant : the constant C; must be exactly representable as
//              sign * 2^t * C_odd with C_odd odd and |C_odd| < 2^31.
//   k        : H(k) extra-weight budget. The algorithm searches all signed-
//              digit representations with weight <= weight(CSD(C_odd)) + k.
//
// Returns a CombLogicResult with shape {1, 1}.
CombLogicResult scm(double constant, int k, const QInterval &qint);
