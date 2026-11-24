
"""
Element-local evaluation of finite-element output functionals.

The initial implementation supports exact global integration and
averaging of continuous P1 finite-element functions on affine
triangular meshes.

Firedrake assembly is used only to provide a reference value.
The local implementation uses scalar DoF connectivity, triangle
areas, and solution coefficients directly.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FunctionalResult:
    """
    Comparison between reference and local functional evaluations.
    """

    reference_value: float
    local_value: float
    absolute_difference: float
    relative_difference: float

def _validate_solution(problem, solution):
    """
    Validate a finite-element solution before functional evaluation.
    """

    if solution.function_space() is not problem.V:
        raise ValueError(
            "The supplied solution belongs to a different "
            "function space."
        )

    values = np.asarray(
        solution.dat.data_ro,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "The solution coefficient vector must be "
            "one-dimensional."
        )

    if values.size != problem.V.dim():
        raise ValueError(
            f"Expected {problem.V.dim()} solution coefficients, "
            f"received {values.size}."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "The solution contains nonfinite coefficients."
        )

    return values
