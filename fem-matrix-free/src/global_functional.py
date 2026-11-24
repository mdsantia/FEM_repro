
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

from firedrake import assemble, dx

from src.local_operator import LocalP1Operator

from src.functionals import (
    _validate_solution,
    FunctionalResult,
)

# ==================================================
# Mesh-aligned rectangular global functionals
# J_{\mathrm{global}}(u_h) = \frac{1}{|\Omega|} \sum_{T\in\mathcal T_h} \frac{|T|}{3} \left( u_{T,1}+u_{T,2}+u_{T,3} \right)
# ==================================================

@dataclass(frozen=True)
class GlobalFunctionalData:
    """
    Local data required for global P1 functional evaluation.
    """

    dof_connectivity: np.ndarray
    areas: np.ndarray
    domain_area: float
    num_cells: int
    num_dofs: int


def create_global_functional_data(
    problem,
) -> GlobalFunctionalData:
    """
    Construct the local data needed for global P1 integration.

    The verified LocalP1Operator geometry is reused so that
    functional evaluation uses the same scalar DoF ordering and
    triangle areas as the independent local operator.
    """

    local_geometry = LocalP1Operator(
        problem,
        precompute_matrices=False,
    )

    domain_area = float(
        np.sum(local_geometry.areas)
    )

    if not np.isfinite(domain_area):
        raise FloatingPointError(
            "The computed domain area is not finite."
        )

    if domain_area <= 0.0:
        raise ValueError(
            "The computed domain area must be positive."
        )

    return GlobalFunctionalData(
        dof_connectivity=(
            local_geometry.dof_connectivity.copy()
        ),
        areas=local_geometry.areas.copy(),
        domain_area=domain_area,
        num_cells=local_geometry.num_cells,
        num_dofs=local_geometry.num_dofs,
    )


def global_integral_reference(
    problem,
    solution,
) -> float:
    """
    Evaluate integral_Omega u_h dx using Firedrake assembly.
    """

    _validate_solution(
        problem,
        solution,
    )

    return float(
        assemble(
            solution * dx
        )
    )


def global_integral_local(
    problem,
    solution,
    data=None,
) -> float:
    """
    Evaluate integral_Omega u_h dx by exact element-local P1 formulas.

    For each affine triangle T,

        integral_T u_h dx
            = |T|/3 * sum_i u_{T,i}.
    """

    coefficients = _validate_solution(
        problem,
        solution,
    )

    if data is None:
        data = create_global_functional_data(
            problem
        )

    if data.num_dofs != coefficients.size:
        raise ValueError(
            "Functional data and solution dimensions do not agree."
        )

    element_coefficients = coefficients[
        data.dof_connectivity
    ]

    element_contributions = (
        data.areas
        * np.sum(
            element_coefficients,
            axis=1,
        )
        / 3.0
    )

    if not np.isfinite(
        element_contributions
    ).all():
        raise FloatingPointError(
            "Local functional contributions contain "
            "nonfinite values."
        )

    return float(
        np.sum(
            element_contributions,
            dtype=np.float64,
        )
    )


def global_average_reference(
    problem,
    solution,
) -> float:
    """
    Evaluate the global average using Firedrake assembly.
    """

    integral = global_integral_reference(
        problem,
        solution,
    )

    domain_area = float(
        assemble(
            1.0 * dx(
                domain=problem.mesh
            )
        )
    )

    if domain_area <= 0.0:
        raise ValueError(
            "The assembled domain area must be positive."
        )

    return integral / domain_area


def global_average_local(
    problem,
    solution,
    data=None,
) -> float:
    """
    Evaluate the global average using local triangle contributions.
    """

    if data is None:
        data = create_global_functional_data(
            problem
        )

    integral = global_integral_local(
        problem,
        solution,
        data=data,
    )

    return integral / data.domain_area


def compare_global_average(
    problem,
    solution,
    data=None,
) -> FunctionalResult:
    """
    Compare Firedrake and local global-average evaluations.
    """

    if data is None:
        data = create_global_functional_data(
            problem
        )

    reference = global_average_reference(
        problem,
        solution,
    )

    local = global_average_local(
        problem,
        solution,
        data=data,
    )

    absolute_difference = abs(
        local - reference
    )

    if reference == 0.0:
        relative_difference = (
            absolute_difference
        )
    else:
        relative_difference = (
            absolute_difference
            / abs(reference)
        )

    return FunctionalResult(
        reference_value=float(reference),
        local_value=float(local),
        absolute_difference=float(
            absolute_difference
        ),
        relative_difference=float(
            relative_difference
        ),
    )


def exact_global_average() -> float:
    """
    Return the exact global average of the manufactured solution.

    Since Omega is the unit square,

        integral_Omega sin(pi*x) sin(pi*y) dx dy = 4/pi^2.
    """

    return float(
        4.0 / np.pi**2
    )
