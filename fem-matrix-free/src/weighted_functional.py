
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
# Smoothly weighted local output
# \int_T q\,u_h\,dx
# J_{q,T}(u_h) = \sum_{i=1}^3 u_{T,i} \int_Tq(x)\phi_{T,i}(x)\,dx
# (j_T)_i = \int_Tq(x)\phi_{T,i}(x)\,dx
# J_q(u_h) = \frac{\displaystyle \sum_Tj_T^TR_Tu_h }{ \displaystyle \sum_T\int_Tq(x)\,dx }
# ==================================================


@dataclass(frozen=True)
class WeightedFunctionalData:
    """
    Precomputed element-local data for a normalized weighted output.

    The local functional vectors satisfy

        local_vectors[T, i]
            = integral_T weight(x) * phi_i(x) dx.

    The weighted output is then

        sum_T local_vectors[T]^T u_T
        --------------------------------.
               integral_Omega weight dx
    """

    center: tuple
    sigma: float
    quadrature_order: int

    dof_connectivity: np.ndarray
    local_vectors: np.ndarray
    element_weight_integrals: np.ndarray

    weight_integral: float
    num_cells: int
    num_dofs: int


def _validate_weight_parameters(
    center,
    sigma,
    quadrature_order,
):
    """
    Validate Gaussian-weight and quadrature parameters.
    """

    center = np.asarray(
        center,
        dtype=np.float64,
    )

    if center.shape != (2,):
        raise ValueError(
            "center must contain exactly two coordinates."
        )

    if not np.isfinite(center).all():
        raise ValueError(
            "center coordinates must be finite."
        )

    sigma = float(
        sigma
    )

    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
    ):
        raise ValueError(
            "sigma must be finite and positive."
        )

    if isinstance(
        quadrature_order,
        bool,
    ):
        raise ValueError(
            "quadrature_order must be a positive integer."
        )

    try:
        integer_order = int(
            quadrature_order
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise ValueError(
            "quadrature_order must be a positive integer."
        ) from exc

    if (
        integer_order <= 0
        or integer_order != quadrature_order
    ):
        raise ValueError(
            "quadrature_order must be a positive integer."
        )

    return (
        (
            float(center[0]),
            float(center[1]),
        ),
        sigma,
        integer_order,
    )


def gaussian_weight_values(
    points,
    center=(0.70, 0.35),
    sigma=0.10,
):
    """
    Evaluate the unnormalized Gaussian weight at physical points.

    Parameters
    ----------
    points:
        Array of shape (..., 2).

    center:
        Gaussian center (x_0, y_0).

    sigma:
        Gaussian width.
    """

    center, sigma, _ = (
        _validate_weight_parameters(
            center,
            sigma,
            1,
        )
    )

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if (
        points.ndim < 1
        or points.shape[-1] != 2
    ):
        raise ValueError(
            "points must have final dimension two."
        )

    if not np.isfinite(
        points
    ).all():
        raise ValueError(
            "points must contain only finite values."
        )

    displacement = (
        points
        - np.asarray(
            center,
            dtype=np.float64,
        )
    )

    squared_distance = np.sum(
        displacement**2,
        axis=-1,
    )

    return np.exp(
        -squared_distance
        / (2.0 * sigma**2)
    )


def reference_triangle_quadrature(
    order=12,
):
    """
    Construct product Gauss-Legendre quadrature on the reference triangle.

    The reference triangle is

        T_hat = {
            (xi, eta):
            0 <= xi <= 1,
            0 <= eta <= 1 - xi
        }.

    Use the Duffy transformation

        xi  = r,
        eta = (1-r) s,

    where (r,s) belongs to [0,1]^2. The Jacobian is (1-r).

    Returns
    -------
    points:
        Array of shape (order**2, 2).

    weights:
        Array of shape (order**2,).

    The weights sum to 1/2, the area of the reference triangle.
    """

    _, _, order = _validate_weight_parameters(
        center=(0.0, 0.0),
        sigma=1.0,
        quadrature_order=order,
    )

    nodes, weights = np.polynomial.legendre.leggauss(
        order
    )

    # Transform Gauss-Legendre points and weights
    # from [-1, 1] to [0, 1].
    nodes = 0.5 * (
        nodes + 1.0
    )

    weights = 0.5 * weights

    r, s = np.meshgrid(
        nodes,
        nodes,
        indexing="ij",
    )

    weight_r, weight_s = np.meshgrid(
        weights,
        weights,
        indexing="ij",
    )

    xi = r
    eta = (
        1.0 - r
    ) * s

    triangle_weights = (
        weight_r
        * weight_s
        * (1.0 - r)
    )

    points = np.column_stack(
        (
            xi.ravel(),
            eta.ravel(),
        )
    )

    triangle_weights = (
        triangle_weights.ravel()
    )

    return (
        points,
        triangle_weights,
    )


def create_weighted_functional_data(
    problem,
    center=(0.70, 0.35),
    sigma=0.10,
    quadrature_order=12,
) -> WeightedFunctionalData:
    """
    Precompute the element-local vectors for a Gaussian-weighted output.

    For every triangle T and local basis function phi_i,

        local_vectors[T, i]
            = integral_T q(x) phi_i(x) dx.

    The calculation uses product Gauss-Legendre quadrature after a
    Duffy transformation of the reference triangle.
    """

    center, sigma, quadrature_order = (
        _validate_weight_parameters(
            center,
            sigma,
            quadrature_order,
        )
    )

    local_geometry = LocalP1Operator(
        problem,
        precompute_matrices=False,
    )

    reference_points, reference_weights = (
        reference_triangle_quadrature(
            quadrature_order
        )
    )

    xi = reference_points[:, 0]
    eta = reference_points[:, 1]

    # Reference P1 basis values.
    basis_values = np.column_stack(
        (
            1.0 - xi - eta,
            xi,
            eta,
        )
    )

    num_cells = local_geometry.num_cells
    num_quadrature_points = (
        reference_points.shape[0]
    )

    local_vectors = np.zeros(
        (
            num_cells,
            3,
        ),
        dtype=np.float64,
    )

    element_weight_integrals = np.zeros(
        num_cells,
        dtype=np.float64,
    )

    for cell in range(
        num_cells
    ):
        vertices = (
            local_geometry.cell_coordinates[
                cell
            ]
        )

        vertex_zero = vertices[0]

        edge_one = (
            vertices[1]
            - vertices[0]
        )

        edge_two = (
            vertices[2]
            - vertices[0]
        )

        physical_points = (
            vertex_zero[None, :]
            + xi[:, None]
            * edge_one[None, :]
            + eta[:, None]
            * edge_two[None, :]
        )

        weight_values = gaussian_weight_values(
            physical_points,
            center=center,
            sigma=sigma,
        )

        # The affine reference-to-physical Jacobian has
        # absolute determinant 2*area.
        physical_quadrature_weights = (
            2.0
            * local_geometry.areas[cell]
            * reference_weights
        )

        weighted_quadrature = (
            physical_quadrature_weights
            * weight_values
        )

        element_weight_integrals[cell] = (
            np.sum(
                weighted_quadrature,
                dtype=np.float64,
            )
        )

        local_vectors[cell] = np.einsum(
            "q,qi->i",
            weighted_quadrature,
            basis_values,
        )

    if not np.isfinite(
        local_vectors
    ).all():
        raise FloatingPointError(
            "Weighted local functional vectors contain "
            "nonfinite values."
        )

    if not np.isfinite(
        element_weight_integrals
    ).all():
        raise FloatingPointError(
            "Element weight integrals contain nonfinite values."
        )

    weight_integral = float(
        np.sum(
            element_weight_integrals,
            dtype=np.float64,
        )
    )

    if (
        not np.isfinite(weight_integral)
        or weight_integral <= 0.0
    ):
        raise ValueError(
            "The global Gaussian-weight integral must be positive."
        )

    return WeightedFunctionalData(
        center=center,
        sigma=sigma,
        quadrature_order=quadrature_order,
        dof_connectivity=(
            local_geometry.dof_connectivity.copy()
        ),
        local_vectors=local_vectors,
        element_weight_integrals=(
            element_weight_integrals
        ),
        weight_integral=weight_integral,
        num_cells=num_cells,
        num_dofs=local_geometry.num_dofs,
    )


def weighted_integral_local(
    problem,
    solution,
    data,
) -> float:
    """
    Evaluate integral_Omega q(x) u_h(x) dx locally.
    """

    coefficients = _validate_solution(
        problem,
        solution,
    )

    if not isinstance(
        data,
        WeightedFunctionalData,
    ):
        # This check may fail after notebook module reloads.
        # The structural checks below remain decisive.
        required_attributes = (
            "dof_connectivity",
            "local_vectors",
            "weight_integral",
            "num_cells",
            "num_dofs",
        )

        for attribute in required_attributes:
            if not hasattr(
                data,
                attribute,
            ):
                raise TypeError(
                    "Invalid weighted functional data."
                )

    if data.num_dofs != coefficients.size:
        raise ValueError(
            "Weighted functional data and solution "
            "dimensions do not agree."
        )

    if data.dof_connectivity.shape != (
        data.num_cells,
        3,
    ):
        raise ValueError(
            "Invalid weighted functional connectivity."
        )

    if data.local_vectors.shape != (
        data.num_cells,
        3,
    ):
        raise ValueError(
            "Invalid local functional-vector dimensions."
        )

    element_coefficients = coefficients[
        data.dof_connectivity
    ]

    element_contributions = np.einsum(
        "ei,ei->e",
        data.local_vectors,
        element_coefficients,
    )

    if not np.isfinite(
        element_contributions
    ).all():
        raise FloatingPointError(
            "Weighted element contributions contain "
            "nonfinite values."
        )

    return float(
        np.sum(
            element_contributions,
            dtype=np.float64,
        )
    )


def weighted_average_local(
    problem,
    solution,
    data,
) -> float:
    """
    Evaluate the normalized Gaussian-weighted average locally.
    """

    numerator = weighted_integral_local(
        problem,
        solution,
        data,
    )

    return float(
        numerator
        / data.weight_integral
    )


def _gaussian_weight_expression(
    problem,
    center,
    sigma,
):
    """
    Construct the Gaussian UFL expression used by Firedrake.
    """

    from firedrake import (
        SpatialCoordinate,
        exp,
    )

    center, sigma, _ = (
        _validate_weight_parameters(
            center,
            sigma,
            1,
        )
    )

    x, y = SpatialCoordinate(
        problem.mesh
    )

    x_center, y_center = center

    return exp(
        -(
            (x - x_center) ** 2
            + (y - y_center) ** 2
        )
        / (2.0 * sigma**2)
    )


def weighted_integral_reference(
    problem,
    solution,
    center=(0.70, 0.35),
    sigma=0.10,
    quadrature_degree=24,
) -> float:
    """
    Evaluate integral_Omega q u_h dx using Firedrake assembly.
    """

    _validate_solution(
        problem,
        solution,
    )

    weight = _gaussian_weight_expression(
        problem,
        center,
        sigma,
    )

    return float(
        assemble(
            weight
            * solution
            * dx(
                degree=quadrature_degree
            )
        )
    )


def weight_integral_reference(
    problem,
    center=(0.70, 0.35),
    sigma=0.10,
    quadrature_degree=24,
) -> float:
    """
    Evaluate integral_Omega q dx using Firedrake assembly.
    """

    weight = _gaussian_weight_expression(
        problem,
        center,
        sigma,
    )

    return float(
        assemble(
            weight
            * dx(
                degree=quadrature_degree
            )
        )
    )


def weighted_average_reference(
    problem,
    solution,
    center=(0.70, 0.35),
    sigma=0.10,
    quadrature_degree=24,
) -> float:
    """
    Evaluate the normalized Gaussian-weighted average with Firedrake.
    """

    numerator = weighted_integral_reference(
        problem,
        solution,
        center=center,
        sigma=sigma,
        quadrature_degree=quadrature_degree,
    )

    denominator = weight_integral_reference(
        problem,
        center=center,
        sigma=sigma,
        quadrature_degree=quadrature_degree,
    )

    if denominator <= 0.0:
        raise ValueError(
            "The reference weight integral must be positive."
        )

    return float(
        numerator / denominator
    )


def exact_weighted_average_reference(
    problem,
    center=(0.70, 0.35),
    sigma=0.10,
    quadrature_degree=30,
) -> float:
    """
    Evaluate the normalized weighted output of the exact solution.

    This is a high-order quadrature reference, not a symbolic
    closed-form value.
    """

    weight = _gaussian_weight_expression(
        problem,
        center,
        sigma,
    )

    numerator = float(
        assemble(
            weight
            * problem.u_exact
            * dx(
                degree=quadrature_degree
            )
        )
    )

    denominator = float(
        assemble(
            weight
            * dx(
                degree=quadrature_degree
            )
        )
    )

    if denominator <= 0.0:
        raise ValueError(
            "The exact-reference weight integral must be positive."
        )

    return float(
        numerator / denominator
    )


def compare_weighted_average(
    problem,
    solution,
    data,
    reference_quadrature_degree=24,
) -> FunctionalResult:
    """
    Compare Firedrake and element-local weighted evaluations.
    """

    reference = weighted_average_reference(
        problem,
        solution,
        center=data.center,
        sigma=data.sigma,
        quadrature_degree=(
            reference_quadrature_degree
        ),
    )

    local = weighted_average_local(
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
