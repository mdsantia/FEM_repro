
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
# Mesh-aligned rectangular subdomain functionals
# The local formula J_D(u_h) = \frac{1}{|D|} \sum_{T\subset D} \frac{|T|}{3} \sum_{i=1}^3u_{T,i}
# ==================================================


@dataclass(frozen=True)
class SubdomainFunctionalData:
    """
    Local data for a mesh-aligned rectangular subdomain.

    Only triangles fully contained in the rectangle contribute to
    the local functional.
    """

    bounds: tuple
    selected_cells: np.ndarray
    selected_connectivity: np.ndarray
    selected_areas: np.ndarray
    subdomain_area: float
    num_selected_cells: int
    num_cells: int
    num_dofs: int


def _validate_rectangle_bounds(bounds):
    """
    Validate rectangular subdomain bounds.

    The expected ordering is:

        (x_min, x_max, y_min, y_max)
    """

    bounds = np.asarray(
        bounds,
        dtype=np.float64,
    )

    if bounds.shape != (4,):
        raise ValueError(
            "bounds must contain "
            "(x_min, x_max, y_min, y_max)."
        )

    if not np.isfinite(bounds).all():
        raise ValueError(
            "Subdomain bounds must be finite."
        )

    x_min, x_max, y_min, y_max = bounds

    if x_min >= x_max:
        raise ValueError(
            "x_min must be smaller than x_max."
        )

    if y_min >= y_max:
        raise ValueError(
            "y_min must be smaller than y_max."
        )

    return tuple(
        float(value)
        for value in bounds
    )


def _cell_is_cut_by_rectangle_boundary(
    cell_coordinates,
    bounds,
    tolerance,
):
    """
    Determine whether a triangle interior is cut by a rectangle edge.

    A strict crossing is detected when a rectangle boundary lies
    between the minimum and maximum cell coordinates.

    If a rectangle boundary coincides with a triangle edge, the cell
    is not considered cut.
    """

    x_min, x_max, y_min, y_max = bounds

    cell_x_min = float(
        np.min(
            cell_coordinates[:, 0]
        )
    )

    cell_x_max = float(
        np.max(
            cell_coordinates[:, 0]
        )
    )

    cell_y_min = float(
        np.min(
            cell_coordinates[:, 1]
        )
    )

    cell_y_max = float(
        np.max(
            cell_coordinates[:, 1]
        )
    )

    vertical_boundaries = (
        x_min,
        x_max,
    )

    horizontal_boundaries = (
        y_min,
        y_max,
    )

    for boundary in vertical_boundaries:

        crosses = (
            cell_x_min + tolerance
            < boundary
            < cell_x_max - tolerance
        )

        if crosses:
            return True

    for boundary in horizontal_boundaries:

        crosses = (
            cell_y_min + tolerance
            < boundary
            < cell_y_max - tolerance
        )

        if crosses:
            return True

    return False


def create_subdomain_functional_data(
    problem,
    bounds,
    tolerance=1.0e-12,
) -> SubdomainFunctionalData:
    """
    Construct local functional data for a mesh-aligned rectangle.

    Parameters
    ----------
    problem:
        Manufactured Poisson problem.

    bounds:
        Tuple

            (x_min, x_max, y_min, y_max).

    tolerance:
        Geometric comparison tolerance.

    Notes
    -----
    This function rejects subdomains whose boundary cuts through
    triangle interiors. Non-mesh-aligned subdomains require cut-cell
    integration and belong to a later extension.
    """

    bounds = _validate_rectangle_bounds(
        bounds
    )

    tolerance = float(
        tolerance
    )

    if (
        not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError(
            "tolerance must be finite and nonnegative."
        )

    x_min, x_max, y_min, y_max = bounds

    local_geometry = LocalP1Operator(
        problem,
        precompute_matrices=False,
    )

    cell_coordinates = (
        local_geometry.cell_coordinates
    )

    for cell in range(
        local_geometry.num_cells
    ):
        if _cell_is_cut_by_rectangle_boundary(
            cell_coordinates[cell],
            bounds,
            tolerance,
        ):
            raise ValueError(
                "The requested rectangle is not aligned "
                "with the triangular mesh."
            )

    cell_centroids = np.mean(
        cell_coordinates,
        axis=1,
    )

    centroid_x = cell_centroids[:, 0]
    centroid_y = cell_centroids[:, 1]

    selected_mask = (
        (centroid_x >= x_min - tolerance)
        & (centroid_x <= x_max + tolerance)
        & (centroid_y >= y_min - tolerance)
        & (centroid_y <= y_max + tolerance)
    )

    selected_cells = np.flatnonzero(
        selected_mask
    )

    if selected_cells.size == 0:
        raise ValueError(
            "The subdomain contains no mesh cells."
        )

    selected_connectivity = (
        local_geometry.dof_connectivity[
            selected_cells
        ].copy()
    )

    selected_areas = (
        local_geometry.areas[
            selected_cells
        ].copy()
    )

    geometric_area = float(
        np.sum(
            selected_areas,
            dtype=np.float64,
        )
    )

    exact_rectangle_area = float(
        (x_max - x_min)
        * (y_max - y_min)
    )

    area_tolerance = max(
        1.0e-13,
        100.0
        * np.finfo(np.float64).eps
        * exact_rectangle_area,
    )

    if not np.isclose(
        geometric_area,
        exact_rectangle_area,
        rtol=0.0,
        atol=area_tolerance,
    ):
        raise ValueError(
            "Selected triangle areas do not reproduce the "
            "requested rectangle area. The subdomain may not "
            "be mesh-aligned."
        )

    return SubdomainFunctionalData(
        bounds=bounds,
        selected_cells=selected_cells,
        selected_connectivity=(
            selected_connectivity
        ),
        selected_areas=selected_areas,
        subdomain_area=geometric_area,
        num_selected_cells=int(
            selected_cells.size
        ),
        num_cells=local_geometry.num_cells,
        num_dofs=local_geometry.num_dofs,
    )


def subdomain_integral_local(
    problem,
    solution,
    bounds=None,
    data=None,
) -> float:
    """
    Evaluate the integral of a P1 function over a mesh-aligned
    rectangular subdomain.

    For every selected triangle,

        integral_T u_h dx
            = |T|/3 * sum_i u_{T,i}.
    """

    coefficients = _validate_solution(
        problem,
        solution,
    )

    if data is None:

        if bounds is None:
            raise ValueError(
                "Either bounds or precomputed subdomain "
                "data must be supplied."
            )

        data = create_subdomain_functional_data(
            problem,
            bounds,
        )

    if data.num_dofs != coefficients.size:
        raise ValueError(
            "Subdomain data and solution dimensions "
            "do not agree."
        )

    element_coefficients = coefficients[
        data.selected_connectivity
    ]

    element_contributions = (
        data.selected_areas
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
            "Subdomain contributions contain "
            "nonfinite values."
        )

    return float(
        np.sum(
            element_contributions,
            dtype=np.float64,
        )
    )


def subdomain_average_local(
    problem,
    solution,
    bounds=None,
    data=None,
) -> float:
    """
    Evaluate the average of a P1 function over a mesh-aligned
    rectangular subdomain.
    """

    if data is None:

        if bounds is None:
            raise ValueError(
                "Either bounds or precomputed subdomain "
                "data must be supplied."
            )

        data = create_subdomain_functional_data(
            problem,
            bounds,
        )

    integral = subdomain_integral_local(
        problem,
        solution,
        data=data,
    )

    return float(
        integral / data.subdomain_area
    )


def _rectangle_indicator(
    problem,
    bounds,
):
    """
    Return a UFL indicator for a rectangular subdomain.
    """

    from firedrake import (
        And,
        SpatialCoordinate,
        conditional,
        ge,
        le,
    )

    x_min, x_max, y_min, y_max = (
        _validate_rectangle_bounds(
            bounds
        )
    )

    x, y = SpatialCoordinate(
        problem.mesh
    )

    inside = And(
        And(
            ge(x, x_min),
            le(x, x_max),
        ),
        And(
            ge(y, y_min),
            le(y, y_max),
        ),
    )

    return conditional(
        inside,
        1.0,
        0.0,
    )


def subdomain_integral_reference(
    problem,
    solution,
    bounds,
) -> float:
    """
    Evaluate the rectangular subdomain integral with Firedrake.

    The rectangle should be aligned with the mesh used in the
    experiment.
    """

    _validate_solution(
        problem,
        solution,
    )

    indicator = _rectangle_indicator(
        problem,
        bounds,
    )

    return float(
        assemble(
            indicator
            * solution
            * dx
        )
    )


def subdomain_average_reference(
    problem,
    solution,
    bounds,
) -> float:
    """
    Evaluate the rectangular subdomain average with Firedrake.
    """

    bounds = _validate_rectangle_bounds(
        bounds
    )

    x_min, x_max, y_min, y_max = bounds

    integral = subdomain_integral_reference(
        problem,
        solution,
        bounds,
    )

    area = (
        (x_max - x_min)
        * (y_max - y_min)
    )

    return float(
        integral / area
    )


def compare_subdomain_average(
    problem,
    solution,
    bounds=None,
    data=None,
) -> FunctionalResult:
    """
    Compare Firedrake and local subdomain-average evaluations.
    """

    if data is None:

        if bounds is None:
            raise ValueError(
                "Either bounds or precomputed subdomain "
                "data must be supplied."
            )

        data = create_subdomain_functional_data(
            problem,
            bounds,
        )

    reference = subdomain_average_reference(
        problem,
        solution,
        data.bounds,
    )

    local = subdomain_average_local(
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


def exact_subdomain_average(
    bounds,
) -> float:
    """
    Return the exact rectangular average of

        sin(pi*x) sin(pi*y).

    For D = [a,b] x [c,d],

        integral_D u dx
          = ((cos(pi*a) - cos(pi*b))/pi)
            ((cos(pi*c) - cos(pi*d))/pi).
    """

    x_min, x_max, y_min, y_max = (
        _validate_rectangle_bounds(
            bounds
        )
    )

    x_integral = (
        np.cos(
            np.pi * x_min
        )
        - np.cos(
            np.pi * x_max
        )
    ) / np.pi

    y_integral = (
        np.cos(
            np.pi * y_min
        )
        - np.cos(
            np.pi * y_max
        )
    ) / np.pi

    area = (
        (x_max - x_min)
        * (y_max - y_min)
    )

    return float(
        x_integral
        * y_integral
        / area
    )
