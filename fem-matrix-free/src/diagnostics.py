
"""
Numerical diagnostics shared by FEM experiments.

This module contains comparison metrics only. It does not assemble,
apply, or solve a finite-element problem.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VectorDifference:
    """
    Absolute and relative differences between two vectors.
    """

    absolute_l2: float
    relative_l2: float
    absolute_linf: float
    reference_l2: float


def validate_comparison_vectors(
    reference,
    candidate,
):
    """
    Validate and return two one-dimensional finite NumPy arrays.
    """

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    candidate = np.asarray(
        candidate,
        dtype=np.float64,
    )

    if reference.ndim != 1:
        raise ValueError(
            "reference must be one-dimensional."
        )

    if candidate.ndim != 1:
        raise ValueError(
            "candidate must be one-dimensional."
        )

    if reference.shape != candidate.shape:
        raise ValueError(
            "reference and candidate must have the same shape."
        )

    if not np.isfinite(reference).all():
        raise ValueError(
            "reference contains nonfinite values."
        )

    if not np.isfinite(candidate).all():
        raise ValueError(
            "candidate contains nonfinite values."
        )

    return reference, candidate


def vector_difference(
    reference,
    candidate,
) -> VectorDifference:
    """
    Compute L2 and infinity-norm vector differences.

    If the reference norm is zero, the relative L2 difference is
    defined as the absolute L2 difference.
    """

    reference, candidate = (
        validate_comparison_vectors(
            reference,
            candidate,
        )
    )

    difference = candidate - reference

    absolute_l2 = float(
        np.linalg.norm(difference)
    )

    reference_l2 = float(
        np.linalg.norm(reference)
    )

    absolute_linf = float(
        np.linalg.norm(
            difference,
            ord=np.inf,
        )
    )

    if reference_l2 == 0.0:
        relative_l2 = absolute_l2
    else:
        relative_l2 = (
            absolute_l2 / reference_l2
        )

    return VectorDifference(
        absolute_l2=absolute_l2,
        relative_l2=float(relative_l2),
        absolute_linf=absolute_linf,
        reference_l2=reference_l2,
    )


def restricted_vector_difference(
    reference,
    candidate,
    indices,
) -> VectorDifference:
    """
    Compare vectors only on a supplied one-dimensional index set.
    """

    reference, candidate = (
        validate_comparison_vectors(
            reference,
            candidate,
        )
    )

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if indices.ndim != 1:
        raise ValueError(
            "indices must be one-dimensional."
        )

    if indices.size == 0:
        raise ValueError(
            "indices must not be empty."
        )

    if (
        np.any(indices < 0)
        or np.any(indices >= reference.size)
    ):
        raise ValueError(
            "indices contain invalid vector positions."
        )

    return vector_difference(
        reference[indices],
        candidate[indices],
    )


def observed_rate(
    error_coarse,
    error_fine,
    h_coarse,
    h_fine,
):
    """
    Compute an experimental convergence rate.
    """

    values = np.asarray(
        [
            error_coarse,
            error_fine,
            h_coarse,
            h_fine,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Errors and mesh sizes must be finite."
        )

    if np.any(values <= 0.0):
        raise ValueError(
            "Errors and mesh sizes must be positive."
        )

    if h_coarse == h_fine:
        raise ValueError(
            "Mesh sizes must be distinct."
        )

    return float(
        np.log(error_coarse / error_fine)
        / np.log(h_coarse / h_fine)
    )
