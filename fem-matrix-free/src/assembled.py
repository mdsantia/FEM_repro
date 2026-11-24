"""
Assembled reference implementation for the manufactured Poisson problem.

This module assembles the global finite-element stiffness matrix and
load vector using the variational forms defined in problem.py.

The assembled matrix serves as the reference operator for validating
the matrix-free implementations.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from firedrake import assemble, Function, LinearSolver


@dataclass
class AssembledSystem:
    """
    Container for the assembled Dirichlet-constrained system.
    """
    A: Any
    b: Any
    function_space: Any
    boundary_dofs: np.ndarray
    free_dofs: np.ndarray
    ndofs: int


@dataclass
class AssembledSolveResult:
    """
    Container for the assembled solution and solver diagnostics.
    """

    solution: Function
    system: AssembledSystem
    residual_norm: float
    relative_residual: float
    converged_reason: int | None = None
    iterations: int | None = None


def assemble_system(problem) -> AssembledSystem:
    """
    Assemble the global stiffness matrix and load vector.

    The initial implementation supports serial execution only.

    Homogeneous Dirichlet conditions are applied using Firedrake's
    boundary-condition assembly convention.
    """

    # Initial implementation: serial execution only.
    if problem.mesh.comm.size != 1:
        raise NotImplementedError(
            "The initial assembled reference supports serial execution only."
        )

    # Assemble the global stiffness matrix.
    A = assemble(
        problem.a,
        bcs=problem.bc,
        mat_type="aij",
    )

    # Assemble the constrained load vector.
    b = assemble(
        problem.L,
        bcs=problem.bc,
    )

    # Identify boundary degrees of freedom.
    boundary_dofs = np.asarray(
        problem.bc.nodes,
        dtype=np.int64,
    )

    # Total number of degrees of freedom.
    ndofs = problem.V.dim()

    # Identify unconstrained degrees of freedom.
    free_mask = np.ones(ndofs, dtype=bool)
    free_mask[boundary_dofs] = False

    free_dofs = np.flatnonzero(free_mask)

    return AssembledSystem(
        A=A,
        b=b,
        function_space=problem.V,
        boundary_dofs=boundary_dofs,
        free_dofs=np.flatnonzero(free_mask),
        ndofs=ndofs,
    )


def get_free_dofs(system: AssembledSystem) -> np.ndarray:
    """
    Return the unconstrained degrees of freedom.
    """

    return system.free_dofs.copy()


def _validate_vector(system, x, name="x"):

    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array"
        )

    if x.size != system.ndofs:
        raise ValueError(
            f"Expected {name} of length {system.ndofs}, "
            f"received length {x.size}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"{name} contains NaN or infinite values"
        )

    return x

def apply_assembled_operator(
        system: AssembledSystem,
        x,
    ) -> np.ndarray:
    """
    Apply the assembled constrained matrix to a serial vector.

    Computes:

        y = A*x

    Parameters
    ----------
    system:
        Previously assembled finite-element system.

    x:
        One-dimensional NumPy vector.

    Returns
    -------
    numpy.ndarray
        Result of the assembled matrix-vector product.
    """

    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError(
            "x must be a one-dimensional array"
        )

    if x.size != system.ndofs:
        raise ValueError(
            f"Expected vector of length {system.ndofs}, "
            f"received length {x.size}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "x contains NaN or infinite values"
        )

    A_petsc = system.A.petscmat

    x_petsc = A_petsc.createVecRight()
    y_petsc = A_petsc.createVecLeft()

    x_petsc.setArray(x.copy())

    A_petsc.mult(x_petsc, y_petsc)

    y = y_petsc.getArray(readonly=True).copy()

    return y

def enforce_homogeneous_boundary(
        system: AssembledSystem,
        x,
    ) -> np.ndarray:
    """
    Return a copy of x with all boundary DOFs set to zero.

    The original input vector is not modified.
    """

    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError(
            "x must be a one-dimensional array"
        )

    if x.size != system.ndofs:
        raise ValueError(
            f"Expected vector of length {system.ndofs}, "
            f"received length {x.size}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "x contains NaN or infinite values"
        )

    x_bc = x.copy()

    x_bc[system.boundary_dofs] = 0.0

    return x_bc


def compute_residual_vector(
      system: AssembledSystem,
      solution: Function,
    ) -> np.ndarray:
    """
    Compute the algebraic residual:

        r = b - A*u_h

    The initial implementation supports serial execution.
    """
    # First check function-space compatibility.
    if solution.function_space() is not system.function_space:
        raise ValueError(
            "The solution belongs to a different function space."
        )

    # Only validate vector data after compatibility is established.
    u = _validate_vector(
        system,
        solution.dat.data_ro,
        name="solution",
    )

    Au = apply_assembled_operator(system, u)

    b = np.asarray(
        system.b.dat.data_ro
    ).copy()

    return b - Au


def compute_residual_norm(
        system: AssembledSystem,
        solution: Function,
    ) -> float:
    """
    Compute the Euclidean norm of the algebraic residual.
    """

    residual = compute_residual_vector(
        system,
        solution,
    )

    return float(np.linalg.norm(residual))


def compute_relative_residual(
        system: AssembledSystem,
        solution: Function,
    ) -> float:
    """
    Compute the relative algebraic residual:

        ||b - A*u_h||_2 / ||b||_2
    """

    residual = compute_residual_vector(
        system,
        solution,
    )

    rhs = np.asarray(
        system.b.dat.data_ro,
        dtype=np.float64,
    )

    denominator = np.linalg.norm(rhs)

    if denominator == 0.0:
        return float(np.linalg.norm(residual))

    return float(
        np.linalg.norm(residual) / denominator
    )

DEFAULT_ASSEMBLED_SOLVER_PARAMETERS = {
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-12,
    "ksp_atol": 1.0e-14,
    "ksp_max_it": 1000,
    "ksp_error_if_not_converged": None,
    "pc_type": "jacobi",
}

DIRECT_SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
}

def solve_assembled(
        problem,
        system=None,
        solver_parameters=None,
    ) -> AssembledSolveResult:
    """
    Solve the explicitly assembled finite-element system.

    If a previously assembled system is supplied, it is reused.

    Parameters
    ----------
    problem:
        Manufactured Poisson problem.

    system:
        Existing AssembledSystem. If omitted, assemble once.

    solver_parameters:
        PETSc solver parameters. Defaults to CG with Jacobi
        preconditioning.

    Returns
    -------
    AssembledSolveResult
        Numerical solution and algebraic residual diagnostics.
    """
    if system is None:
        system = assemble_system(problem)

    # Validate compatibility BEFORE creating the solver.
    if system.function_space is not problem.V:
        raise ValueError(
            "The supplied assembled system belongs to a "
            "different function space."
        )

    parameters = DEFAULT_ASSEMBLED_SOLVER_PARAMETERS.copy()

    if solver_parameters is not None:
        parameters.update(solver_parameters)

    uh = Function(
        problem.V,
        name="u_h_assembled",
    )

    solver = LinearSolver(
        system.A,
        solver_parameters=parameters,
    )

    solver.solve(
        uh,
        system.b,
    )

    residual_norm = compute_residual_norm(
        system,
        uh,
    )

    relative_residual = compute_relative_residual(
        system,
        uh,
    )

    ksp = solver.ksp

    converged_reason = int(ksp.getConvergedReason())
    iterations = int(ksp.getIterationNumber())

    return AssembledSolveResult(
        solution=uh,
        system=system,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        converged_reason=converged_reason,
        iterations=iterations,
    )
