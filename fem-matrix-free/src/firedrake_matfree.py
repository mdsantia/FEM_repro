
"""
Firedrake matrix-free implementation of the manufactured Poisson problem.

The Krylov operator is evaluated matrix-free from the variational form.

Firedrake assembles an AIJ preconditioning matrix, from which PETSc
constructs a Jacobi preconditioner using its diagonal.

The assembled solver remains the numerical reference implementation.

The initial validation interface supports serial execution only.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from firedrake import (
    Function,
    LinearVariationalProblem,
    LinearVariationalSolver,
)


# --------------------------------------------------
# Solver configuration
# --------------------------------------------------

DEFAULT_MATFREE_SOLVER_PARAMETERS = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-12,
    "ksp_atol": 1.0e-14,
    "ksp_max_it": 1000,
    "ksp_error_if_not_converged": None,
    "pc_type": "jacobi",
}


REQUIRED_OPERATOR_PARAMETERS = {
    "mat_type": "matfree",
    "pmat_type": "aij",
}


# --------------------------------------------------
# Result container
# --------------------------------------------------

@dataclass
class FiredrakeMatfreeResult:
    """
    Container for the Firedrake matrix-free solution
    and PETSc solver diagnostics.
    """

    solution: Function
    function_space: Any

    converged_reason: int
    iterations: int

    ksp_residual_norm: float

    operator_type: str
    preconditioner_matrix_type: str


# --------------------------------------------------
# Parameter validation
# --------------------------------------------------

def _validate_solver_parameters(
        solver_parameters=None,
    ) -> dict:
    """
    Merge solver parameters while preserving the required
    matrix-free operator and AIJ preconditioning matrix.

    User-supplied dictionaries are never modified.
    """

    if solver_parameters is not None and not isinstance(
        solver_parameters,
        dict,
    ):
        raise TypeError(
            "solver_parameters must be a dictionary or None."
        )

    parameters = (
        DEFAULT_MATFREE_SOLVER_PARAMETERS.copy()
    )

    if solver_parameters is not None:
        parameters.update(solver_parameters)

    # Validate the final configuration rather than only
    # checking individual user overrides.
    for key, expected in REQUIRED_OPERATOR_PARAMETERS.items():

        if parameters.get(key) != expected:
            raise ValueError(
                f"{key} must be '{expected}' for the "
                "Firedrake matrix-free implementation."
            )

    return parameters


# --------------------------------------------------
# Problem validation
# --------------------------------------------------

def _validate_problem(problem):
    """
    Validate the problem before constructing the solver.

    The current validation interface supports serial execution
    because subsequent reference comparisons use serial NumPy
    vectors and assembled PETSc matrices.
    """

    if problem.mesh.comm.size != 1:
        raise NotImplementedError(
            "The initial matrix-free validation interface "
            "supports serial execution only."
        )

    if problem.V.dim() <= 0:
        raise ValueError(
            "The supplied function space has no degrees of freedom."
        )


# --------------------------------------------------
# Matrix-free solver
# --------------------------------------------------

def solve_firedrake_matfree(
        problem,
        solver_parameters=None,
    ) -> FiredrakeMatfreeResult:
    """
    Solve the supplied variational problem using Firedrake's
    matrix-free operator implementation.

    The same bilinear form, linear form and boundary conditions
    as the assembled implementation are used.

    The operator is matrix-free, while the preconditioning
    matrix is explicitly assembled as AIJ.

    Parameters
    ----------
    problem:
        Manufactured Poisson problem.

    solver_parameters:
        Optional PETSc solver parameters.

        The following parameters are fixed:

            mat_type = matfree
            pmat_type = aij

    Returns
    -------
    FiredrakeMatfreeResult
        Solution and PETSc solver diagnostics.
    """

    # Validate the mathematical problem.
    _validate_problem(problem)

    # Validate and construct solver parameters.
    parameters = _validate_solver_parameters(
        solver_parameters
    )

    # Allocate the finite-element solution.
    solution = Function(
        problem.V,
        name="u_h_firedrake_matfree",
    )

    # Construct the variational problem using the original
    # mathematical forms and boundary conditions.
    variational_problem = LinearVariationalProblem(
        problem.a,
        problem.L,
        solution,
        bcs=problem.bc,
    )

    # Construct the Firedrake matrix-free solver.
    solver = LinearVariationalSolver(
        variational_problem,
        solver_parameters=parameters,
    )

    # Solve the discrete system.
    solver.solve()

    # Obtain PETSc Krylov solver diagnostics.
    ksp = solver.snes.getKSP()

    converged_reason = int(
        ksp.getConvergedReason()
    )

    iterations = int(
        ksp.getIterationNumber()
    )

    ksp_residual_norm = float(
        ksp.getResidualNorm()
    )

    # Explicit convergence validation.
    if converged_reason <= 0:
        raise RuntimeError(
            "Firedrake matrix-free solver did not converge. "
            f"PETSc convergence reason: {converged_reason}."
        )

    # Validate the resulting solution.
    solution_values = solution.dat.data_ro

    if not np.isfinite(solution_values).all():
        raise FloatingPointError(
            "The Firedrake matrix-free solution contains "
            "NaN or infinite values."
        )

    # Inspect the actual PETSc operator and preconditioning
    # matrix constructed by Firedrake.
    operator, preconditioner_matrix = ksp.getOperators()

    operator_type = str(
        operator.getType()
    )

    preconditioner_matrix_type = str(
        preconditioner_matrix.getType()
    )

    return FiredrakeMatfreeResult(
        solution=solution,
        function_space=problem.V,
        converged_reason=converged_reason,
        iterations=iterations,
        ksp_residual_norm=ksp_residual_norm,
        operator_type=operator_type,
        preconditioner_matrix_type=(
            preconditioner_matrix_type
        ),
    )
