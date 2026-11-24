

"""
Independent element-local P1 Poisson operator.

Firedrake supplies:
    - Mesh and function-space metadata.
    - Scalar DoF numbering and cell connectivity.
    - Boundary DoFs.
    - Right-hand-side assembly.
    - The assembled AIJ preconditioning matrix.

The stiffness operator action is independently implemented using
NumPy gather, element-local arithmetic, and scatter-add.

Two storage modes are supported:

    precompute_matrices=True:
        Stores one 3x3 stiffness matrix per triangle.
        This is an element-local precomputed operator.

    precompute_matrices=False:
        Stores element gradients and areas and evaluates the
        stiffness action directly during every application.
        This is the matrix-free element-action mode.

The NumPy scatter implementation is a correctness reference,
not a performance-equivalent replacement for Firedrake/PyOP2.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from firedrake import (
    Function,
    SpatialCoordinate,
    assemble,
)

from firedrake.petsc import PETSc


DEFAULT_LOCAL_SOLVER_PARAMETERS = {
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-12,
    "ksp_atol": 1.0e-14,
    "ksp_max_it": 1000,
    "pc_type": "jacobi",
}


@dataclass
class LocalOperatorResult:
    solution: Function
    function_space: Any
    converged_reason: int
    iterations: int
    ksp_residual_norm: float
    operator_type: str
    preconditioner_matrix_type: str


def p1_triangle_geometry(vertices):
    """
    Compute P1 triangle geometry.

    Returns
    -------
    area : float
        Positive physical triangle area.

    gradients : ndarray, shape (3, 2)
        Physical gradients of the three P1 basis functions.
    """

    vertices = np.asarray(
        vertices,
        dtype=np.float64,
    )

    if vertices.shape != (3, 2):
        raise ValueError(
            "vertices must have shape (3, 2)."
        )

    if not np.isfinite(vertices).all():
        raise ValueError(
            "Triangle coordinates must be finite."
        )

    x = vertices[:, 0]
    y = vertices[:, 1]

    determinant = (
        (x[1] - x[0]) * (y[2] - y[0])
        - (x[2] - x[0]) * (y[1] - y[0])
    )

    edge_lengths_squared = np.array([
        np.sum((vertices[1] - vertices[0]) ** 2),
        np.sum((vertices[2] - vertices[1]) ** 2),
        np.sum((vertices[0] - vertices[2]) ** 2),
    ])

    scale = np.max(edge_lengths_squared)

    if scale == 0.0:
        raise ValueError(
            "Degenerate triangle encountered."
        )

    tolerance = (
        100.0
        * np.finfo(np.float64).eps
        * scale
    )

    if abs(determinant) <= tolerance:
        raise ValueError(
            "Degenerate triangle encountered."
        )

    b = np.array([
        y[1] - y[2],
        y[2] - y[0],
        y[0] - y[1],
    ])

    c = np.array([
        x[2] - x[1],
        x[0] - x[2],
        x[1] - x[0],
    ])

    gradients = np.column_stack(
        (b, c)
    ) / determinant

    area = 0.5 * abs(determinant)

    return area, gradients


def p1_triangle_stiffness(vertices):
    """
    Compute the 3x3 P1 Laplacian stiffness matrix.
    """

    area, gradients = p1_triangle_geometry(
        vertices
    )

    return (
        area
        * gradients
        @ gradients.T
    )


class LocalP1Operator:

    def __init__(
        self,
        problem,
        precompute_matrices=True,
    ):

        if problem.mesh.comm.size != 1:
            raise NotImplementedError(
                "The initial local operator supports "
                "serial execution only."
            )

        if problem.degree != 1:
            raise NotImplementedError(
                "LocalP1Operator supports only P1 elements."
            )

        if (
            problem.mesh.topological_dimension != 2
            or problem.mesh.geometric_dimension != 2
        ):
            raise NotImplementedError(
                "LocalP1Operator requires a two-dimensional mesh."
            )

        self.problem = problem
        self.V = problem.V

        self.num_dofs = int(
            problem.V.dim()
        )

        self.precompute_matrices = bool(
            precompute_matrices
        )

        # Scalar P1 DoF connectivity.
        self.dof_connectivity = np.asarray(
            problem.V.cell_node_map().values,
            dtype=np.int64,
        ).copy()

        if (
            self.dof_connectivity.ndim != 2
            or self.dof_connectivity.shape[1] != 3
        ):
            raise ValueError(
                "Expected three scalar P1 DoFs per triangle."
            )

        if (
            np.any(self.dof_connectivity < 0)
            or np.any(self.dof_connectivity >= self.num_dofs)
        ):
            raise ValueError(
                "Invalid scalar DoF connectivity."
            )

        self.num_cells = (
            self.dof_connectivity.shape[0]
        )

        # Boundary DoFs.
        self.boundary_dofs = np.asarray(
            problem.bc.nodes,
            dtype=np.int64,
        ).copy()

        if (
            np.any(self.boundary_dofs < 0)
            or np.any(self.boundary_dofs >= self.num_dofs)
        ):
            raise ValueError(
                "Invalid boundary DoF indices."
            )

        if np.unique(self.boundary_dofs).size != (
            self.boundary_dofs.size
        ):
            raise ValueError(
                "Boundary DoF indices must be unique."
            )

        # Construct physical coordinates in scalar-DoF ordering.
        #
        # This ensures that cell_coordinates[e, i] and
        # vector[dof_connectivity[e, i]] refer to the
        # same local P1 basis function.

        x_coordinate, y_coordinate = SpatialCoordinate(
            problem.mesh
        )

        x_dof = Function(
            self.V
        ).interpolate(
            x_coordinate
        ).dat.data_ro.copy()

        y_dof = Function(
            self.V
        ).interpolate(
            y_coordinate
        ).dat.data_ro.copy()

        self.dof_coordinates = np.column_stack(
            (x_dof, y_dof)
        )

        if self.dof_coordinates.shape != (
            self.num_dofs,
            2,
        ):
            raise ValueError(
                "Coordinate and scalar DoF dimensions do not agree."
            )

        self.cell_coordinates = (
            self.dof_coordinates[
                self.dof_connectivity
            ]
        )

        self.areas = None
        self.gradients = None
        self.element_matrices = None

        self._compute_element_geometry()

        if self.precompute_matrices:

            self.element_matrices = (
                self.areas[:, None, None]
                * np.einsum(
                    "eik,ejk->eij",
                    self.gradients,
                    self.gradients,
                )
            )

    def _compute_element_geometry(self):

        self.areas = np.empty(
            self.num_cells,
            dtype=np.float64,
        )

        self.gradients = np.empty(
            (self.num_cells, 3, 2),
            dtype=np.float64,
        )

        for cell in range(self.num_cells):

            area, gradients = p1_triangle_geometry(
                self.cell_coordinates[cell]
            )

            self.areas[cell] = area

            self.gradients[cell] = gradients

    def apply(self, vector):
        """
        Apply the constrained local stiffness operator.

        Boundary convention:
            - Boundary input values are eliminated.
            - Boundary output rows act as identity.

        This corresponds to homogeneous Dirichlet
        elimination used by the assembled reference.
        """

        vector = np.asarray(
            vector,
            dtype=np.float64,
        )

        if vector.ndim != 1:
            raise ValueError(
                "Input vector must be one-dimensional."
            )

        if vector.size != self.num_dofs:
            raise ValueError(
                f"Expected vector of length {self.num_dofs}, "
                f"received length {vector.size}."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Input vector contains nonfinite values."
            )

        # Eliminate boundary input values.
        constrained_input = vector.copy()

        constrained_input[
            self.boundary_dofs
        ] = 0.0

        # Gather element-local DoFs.
        element_values = constrained_input[
            self.dof_connectivity
        ]

        if self.precompute_matrices:

            # Precomputed element stiffness action.
            local_result = np.einsum(
                "eij,ej->ei",
                self.element_matrices,
                element_values,
            )

        else:

            # Direct element-gradient action.
            local_gradient_values = np.einsum(
                "eik,ei->ek",
                self.gradients,
                element_values,
            )

            local_result = (
                self.areas[:, None]
                * np.einsum(
                    "eik,ek->ei",
                    self.gradients,
                    local_gradient_values,
                )
            )

        # Scatter-add element contributions.
        result = np.zeros(
            self.num_dofs,
            dtype=np.float64,
        )

        np.add.at(
            result,
            self.dof_connectivity.ravel(),
            local_result.ravel(),
        )

        # Identity boundary rows.
        result[
            self.boundary_dofs
        ] = vector[
            self.boundary_dofs
        ]

        return result

    def mult(self, matrix, x, y):
        """
        PETSc Python matrix multiplication callback.
        """

        input_values = x.getArray(
            readonly=True
        )

        output_values = y.getArray()

        output_values[:] = self.apply(
            input_values
        )


def _validate_solver_parameters(
    solver_parameters,
):

    if solver_parameters is None:

        parameters = (
            DEFAULT_LOCAL_SOLVER_PARAMETERS.copy()
        )

    elif isinstance(solver_parameters, dict):

        parameters = (
            DEFAULT_LOCAL_SOLVER_PARAMETERS.copy()
        )

        parameters.update(
            solver_parameters
        )

    else:

        raise TypeError(
            "solver_parameters must be a dictionary."
        )

    allowed = set(
        DEFAULT_LOCAL_SOLVER_PARAMETERS
    )

    unknown = (
        set(parameters) - allowed
    )

    if unknown:

        raise ValueError(
            f"Unknown solver parameters: {sorted(unknown)}"
        )

    if parameters["ksp_type"] != "cg":

        raise ValueError(
            "The local solver requires ksp_type='cg'."
        )

    if parameters["pc_type"] != "jacobi":

        raise ValueError(
            "The local solver requires pc_type='jacobi'."
        )

    try:

        rtol = float(
            parameters["ksp_rtol"]
        )

        atol = float(
            parameters["ksp_atol"]
        )

        max_it = int(
            parameters["ksp_max_it"]
        )

    except (TypeError, ValueError, OverflowError) as exc:

        raise ValueError(
            "Invalid numerical solver parameter."
        ) from exc

    if not np.isfinite(rtol) or rtol < 0.0:

        raise ValueError(
            "ksp_rtol must be finite and nonnegative."
        )

    if not np.isfinite(atol) or atol < 0.0:

        raise ValueError(
            "ksp_atol must be finite and nonnegative."
        )

    if isinstance(parameters["ksp_max_it"], bool):

        raise ValueError(
            "ksp_max_it must be a positive integer."
        )

    if (
        max_it <= 0
        or max_it != parameters["ksp_max_it"]
    ):

        raise ValueError(
            "ksp_max_it must be a positive integer."
        )

    parameters["ksp_rtol"] = rtol
    parameters["ksp_atol"] = atol
    parameters["ksp_max_it"] = max_it

    return parameters


def solve_local_operator(
    problem,
    solver_parameters=None,
    precompute_matrices=True,
):
    """
    Solve the P1 Poisson problem using the local operator.

    The local stiffness operator is supplied through a PETSc
    Python matrix. The assembled AIJ matrix is used only
    as the preconditioning matrix.
    """

    parameters = _validate_solver_parameters(
        solver_parameters
    )

    local_operator = LocalP1Operator(
        problem,
        precompute_matrices=precompute_matrices,
    )

    ndofs = local_operator.num_dofs

    # Assemble the constrained right-hand side.
    rhs = assemble(
        problem.L,
        bcs=problem.bc,
    )

    # Assemble the constrained AIJ preconditioning matrix.
    preconditioning_matrix = assemble(
        problem.a,
        bcs=problem.bc,
        mat_type="aij",
    )

    solution = Function(
        problem.V,
        name="u_h_local_operator",
    )

    # Create the independent PETSc Python operator.
    operator = PETSc.Mat().createPython(
        size=(ndofs, ndofs),
        context=local_operator,
        comm=PETSc.COMM_SELF,
    )

    operator.setUp()

    operator.setOption(
        PETSc.Mat.Option.SYMMETRIC,
        True,
    )

    ksp = PETSc.KSP().create(
        comm=PETSc.COMM_SELF
    )

    converged_reason = None
    iterations = None
    residual_norm = None
    operator_type = None
    preconditioner_matrix_type = None

    try:

        ksp.setType(
            parameters["ksp_type"]
        )

        ksp.setTolerances(
            rtol=parameters["ksp_rtol"],
            atol=parameters["ksp_atol"],
            max_it=parameters["ksp_max_it"],
        )

        ksp.getPC().setType(
            parameters["pc_type"]
        )

        # This Firedrake version exposes the underlying
        # PETSc matrix through `.petscmat`.
        #
        # Keep the wrapper `preconditioning_matrix` alive
        # throughout the solve.
        pmat = preconditioning_matrix.petscmat

        ksp.setOperators(
            operator,
            pmat,
        )

        with rhs.dat.vec_ro as b:

            with solution.dat.vec as x:

                # Start from a deterministic zero initial guess.
                x.set(0.0)

                ksp.solve(
                    b,
                    x,
                )

        converged_reason = int(
            ksp.getConvergedReason()
        )

        iterations = int(
            ksp.getIterationNumber()
        )

        residual_norm = float(
            ksp.getResidualNorm()
        )

        operator_type = operator.getType()

        preconditioner_matrix_type = (
            pmat.getType()
        )

    finally:

        ksp.destroy()
        operator.destroy()

    if converged_reason is None:

        raise RuntimeError(
            "The local-operator solve terminated before "
            "PETSc convergence information was available."
        )

    if converged_reason <= 0:

        raise RuntimeError(
            "Local-operator solver failed to converge. "
            f"PETSc converged reason: {converged_reason}."
        )

    solution_values = (
        solution.dat.data_ro
    )

    if not np.isfinite(
        solution_values
    ).all():

        raise FloatingPointError(
            "The local-operator solution contains "
            "NaN or infinite values."
        )

    if not np.isfinite(
        residual_norm
    ):

        raise FloatingPointError(
            "The local-operator KSP residual norm "
            "is not finite."
        )

    return LocalOperatorResult(
        solution=solution,
        function_space=problem.V,
        converged_reason=converged_reason,
        iterations=iterations,
        ksp_residual_norm=residual_norm,
        operator_type=operator_type,
        preconditioner_matrix_type=(
            preconditioner_matrix_type
        ),
    )
