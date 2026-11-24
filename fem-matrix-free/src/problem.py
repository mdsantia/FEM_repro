"""
Mathematical definition of the manufactured Poisson problem.

Problem
-------
Find u in H_0^1(Omega) such that

    integral_Omega grad(u) . grad(v) dx
        = integral_Omega f v dx

for every v in H_0^1(Omega), where

    Omega = (0, 1)^2,
    u_exact(x, y) = sin(pi*x) sin(pi*y),
    f(x, y) = 2*pi^2 sin(pi*x) sin(pi*y).

The finite-element discretization uses continuous Lagrange elements on
triangular meshes. The default polynomial degree is one.
"""

from dataclasses import dataclass

from firedrake import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    UnitSquareMesh,
    as_vector,
    cos,
    dx,
    grad,
    inner,
    pi,
    sin,
)


@dataclass(frozen=True)
class PoissonProblem:
    """
    Container for one discretized manufactured Poisson problem.

    The object contains mathematical and discretization data only.
    Solver parameters belong in assembled.py or firedrake_matfree.py.
    """

    mesh: object
    V: object
    u: object
    v: object
    u_exact: object
    grad_u_exact: object
    forcing: object
    a: object
    L: object
    bc: object
    h: float # nominal mesh-size parameter
    degree: int
    refinement_level: int
    diagonal: str


def create_mesh(refinement_level: int, diagonal: str = "left"):
    """
    Create a uniformly refined triangular mesh of the unit square.

    Parameters
    ----------
    refinement_level:
        Defines N = 2**refinement_level cells along each coordinate
        direction. Each square is divided into two triangles.

    diagonal:
        Firedrake diagonal convention for splitting quadrilaterals into
        triangles. Keep this fixed throughout the numerical study.

    Returns
    -------
    mesh:
        Firedrake triangular mesh of the unit square.

    h:
        Nominal Cartesian mesh spacing, h = 1/N.
    """
    if refinement_level < 1:
        raise ValueError("refinement_level must be at least one")

    N = 2**refinement_level
    mesh = UnitSquareMesh(N, N, diagonal=diagonal)

    return mesh, 1.0 / N


def exact_solution(mesh):
    """Return the exact manufactured scalar solution."""
    x, y = SpatialCoordinate(mesh)
    return sin(pi * x) * sin(pi * y)


def exact_gradient(mesh):
    """
    Return the analytical gradient of the exact solution.

    This is defined explicitly rather than obtained through grad(u_exact)
    so that analytical and symbolic-gradient checks can be compared.
    """
    x, y = SpatialCoordinate(mesh)

    return as_vector(
        (
            pi * cos(pi * x) * sin(pi * y),
            pi * sin(pi * x) * cos(pi * y),
        )
    )


def forcing_term(mesh):
    """Return f = -Laplace(u_exact)."""
    x, y = SpatialCoordinate(mesh)
    return 2.0 * pi**2 * sin(pi * x) * sin(pi * y)


def create_problem(
    refinement_level: int,
    degree: int = 1,
    diagonal: str = "left",
) -> PoissonProblem:
    """
    Construct the variational problem on one mesh.

    Parameters
    ----------
    refinement_level:
        Mesh level with N = 2**refinement_level subdivisions per
        coordinate direction.

    degree:
        Polynomial degree of the continuous Lagrange space. The proposal
        uses degree=1, but the argument permits controlled extensions.

    diagonal:
        Orientation used to split each Cartesian cell into triangles.

    Returns
    -------
    PoissonProblem
        All objects needed to solve the manufactured Poisson problem.
    """
    if degree < 1:
        raise ValueError("degree must be at least one")

    mesh, h = create_mesh(
        refinement_level=refinement_level,
        diagonal=diagonal,
    )

    V = FunctionSpace(mesh, "Lagrange", degree)

    u = TrialFunction(V)
    v = TestFunction(V)

    u_exact = exact_solution(mesh)
    grad_u_exact = exact_gradient(mesh)
    forcing = forcing_term(mesh)

    # Weak formulation:
    #
    #   a(u, v) = integral grad(u) . grad(v) dx
    #   L(v)    = integral f v dx
    #
    a = inner(grad(u), grad(v)) * dx
    L = forcing * v * dx

    # The manufactured solution vanishes on the entire boundary.
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    return PoissonProblem(
        mesh=mesh,
        V=V,
        u=u,
        v=v,
        u_exact=u_exact,
        grad_u_exact=grad_u_exact,
        forcing=forcing,
        a=a,
        L=L,
        bc=bc,
        h=h,
        degree=degree,
        refinement_level=refinement_level,
        diagonal=diagonal,
    )


def empty_solution(problem: PoissonProblem, name: str = "u_h") -> Function:
    """
    Allocate an empty finite-element solution in the problem space.

    Solving is intentionally left to assembled.py or
    firedrake_matfree.py.
    """
    return Function(problem.V, name=name)


def l2_error(problem: PoissonProblem, u_h: Function) -> float:
    """
    Compute ||u_exact - u_h||_L2.
    """
    error_squared = (problem.u_exact - u_h) ** 2
    return assemble_scalar(error_squared * dx) ** 0.5


def h1_seminorm_error(problem: PoissonProblem, u_h: Function) -> float:
    """
    Compute |u_exact - u_h|_H1.

    The analytical exact gradient is used so this routine does not depend
    on differentiating the exact UFL expression again.
    """
    gradient_error = problem.grad_u_exact - grad(u_h)
    error_squared = inner(gradient_error, gradient_error)

    return assemble_scalar(error_squared * dx) ** 0.5


def energy_error(problem: PoissonProblem, u_h: Function) -> float:
    """
    Compute the Poisson energy norm of the discretization error.

    For the unit-diffusion Poisson problem, this equals the H1 seminorm
    error.
    """
    return h1_seminorm_error(problem, u_h)


def assemble_scalar(form) -> float:
    """
    Assemble a scalar UFL form.

    The local wrapper keeps all scalar assembly calls in one place and
    avoids repeating Firedrake-specific conversion code.
    """
    from firedrake import assemble

    return float(assemble(form))
