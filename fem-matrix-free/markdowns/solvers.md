# Matrix-Free Finite Element Methods: Implementation Design

## 1. Project objective

This project develops and validates three implementations of the same conforming finite-element discretization of the Poisson equation:

1. an explicitly assembled sparse operator;
2. Firedrake's matrix-free operator;
3. an independent element-local operator implemented with NumPy and exposed to PETSc through a Python matrix.

The implementations are developed in stages so that mathematical correctness, algebraic equivalence, solver behavior, and finite-element convergence can be tested separately. The assembled implementation is the numerical reference. The independent local operator is the principal object of study because its gather, element-action, and scatter-add structure makes the local data requirements explicit.

The manufactured problem is

$$
-\Delta u=f \quad \text{in } \Omega=(0,1)^2,
\qquad
u=0 \quad \text{on } \partial\Omega,
$$

with exact solution and forcing

$$
u_{\mathrm{exact}}(x,y)=\sin(\pi x)\sin(\pi y),
$$

$$
f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y).
$$

The conforming finite-element space uses continuous Lagrange elements on affine triangular meshes. The initial implementation and experiments use polynomial degree one.

The expected discretization-error rates are

$$
\|u-u_h\|_{L^2(\Omega)}=O(h^2),
\qquad
|u-u_h|_{H^1(\Omega)}=O(h).
$$

---

## 2. File structure

```text
fem-matrix-free/
├── README.md
├── environment.yml or requirements notes
├── src/
│   ├── __init__.py
│   ├── problem.py
│   ├── assembled.py
│   ├── firedrake_matfree.py
│   ├── local_operator.py
│   ├── functionals.py
│   └── diagnostics.py
├── experiments/
│   ├── convergence.py
│   └── operator_equivalence.py
├── tests/
│   ├── test_problem.py
│   ├── test_assembled.py
│   ├── test_firedrake_matfree.py
│   ├── test_single_element.py
│   ├── test_local_operator.py
│   └── test_functionals.py
└── results/
```

The current implemented sequence is:

```text
problem.py
    ↓
assembled.py
    ↓
firedrake_matfree.py
    ↓
local_operator.py
```

The remaining modules will use these implementations rather than redefine the PDE or operators.

---

# 3. `src/problem.py`

## 3.1 Purpose

`problem.py` is the mathematical source of truth. It defines the mesh, function space, manufactured solution, forcing term, weak form, boundary condition, and finite-element error measures.

It must not choose a matrix representation, preconditioner, or Krylov solver. Those decisions belong to the operator-specific modules.

## 3.2 Mathematical definition

The weak problem is: find $u\in H_0^1(\Omega)$ such that

$$
a(u,v)=L(v)
\qquad \text{for every } v\in H_0^1(\Omega),
$$

where

$$
a(u,v)=\int_\Omega \nabla u\cdot\nabla v\,dx,
$$

and

$$
L(v)=\int_\Omega f v\,dx.
$$

The analytical gradient used for the $H^1$-seminorm error is

$$
\nabla u_{\mathrm{exact}}=
\begin{bmatrix}
\pi\cos(\pi x)\sin(\pi y)\\
\pi\sin(\pi x)\cos(\pi y)
\end{bmatrix}.
$$

## 3.3 Main data structure

`PoissonProblem` stores:

- the mesh;
- the finite-element space `V`;
- trial and test functions;
- the exact solution;
- the analytical exact gradient;
- the forcing term;
- the bilinear form `a`;
- the linear form `L`;
- the homogeneous Dirichlet condition;
- the nominal mesh size `h`;
- polynomial degree;
- refinement level;
- triangle-diagonal convention.

This object is passed unchanged to every solver implementation. Consequently, differences among assembled, Firedrake matrix-free, and independent local results cannot originate from separately defined PDEs.

## 3.4 Main functions

### `create_mesh(refinement_level, diagonal="left")`

Creates `UnitSquareMesh(N, N)` with

$$
N=2^{\text{refinement level}},
\qquad h=\frac{1}{N}.
$$

The stored value $h$ is the Cartesian grid spacing. The maximum diameter of the resulting right triangles is $\sqrt{2}h$.

### `exact_solution(mesh)`

Returns the UFL expression $\sin(\pi x)\sin(\pi y)$.

### `exact_gradient(mesh)`

Returns the analytical gradient explicitly, independently of `grad(u_exact)`.

### `forcing_term(mesh)`

Returns $f=-\Delta u_{\mathrm{exact}}$.

### `create_problem(...)`

Creates the mesh, function space, forms, and boundary condition and returns a `PoissonProblem`.

### `empty_solution(problem, name="u_h")`

Allocates an unsolved Firedrake `Function` in the problem space.

### Error routines

The current draft includes:

```python
l2_error(problem, u_h)
h1_seminorm_error(problem, u_h)
energy_error(problem, u_h)
assemble_scalar(form)
```

For the unit-diffusion Poisson problem, the energy norm equals the $H^1$ seminorm.

## 3.5 Tests for `problem.py`

The current test draft verifies:

1. mesh and function-space construction;
2. positive function-space dimension;
3. exact-solution interpolation;
4. zero exact solution on boundary nodes;
5. agreement between the analytical forcing and $-\nabla\cdot\nabla u_{\mathrm{exact}}$;
6. successful assembly of the bilinear and linear forms;
7. positive energy for a nonzero finite-element function;
8. a finite and reasonably accurate numerical solution;
9. correct mesh-size value;
10. agreement between the analytical gradient and `grad(u_exact)`;
11. correct empty-solution allocation;
12. unit-square area from scalar assembly;
13. finite $L^2$, $H^1$-seminorm, and energy errors;
14. equality of the energy error and $H^1$-seminorm error;
15. rejection of invalid refinement levels and polynomial degrees;
16. observed second-order $L^2$ convergence.

### Recommended test organization

`tests/test_problem.py` should contain assertions only. The multi-level convergence table belongs in `experiments/convergence.py`. A compact two-level rate test may remain in the unit tests.

### Current cautions

- Firedrake versions may expose `geometric_dimension()` and `topological_dimension()` as methods.
- Test solves should use explicit solver parameters rather than environment-dependent defaults.
- `h=1/N` should be documented as nominal grid spacing, not triangle diameter.
- Tests should be run through `pytest` or an explicit runner, but not automatically when `src.problem` is imported.

---

# 4. `src/assembled.py`

## 4.1 Purpose

`assembled.py` constructs the classical sparse finite-element reference system. It provides the reference matrix, right-hand side, solution, matrix-vector action, boundary metadata, and algebraic residuals used to validate both matrix-free implementations.

It does not redefine the PDE. It consumes `problem.a`, `problem.L`, and `problem.bc` from `problem.py`.

## 4.2 Algebraic system

The assembled method forms

$$
A u_h=b,
$$

where $A$ is stored as a PETSc AIJ sparse matrix. Homogeneous Dirichlet conditions are applied using Firedrake's constrained assembly convention.

The initial implementation assumes:

- serial execution;
- identity rows at boundary DoFs;
- eliminated boundary coupling consistent with a symmetric constrained matrix.

## 4.3 Main data structures

### `AssembledSystem`

Stores:

- assembled matrix `A`;
- assembled load vector `b`;
- boundary DoFs;
- free DoFs;
- total number of DoFs;
- optionally, the originating function space for compatibility checks.

### `AssembledSolveResult`

Stores:

- the finite-element solution;
- the assembled system;
- absolute residual norm;
- relative residual;
- optional PETSc convergence reason;
- optional iteration count.

## 4.4 Main functions

### `assemble_system(problem)`

Assembles the AIJ matrix and constrained right-hand side and partitions DoFs into boundary and free sets.

### `get_free_dofs(system)`

Returns a copy of the free-DoF indices.

### `apply_assembled_operator(system, x)`

Computes

$$
y=Ax
$$

through PETSc. It validates vector dimension, length, and finite values and returns an independent NumPy array.

### `enforce_homogeneous_boundary(system, x)`

Returns a copy of `x` with boundary entries set to zero.

### Residual functions

```python
compute_residual_vector(system, solution)
compute_residual_norm(system, solution)
compute_relative_residual(system, solution)
```

They evaluate

$$
r=b-Au_h,
$$

and

$$
\rho=\frac{\|b-Au_h\|_2}{\|b\|_2}.
$$

### `solve_assembled(...)`

Solves the explicitly assembled system. The draft supports:

- CG with Jacobi as the default experiment configuration;
- `preonly` with LU as the small-problem direct reference.

## 4.5 Tests for `assembled.py`

The current staged tests verify:

1. matrix and vector dimensions;
2. boundary-DoF identification;
3. free-DoF identification;
4. complete and nonoverlapping DoF partition;
5. finite matrix and right-hand-side entries;
6. agreement with direct assembly from the original problem forms;
7. matrix symmetry;
8. positive definiteness on free DoFs;
9. identity boundary-row convention;
10. matrix-vector application against a dense small-matrix product;
11. preservation of input arrays;
12. rejection of invalid vectors;
13. boundary-enforcement behavior;
14. residual calculations;
15. direct and iterative solver convergence;
16. agreement of direct and CG solutions;
17. satisfaction of boundary conditions;
18. finite-element accuracy;
19. error reduction under mesh refinement;
20. expected $L^2$ convergence rate.

### Important test refinements

- Tests must be in `tests/test_assembled.py`, not inside `src/assembled.py`.
- Dense extraction is limited to small systems. It must not be used in refinement experiments.
- Positive definiteness is best tested through eigenvalues of the free-free block for a small mesh, rather than one random quadratic form.
- Boundary columns should be checked in addition to boundary rows because CG requires symmetry.
- Function-space compatibility should be checked when reusing an `AssembledSystem`.
- The installed Firedrake version currently exposes `.petscmat`; code should not assume `.petscmat_ro` exists.

---

# 5. `src/firedrake_matfree.py`

## 5.1 Purpose

`firedrake_matfree.py` solves the same discrete problem while delegating matrix-free operator evaluation to Firedrake and PETSc.

The operator is implicit:

```python
"mat_type": "matfree"
```

The preconditioning matrix is explicitly assembled:

```python
"pmat_type": "aij"
```

With

```python
"pc_type": "jacobi"
```

PETSc uses the diagonal of the assembled AIJ preconditioning matrix. This distinction should be stated precisely: the operator is matrix-free, while the Jacobi preconditioner is constructed from an explicitly assembled matrix.

## 5.2 Main data structure

`FiredrakeMatfreeResult` stores:

- matrix-free finite-element solution;
- function space;
- PETSc convergence reason;
- iteration count;
- optionally the final KSP residual norm;
- operator matrix type;
- preconditioning matrix type.

## 5.3 Solver configuration

The default configuration is:

```python
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
```

The matrix representation is fixed by the module contract. Solver tolerances may be overridden without mutating the defaults.

## 5.4 Main functions

### `_validate_solver_parameters(...)`

Merges user parameters with defaults while enforcing:

- `mat_type="matfree"`;
- `pmat_type="aij"`;
- dictionary input;
- preservation of default and caller dictionaries.

### `solve_firedrake_matfree(problem, solver_parameters=None)`

The solve proceeds through:

1. allocating a solution `Function`;
2. constructing `LinearVariationalProblem` from the shared forms;
3. constructing `LinearVariationalSolver`;
4. solving with PETSc CG;
5. extracting the KSP convergence reason and iteration count;
6. checking convergence;
7. checking that the solution is finite;
8. recording operator and preconditioner matrix types when available.

## 5.5 Tests for `firedrake_matfree.py`

The current draft tests:

1. result type;
2. function-space identity;
3. solution-vector dimensions;
4. finite solution values;
5. positive PETSc convergence reason;
6. nonzero iteration count for the manufactured problem;
7. homogeneous boundary conditions;
8. preservation of default parameter dictionaries;
9. acceptance of custom tolerances;
10. rejection of an assembled operator request;
11. rejection of a non-AIJ preconditioning matrix;
12. rejection of invalid parameter containers;
13. agreement with the direct assembled solution;
14. agreement of $L^2$ errors;
15. agreement of $H^1$-seminorm errors;
16. absolute error thresholds;
17. expected convergence rates.

### Essential structural tests

The defining implementation property must also be tested:

- the Krylov operator is not AIJ and is represented by Firedrake's matrix-free/Python-shell type;
- the preconditioning matrix is AIJ;
- the matrix-free solution has a small residual when inserted into the assembled reference system.

The last test establishes algebraic equivalence more directly than matching FEM error values.

### Timing rule

Reference-matrix assembly must occur outside any timed matrix-free solve. Otherwise, the timing would include validation work that is not part of the matrix-free algorithm.

---

# 6. `src/local_operator.py`

## 6.1 Purpose

`local_operator.py` independently evaluates the $P_1$ Poisson stiffness operator. Firedrake supplies mesh and function-space metadata, but Firedrake's assembled and matrix-free operator actions are not used to compute $Ax$.

For an input vector $x$, the global action is

$$
A_hx=
\sum_{T\in\mathcal T_h}
R_T^T A_T R_Tx.
$$

The computation consists of:

1. gather three scalar DoFs for each triangle;
2. evaluate the local stiffness action;
3. scatter-add three local contributions into the global output;
4. impose the constrained boundary convention.

The initial implementation is restricted to:

- two-dimensional triangular meshes;
- continuous Lagrange $P_1$ elements;
- serial execution;
- homogeneous Dirichlet conditions.

## 6.2 Element geometry

For triangle vertices $(x_i,y_i)$, define the signed double area

$$
\det J=
(x_2-x_1)(y_3-y_1)
-(x_3-x_1)(y_2-y_1).
$$

The physical area is

$$
|T|=\frac12|\det J|.
$$

With

$$
b=
\begin{bmatrix}
y_2-y_3\\
y_3-y_1\\
y_1-y_2
\end{bmatrix},
\qquad
c=
\begin{bmatrix}
x_3-x_2\\
x_1-x_3\\
x_2-x_1
\end{bmatrix},
$$

the basis-function gradients are

$$
\nabla\phi_i=
\frac{1}{\det J}
\begin{bmatrix}
b_i\\c_i
\end{bmatrix}.
$$

The element stiffness matrix is

$$
(A_T)_{ij}
=
|T|\nabla\phi_i\cdot\nabla\phi_j.
$$

## 6.3 Geometry and stiffness helpers

### `p1_triangle_geometry(vertices)`

Validates a `(3, 2)` coordinate array, rejects nonfinite and degenerate triangles, and returns:

- positive triangle area;
- a `(3, 2)` array of physical basis gradients.

The degeneracy tolerance scales with the squared edge lengths.

### `p1_triangle_stiffness(vertices)`

Returns the $3\times3$ scalar Laplacian stiffness matrix.

For the reference triangle

$$
(0,0),\quad(1,0),\quad(0,1),
$$

the exact matrix is

$$
A_T=
\frac12
\begin{bmatrix}
2&-1&-1\\
-1&1&0\\
-1&0&1
\end{bmatrix}.
$$

## 6.4 `LocalP1Operator`

The class extracts scalar cell-to-DoF connectivity and constructs physical coordinates in scalar-DoF order by interpolating the coordinate functions into `problem.V`.

This alignment is important: `cell_coordinates[e, i]` and `vector[dof_connectivity[e, i]]` must refer to the same local basis function.

The class stores:

- scalar DoF connectivity;
- boundary DoFs;
- scalar-DoF physical coordinates;
- cell coordinates;
- element areas;
- element basis gradients;
- optional element stiffness matrices.

### Storage mode 1: precomputed element matrices

With

```python
precompute_matrices=True
```

one $3\times3$ stiffness matrix is stored per triangle. The action is

$$
y_T=A_Tx_T.
$$

This is an element-local precomputed operator. It avoids the global sparse matrix but stores nine floating-point entries per cell.

### Storage mode 2: direct element action

With

```python
precompute_matrices=False
```

the class stores element areas and gradients. Each application evaluates

$$
g_T=\sum_{j=1}^3 x_j\nabla\phi_j,
$$

followed by

$$
(y_T)_i=|T|\nabla\phi_i\cdot g_T.
$$

This is the closer matrix-free element-action model because the local stiffness matrices are not materialized.

## 6.5 Boundary convention

Before gathering element values, the implementation sets input boundary entries to zero. After scattering all element contributions, it sets

$$
y_i=x_i
\qquad \text{for boundary DoFs }i.
$$

Thus:

- boundary rows act as identity;
- boundary-column coupling is eliminated from free equations;
- the constrained operator remains symmetric;
- the action is compatible with CG and the assembled constrained reference.

## 6.6 PETSc interface and solve

`LocalP1Operator.mult(matrix, x, y)` is the multiplication callback for a PETSc Python matrix.

`solve_local_operator(...)` performs the following:

1. validates CG/Jacobi solver parameters;
2. constructs `LocalP1Operator`;
3. assembles the constrained right-hand side;
4. assembles a separate AIJ matrix for Jacobi preconditioning;
5. creates a PETSc Python matrix for the independent operator;
6. marks the operator symmetric;
7. solves with PETSc CG;
8. records convergence reason, iteration count, residual norm, and matrix types;
9. destroys the PETSc KSP and Python matrix after the solve;
10. returns the Firedrake solution in `LocalOperatorResult`.

The installed Firedrake version exposes the PETSc matrix through:

```python
preconditioning_matrix.petscmat
```

rather than `petscmat_ro`. The wrapper object must remain alive for the full solve.

## 6.7 Tests for `local_operator.py`

The current manual suite in `tests/test_local_operator.py` uses `run_all()` and groups tests into stages.

### Stage 1: reference-triangle geometry

Tests:

- exact area and gradients for the reference triangle;
- exact reference stiffness matrix;
- orientation/permutation consistency;
- scaled-triangle geometry;
- rejection of invalid shape;
- rejection of nonfinite coordinates;
- rejection of repeated vertices;
- rejection of collinear vertices.

### Stage 2: operator construction and mesh geometry

Tests:

- total DoF count;
- total cell count;
- connectivity shape and index validity;
- DoF-coordinate shape;
- cell-coordinate shape;
- positive finite areas;
- total mesh area equal to one;
- gradient-array shape;
- partition-of-unity identity $\sum_i\nabla\phi_i=0$;
- valid and unique boundary DoFs.

### Stage 3: element-matrix properties

Tests:

- matrix storage in precomputed mode;
- absence of matrix storage in direct mode;
- `(num_cells, 3, 3)` matrix shape;
- finite values;
- symmetry;
- constant nullspace $A_T\mathbf 1=0$;
- positive semidefiniteness;
- rank two for every nondegenerate triangle;
- agreement with the standalone triangle-stiffness function.

### Stage 4: global action

Tests:

- correct output size;
- finite output;
- input preservation;
- independent output storage;
- linearity;
- symmetry;
- positive quadratic form;
- identity boundary rows;
- elimination of boundary influence on free output entries;
- zero-vector action;
- rejection of multidimensional input;
- rejection of incorrect vector length;
- rejection of nonfinite input.

### Stage 5: storage-mode equivalence

Tests:

- agreement between precomputed and direct actions on several seeded vectors;
- direct-mode linearity;
- direct-mode symmetry.

### Stage 6: assembled-operator equivalence

For several seeded vectors, the suite compares

$$
A_{\mathrm{local}}x
\quad\text{and}\quad
A_{\mathrm{assembled}}x.
$$

It reports or asserts the relative difference

$$
\delta_2=
\frac{
\|A_{\mathrm{local}}x-A_{\mathrm{assembled}}x\|_2
}{
\|A_{\mathrm{assembled}}x\|_2
}.
$$

Separate checks cover:

- precomputed action versus assembled action;
- direct action versus assembled action;
- agreement on free DoFs for zero-boundary inputs;
- boundary-action agreement.

The current target is approximately $10^{-12}$ on the small serial meshes.

### Stage 7: solver-parameter validation

Tests:

- default dictionary preservation;
- caller-dictionary preservation;
- custom tolerance acceptance;
- rejection of non-dictionary input;
- rejection of unknown options;
- enforcement of CG;
- enforcement of Jacobi;
- rejection of negative or nonfinite tolerances;
- rejection of zero, noninteger, or Boolean iteration limits.

### Stage 8: PETSc integration

Tests:

- result type;
- function-space identity;
- solution dimension;
- finite solution;
- positive convergence reason;
- nonzero iteration count;
- finite KSP residual;
- zero boundary values;
- PETSc operator type `python`;
- AIJ preconditioning matrix;
- distinct operator and preconditioner representations;
- convergence of direct-action mode;
- agreement between storage-mode solutions.

### Stage 9: assembled-solution comparison

Tests:

- coefficient-vector agreement;
- relative solution difference;
- direct-mode agreement;
- $L^2$-error agreement;
- $H^1$-seminorm-error agreement;
- small assembled relative residual for both local modes.

### Stage 10: FEM accuracy

Tests:

- absolute $L^2$ error threshold;
- absolute $H^1$-seminorm threshold;
- approximately second-order $L^2$ convergence;
- approximately first-order $H^1$ convergence.

## 6.8 Current implementation cautions

1. Depending on the Firedrake version, mesh dimensions may require method calls:

   ```python
   problem.mesh.topological_dimension()
   problem.mesh.geometric_dimension()
   ```

2. The NumPy `np.add.at` scatter is a correctness reference and is not expected to match PyOP2 performance.

3. The direct-action mode still stores geometry-derived gradients and areas. It avoids local matrices but is not a zero-storage operator.

4. The implementation is serial. MPI support would require owned/ghost DoF treatment, distributed PETSc vectors, and communication-aware scatter operations.

5. The AIJ matrix is used only for preconditioning and validation. Timed local-operator experiments must clearly state whether preconditioner assembly is included.

---

# 7. Validation hierarchy

The project validates each implementation at increasingly global levels.

## Level 1: analytical element identities

- exact triangle area;
- exact basis gradients;
- exact reference stiffness matrix;
- symmetry;
- constant nullspace;
- rank-two local Laplacian.

## Level 2: global operator identities

- linearity;
- symmetry;
- positive quadratic form;
- boundary convention;
- precomputed/direct-mode equality;
- local/assembled action equality.

## Level 3: algebraic solve

- positive PETSc convergence reason;
- finite residual;
- small assembled residual;
- agreement of solution coefficient vectors.

## Level 4: finite-element accuracy

- boundary satisfaction;
- $L^2$ error;
- $H^1$-seminorm error;
- expected convergence rates.

This hierarchy is intentional. A correct solution error alone cannot identify whether an implementation error lies in geometry, local algebra, scatter-add, boundary treatment, or Krylov configuration.

---

# 8. Boundary-condition comparison policy

Operator comparisons must use one explicit convention.

The current convention is the full constrained operator:

- zero boundary inputs before the element action;
- identity boundary rows after scatter-add;
- zero boundary right-hand-side entries.

Under this convention, full-vector comparison is valid:

$$
A_{\mathrm{local}}x
\approx
A_{\mathrm{assembled}}x.
$$

A second diagnostic comparison uses a vector satisfying the homogeneous condition and restricts the result to free DoFs. This isolates the element assembly from boundary-row replacement.

---

# 9. Immediate next modules

## 9.1 `src/diagnostics.py`

This module should centralize reusable comparison metrics:

```python
relative_vector_difference(reference, candidate)
absolute_infinity_difference(reference, candidate)
relative_free_dof_difference(system, reference, candidate)
observed_rate(error_coarse, error_fine, h_coarse, h_fine)
```

It should also hold solution and operator comparison records so experiments do not duplicate formulas.

## 9.2 `experiments/operator_equivalence.py`

For each refinement level and several seeded random vectors, compare:

1. assembled action;
2. precomputed local action;
3. direct local action;
4. optionally Firedrake's matrix-free action if a stable public action interface is available.

Report:

$$
\delta_2=
\frac{\|y_{\mathrm{candidate}}-y_{\mathrm{assembled}}\|_2}
{\|y_{\mathrm{assembled}}\|_2},
$$

and

$$
\delta_\infty=
\|y_{\mathrm{candidate}}-y_{\mathrm{assembled}}\|_\infty.
$$

Both full constrained and free-DoF results should be recorded.

## 9.3 `experiments/convergence.py`

For refinement levels such as $2,3,4,5$, report:

- $h$;
- number of DoFs;
- solver iterations;
- algebraic residual;
- $L^2$ error;
- $H^1$-seminorm error;
- observed rates;
- relative solution difference against assembled reference.

## 9.4 `src/functionals.py`

The proposal's three outputs should be implemented only after operator equivalence is established:

1. global average;
2. subdomain average;
3. smoothly weighted local output.

Each functional should have a reproducible exact or high-accuracy reference value and an explicit quadrature policy.

---

# 10. Running the current draft

From the project root, the source modules should be importable through `src`.

The current manual local-operator suite is run with:

```bash
python tests/test_local_operator.py
```

In a notebook:

```python
%run tests/test_local_operator.py
```

After rewriting a previously imported source module, reload it before running tests:

```python
import importlib
import src.local_operator

importlib.reload(src.local_operator)
```

Then execute the test file. The reload statement should not be embedded inside the test module.

---

# 11. Current status

The current draft establishes:

- one shared manufactured Poisson problem;
- an assembled AIJ reference system;
- a Firedrake matrix-free solve with assembled AIJ preconditioning data;
- an independent PETSc Python operator driven by NumPy element actions;
- two independent local storage modes;
- staged tests from individual triangles through FEM convergence.

The next scientific checkpoint is not additional solver functionality. It is a reproducible operator-equivalence experiment demonstrating that the independent gather–local-action–scatter implementation agrees with the assembled finite-element operator across mesh refinements and seeded input vectors.
