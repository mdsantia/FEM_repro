# Reproducibility Metadata

This file records the software, platform, solver, and experiment configuration used for the finite-element verification study.

## Project

- **working_directory:** /content/fem-matrix-free
- **python_executable:** /usr/bin/python3

## Platform

- **hostname:** d8fac81778a7
- **platform:** Linux-6.6.159+-x86_64-with-glibc2.39
- **system:** Linux
- **release:** 6.6.159+
- **machine:** x86_64
- **processor:** x86_64
- **cpu_model:** Intel(R) Xeon(R) CPU @ 2.20GHz
- **logical_cpu_count:** 2
- **physical_memory_bytes:** 13605818368

## Python

- **version:** 3.13.18 (main, Oct 26 2026, 15:28:15) [GCC 13.3.0]
- **version_info:** `{"major": 3, "minor": 13, "micro": 18}`

## Software

- **firedrake_version:** 2026.10.0
- **numpy_version:** 2.4.6
- **pandas_version:** 2.2.3
- **matplotlib_version:** 3.10.0
- **mpi4py_version:** 4.2.0.dev0
- **petsc4py_version:** 3.26.1
- **pyop2_version:** unknown
- **ufl_version:** 2026.2.0

## PETSc

- **version:** `[3, 26, 1]`
- **version_info:** `{"major": 3, "minor": 26, "subminor": 1, "release": true, "date": "unknown", "authorinfo": ["The PETSc Team", "petsc-maint@mcs.anl.gov", "https://petsc.org/"]}`
- **scalar_type:** <class 'numpy.float64'>
- **integer_type:** <class 'numpy.int32'>

## Environment

- **OMP_NUM_THREADS:** 1
- **OPENBLAS_NUM_THREADS:** Not available
- **MKL_NUM_THREADS:** Not available
- **BLIS_NUM_THREADS:** Not available
- **NUMEXPR_NUM_THREADS:** Not available
- **VECLIB_MAXIMUM_THREADS:** Not available
- **PETSC_OPTIONS:** Not available

## Experiment

- **problem:** Manufactured Poisson problem
- **domain:** (0, 1)^2
- **exact_solution:** sin(pi*x) * sin(pi*y)
- **forcing:** 2*pi^2*sin(pi*x)*sin(pi*y)
- **element:** Continuous Lagrange P1 triangle
- **boundary_condition:** Homogeneous Dirichlet
- **execution_mode:** Serial
- **refinement_levels:** `[2, 3, 4, 5]`
- **operator_random_seeds:** `[2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]`
- **subdomain_bounds:** `[0.25, 0.75, 0.25, 0.75]`
- **gaussian_center:** `[0.7, 0.35]`
- **gaussian_sigma:** 0.1
- **weighted_local_quadrature_order:** 12
- **weighted_reference_quadrature_degree:** 30
- **floating_point_type:** float64
- **operator_equivalence_target:** 1e-11
- **solution_agreement_target:** 1e-09
- **residual_target:** 1e-09

## Solver configurations

- **assembled:** `{"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}`
- **firedrake_matfree:** `{"mat_type": "matfree", "pmat_type": "aij", "ksp_type": "cg", "ksp_rtol": 1e-12, "ksp_atol": 1e-14, "ksp_max_it": 1000, "pc_type": "jacobi"}`
- **local_precomputed:** `{"operator_type": "python", "preconditioner_matrix_type": "aij", "ksp_type": "cg", "ksp_rtol": 1e-12, "ksp_atol": 1e-14, "ksp_max_it": 1000, "pc_type": "jacobi", "precompute_matrices": true}`
- **local_direct:** `{"operator_type": "python", "preconditioner_matrix_type": "aij", "ksp_type": "cg", "ksp_rtol": 1e-12, "ksp_atol": 1e-14, "ksp_max_it": 1000, "pc_type": "jacobi", "precompute_matrices": false}`

## Version control

- **commit:** Not available
- **branch:** Not available
- **remote:** Not available
- **working_tree_clean:** Not available
- **working_tree_status:** Not available

## Environment snapshot

A package snapshot is stored in `results/pip_freeze.txt`.

A pip snapshot may not fully reproduce a Firedrake installation. The original Firedrake environment, container image, or installation procedure should also be retained when available.
