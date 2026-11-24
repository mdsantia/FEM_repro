
"""
Manual tests for exact local functional evaluation.

Run from the project root with:

    python tests/test_functionals.py

or in a notebook with:

    %run tests/test_functionals.py
"""

import numpy as np

from firedrake import Function

from src.problem import (
    create_problem,
)

from src.assembled import (
    DIRECT_SOLVER_PARAMETERS,
    assemble_system,
    solve_assembled,
)

from src.firedrake_matfree import (
    solve_firedrake_matfree,
)

from src.local_operator import (
    solve_local_operator,
)

import importlib
import src.global_functional
importlib.reload(
    src.global_functional
)

from src.global_functional import (
    FunctionalResult,
    compare_global_average,
    create_global_functional_data,
    exact_global_average,
    global_average_local,
    global_average_reference,
    global_integral_local,
    global_integral_reference,
)


# ==================================================
# Stage 1: Functional-data construction
# ==================================================

def test_functional_data_dimensions(
    problem,
    shared,
):
    data = shared["data"]

    assert data.num_dofs == problem.V.dim()

    assert data.dof_connectivity.shape == (
        data.num_cells,
        3,
    )

    assert data.areas.shape == (
        data.num_cells,
    )


def test_functional_data_is_finite(
    problem,
    shared,
):
    data = shared["data"]

    assert np.isfinite(
        data.areas
    ).all()

    assert np.isfinite(
        data.domain_area
    )


def test_functional_domain_area_is_one(
    problem,
    shared,
):
    data = shared["data"]

    assert abs(
        data.domain_area - 1.0
    ) < 1.0e-12


def test_functional_areas_are_positive(
    problem,
    shared,
):
    data = shared["data"]

    assert np.min(
        data.areas
    ) > 0.0


# ==================================================
# Stage 2: Zero and constant functions
# ==================================================

def test_zero_function_has_zero_integral(
    problem,
    shared,
):
    data = shared["data"]

    zero = Function(
        problem.V
    )

    reference = global_integral_reference(
        problem,
        zero,
    )

    local = global_integral_local(
        problem,
        zero,
        data=data,
    )

    assert reference == 0.0
    assert local == 0.0


def test_zero_function_has_zero_average(
    problem,
    shared,
):
    data = shared["data"]

    zero = Function(
        problem.V
    )

    reference = global_average_reference(
        problem,
        zero,
    )

    local = global_average_local(
        problem,
        zero,
        data=data,
    )

    assert reference == 0.0
    assert local == 0.0


def test_constant_one_has_unit_integral(
    problem,
    shared,
):
    data = shared["data"]

    constant_one = Function(
        problem.V
    )

    constant_one.assign(
        1.0
    )

    reference = global_integral_reference(
        problem,
        constant_one,
    )

    local = global_integral_local(
        problem,
        constant_one,
        data=data,
    )

    assert abs(
        reference - 1.0
    ) < 1.0e-12

    assert abs(
        local - 1.0
    ) < 1.0e-12


def test_constant_one_has_unit_average(
    problem,
    shared,
):
    data = shared["data"]

    constant_one = Function(
        problem.V
    )

    constant_one.assign(
        1.0
    )

    reference = global_average_reference(
        problem,
        constant_one,
    )

    local = global_average_local(
        problem,
        constant_one,
        data=data,
    )

    assert abs(
        reference - 1.0
    ) < 1.0e-12

    assert abs(
        local - 1.0
    ) < 1.0e-12


# ==================================================
# Stage 3: Interpolated manufactured solution
# ==================================================

def test_interpolated_solution_local_and_reference_agree(
    problem,
    shared,
):
    data = shared["data"]

    interpolated = Function(
        problem.V
    ).interpolate(
        problem.u_exact
    )

    comparison = compare_global_average(
        problem,
        interpolated,
        data=data,
    )

    assert isinstance(
        comparison,
        FunctionalResult,
    )

    assert (
        comparison.relative_difference
        < 1.0e-13
    )


def test_interpolated_solution_average_is_finite(
    problem,
    shared,
):
    data = shared["data"]

    interpolated = Function(
        problem.V
    ).interpolate(
        problem.u_exact
    )

    value = global_average_local(
        problem,
        interpolated,
        data=data,
    )

    assert np.isfinite(
        value
    )


# ==================================================
# Stage 4: Assembled solution
# ==================================================

def test_assembled_solution_global_average_agrees(
    problem,
    shared,
):
    data = shared["data"]
    result = shared["assembled_result"]

    comparison = compare_global_average(
        problem,
        result.solution,
        data=data,
    )

    assert (
        comparison.relative_difference
        < 1.0e-13
    )


def test_assembled_solution_average_is_reasonable(
    problem,
    shared,
):
    data = shared["data"]
    result = shared["assembled_result"]

    numerical = global_average_local(
        problem,
        result.solution,
        data=data,
    )

    exact = exact_global_average()

    assert abs(
        numerical - exact
    ) < 5.0e-2


# ==================================================
# Stage 5: Comparison across solution methods
# ==================================================

def test_solution_methods_give_same_global_average(
    problem,
    shared,
):
    data = shared["data"]

    assembled_value = global_average_local(
        problem,
        shared[
            "assembled_result"
        ].solution,
        data=data,
    )

    firedrake_matfree_value = (
        global_average_local(
            problem,
            shared[
                "firedrake_matfree_result"
            ].solution,
            data=data,
        )
    )

    local_precomputed_value = (
        global_average_local(
            problem,
            shared[
                "local_precomputed_result"
            ].solution,
            data=data,
        )
    )

    local_direct_value = (
        global_average_local(
            problem,
            shared[
                "local_direct_result"
            ].solution,
            data=data,
        )
    )

    np.testing.assert_allclose(
        firedrake_matfree_value,
        assembled_value,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        local_precomputed_value,
        assembled_value,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        local_direct_value,
        assembled_value,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


# ==================================================
# Stage 6: Global-average convergence
# ==================================================

def test_global_average_error_decreases_under_refinement(
    problem,
    shared,
):
    exact = exact_global_average()

    errors = []

    for level in [
        2,
        3,
        4,
        5,
    ]:
        refined_problem = create_problem(
            refinement_level=level
        )

        refined_result = solve_assembled(
            refined_problem,
            solver_parameters=(
                DIRECT_SOLVER_PARAMETERS
            ),
        )

        value = global_average_local(
            refined_problem,
            refined_result.solution,
        )

        errors.append(
            abs(
                exact - value
            )
        )

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[3] < errors[2]


def test_global_average_observed_rate(
    problem,
    shared,
):
    exact = exact_global_average()

    levels = [
        3,
        4,
        5,
    ]

    errors = []
    mesh_sizes = []

    for level in levels:
        refined_problem = create_problem(
            refinement_level=level
        )

        refined_result = solve_assembled(
            refined_problem,
            solver_parameters=(
                DIRECT_SOLVER_PARAMETERS
            ),
        )

        value = global_average_local(
            refined_problem,
            refined_result.solution,
        )

        errors.append(
            abs(
                exact - value
            )
        )

        mesh_sizes.append(
            refined_problem.h
        )

    rates = []

    for index in range(
        1,
        len(errors),
    ):
        rate = (
            np.log(
                errors[index - 1]
                / errors[index]
            )
            / np.log(
                mesh_sizes[index - 1]
                / mesh_sizes[index]
            )
        )

        rates.append(
            rate
        )

    print(
        "\nObserved global-average rates:"
    )

    for index, rate in enumerate(
        rates,
        start=1,
    ):
        print(
            f"  Level {levels[index - 1]} "
            f"-> {levels[index]}: "
            f"{rate:.6f}"
        )

    # For this smooth linear functional and smooth solution,
    # approximately second-order behavior is expected.
    for rate in rates:
        assert 1.7 < rate < 2.3


# ==================================================
# Manual runner
# ==================================================

def run_all():
    print(
        "\nPhase 2A: Global functional evaluation"
    )

    print(
        "--------------------------------------"
    )

    print(
        "Creating shared level-3 problem..."
    )

    problem = create_problem(
        refinement_level=3
    )

    print(
        "Creating local functional data..."
    )

    data = create_global_functional_data(
        problem
    )

    print(
        "Solving assembled reference problem..."
    )

    assembled_system = assemble_system(
        problem
    )

    assembled_result = solve_assembled(
        problem,
        system=assembled_system,
        solver_parameters=(
            DIRECT_SOLVER_PARAMETERS
        ),
    )

    print(
        "Solving Firedrake matrix-free problem..."
    )

    firedrake_matfree_result = (
        solve_firedrake_matfree(
            problem
        )
    )

    print(
        "Solving precomputed local-operator problem..."
    )

    local_precomputed_result = (
        solve_local_operator(
            problem,
            precompute_matrices=True,
        )
    )

    print(
        "Solving direct local-operator problem..."
    )

    local_direct_result = (
        solve_local_operator(
            problem,
            precompute_matrices=False,
        )
    )

    shared = {
        "data": data,
        "assembled_result": (
            assembled_result
        ),
        "firedrake_matfree_result": (
            firedrake_matfree_result
        ),
        "local_precomputed_result": (
            local_precomputed_result
        ),
        "local_direct_result": (
            local_direct_result
        ),
    }

    stages = [
        (
            "Stage 1: Functional data",
            [
                test_functional_data_dimensions,
                test_functional_data_is_finite,
                test_functional_domain_area_is_one,
                test_functional_areas_are_positive,
            ],
        ),
        (
            "Stage 2: Zero and constant functions",
            [
                test_zero_function_has_zero_integral,
                test_zero_function_has_zero_average,
                test_constant_one_has_unit_integral,
                test_constant_one_has_unit_average,
            ],
        ),
        (
            "Stage 3: Interpolated exact solution",
            [
                test_interpolated_solution_local_and_reference_agree,
                test_interpolated_solution_average_is_finite,
            ],
        ),
        (
            "Stage 4: Assembled solution",
            [
                test_assembled_solution_global_average_agrees,
                test_assembled_solution_average_is_reasonable,
            ],
        ),
        (
            "Stage 5: Solution-method comparison",
            [
                test_solution_methods_give_same_global_average,
            ],
        ),
        (
            "Stage 6: Functional convergence",
            [
                test_global_average_error_decreases_under_refinement,
                test_global_average_observed_rate,
            ],
        ),
    ]

    passed = 0
    failed = []

    for stage_name, tests in stages:
        print(
            f"\n{stage_name}"
        )

        print(
            "-" * len(stage_name)
        )

        for test in tests:
            try:
                test(
                    problem,
                    shared,
                )
            except Exception as exc:
                failed.append(
                    (
                        test.__name__,
                        exc,
                    )
                )

                print(
                    f"FAILED: {test.__name__}"
                )

                print(
                    f"        "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            else:
                passed += 1

                print(
                    f"PASSED: {test.__name__}"
                )

    total = (
        passed + len(failed)
    )

    print(
        "\nPhase 2A summary"
    )

    print(
        "----------------"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    print(
        f"Total:  {total}"
    )

    if failed:
        for test_name, exception in failed:
            print(
                f"  - {test_name}: "
                f"{type(exception).__name__}: "
                f"{exception}"
            )

        raise AssertionError(
            f"{len(failed)} functional tests failed."
        )

    print(
        f"\nAll {total} global-functional tests passed!"
    )


if __name__ == "__main__":
    run_all()
