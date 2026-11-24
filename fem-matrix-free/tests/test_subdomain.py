
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
import src.subdomain_functional
importlib.reload(
    src.subdomain_functional
)
from src.subdomain_functional import (
    SubdomainFunctionalData,
    compare_subdomain_average,
    create_subdomain_functional_data,
    exact_subdomain_average,
    subdomain_average_local,
    subdomain_average_reference,
    subdomain_integral_local,
    subdomain_integral_reference,
)

SUBDOMAIN_BOUNDS = (
    0.25,
    0.75,
    0.25,
    0.75,
)

# ==================================================
# Stage 7: Subdomain functional data
# ==================================================


def test_subdomain_data_type(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    assert isinstance(
        data,
        SubdomainFunctionalData,
    )


def test_subdomain_bounds_are_preserved(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    assert data.bounds == (
        SUBDOMAIN_BOUNDS
    )


def test_subdomain_area_is_correct(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    assert abs(
        data.subdomain_area - 0.25
    ) < 1.0e-12


def test_subdomain_has_expected_number_of_cells(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    # Level 3 has an 8 x 8 square grid.
    #
    # The rectangle [0.25, 0.75]^2 contains a
    # 4 x 4 block of squares, each split into
    # two triangles.
    expected_cells = (
        4
        * 4
        * 2
    )

    assert (
        data.num_selected_cells
        == expected_cells
    )


def test_subdomain_data_dimensions(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    assert data.selected_connectivity.shape == (
        data.num_selected_cells,
        3,
    )

    assert data.selected_areas.shape == (
        data.num_selected_cells,
    )

    assert data.num_dofs == problem.V.dim()


def test_subdomain_areas_are_positive_and_finite(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    assert np.isfinite(
        data.selected_areas
    ).all()

    assert np.min(
        data.selected_areas
    ) > 0.0


# ==================================================
# Stage 8: Zero and constant subdomain values
# ==================================================


def test_zero_function_has_zero_subdomain_integral(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    zero = Function(
        problem.V
    )

    reference = subdomain_integral_reference(
        problem,
        zero,
        SUBDOMAIN_BOUNDS,
    )

    local = subdomain_integral_local(
        problem,
        zero,
        data=data,
    )

    assert reference == 0.0
    assert local == 0.0


def test_zero_function_has_zero_subdomain_average(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    zero = Function(
        problem.V
    )

    reference = subdomain_average_reference(
        problem,
        zero,
        SUBDOMAIN_BOUNDS,
    )

    local = subdomain_average_local(
        problem,
        zero,
        data=data,
    )

    assert reference == 0.0
    assert local == 0.0


def test_constant_one_subdomain_integral_equals_area(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    constant_one = Function(
        problem.V
    )

    constant_one.assign(
        1.0
    )

    reference = subdomain_integral_reference(
        problem,
        constant_one,
        SUBDOMAIN_BOUNDS,
    )

    local = subdomain_integral_local(
        problem,
        constant_one,
        data=data,
    )

    np.testing.assert_allclose(
        reference,
        data.subdomain_area,
        rtol=0.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        local,
        data.subdomain_area,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_constant_one_subdomain_average_is_one(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    constant_one = Function(
        problem.V
    )

    constant_one.assign(
        1.0
    )

    reference = subdomain_average_reference(
        problem,
        constant_one,
        SUBDOMAIN_BOUNDS,
    )

    local = subdomain_average_local(
        problem,
        constant_one,
        data=data,
    )

    np.testing.assert_allclose(
        reference,
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        local,
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )


# ==================================================
# Stage 9: Local/reference subdomain agreement
# ==================================================


def test_interpolated_subdomain_average_agrees(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    interpolated = Function(
        problem.V
    ).interpolate(
        problem.u_exact
    )

    comparison = compare_subdomain_average(
        problem,
        interpolated,
        data=data,
    )

    assert (
        comparison.relative_difference
        < 1.0e-13
    )


def test_assembled_solution_subdomain_average_agrees(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    comparison = compare_subdomain_average(
        problem,
        shared[
            "assembled_result"
        ].solution,
        data=data,
    )

    assert (
        comparison.relative_difference
        < 1.0e-13
    )


def test_subdomain_solution_methods_agree(
    problem,
    shared,
):
    data = shared["subdomain_data"]

    assembled_value = (
        subdomain_average_local(
            problem,
            shared[
                "assembled_result"
            ].solution,
            data=data,
        )
    )

    firedrake_matfree_value = (
        subdomain_average_local(
            problem,
            shared[
                "firedrake_matfree_result"
            ].solution,
            data=data,
        )
    )

    local_precomputed_value = (
        subdomain_average_local(
            problem,
            shared[
                "local_precomputed_result"
            ].solution,
            data=data,
        )
    )

    local_direct_value = (
        subdomain_average_local(
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
# Stage 10: Subdomain functional accuracy
# ==================================================


def test_exact_subdomain_average_is_finite(
    problem,
    shared,
):
    value = exact_subdomain_average(
        SUBDOMAIN_BOUNDS
    )

    assert np.isfinite(
        value
    )

    assert value > 0.0


def test_subdomain_average_error_decreases(
    problem,
    shared,
):
    exact = exact_subdomain_average(
        SUBDOMAIN_BOUNDS
    )

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

        refined_data = (
            create_subdomain_functional_data(
                refined_problem,
                SUBDOMAIN_BOUNDS,
            )
        )

        value = subdomain_average_local(
            refined_problem,
            refined_result.solution,
            data=refined_data,
        )

        errors.append(
            abs(
                exact - value
            )
        )

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[3] < errors[2]


def test_subdomain_average_observed_rate(
    problem,
    shared,
):
    exact = exact_subdomain_average(
        SUBDOMAIN_BOUNDS
    )

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

        refined_data = (
            create_subdomain_functional_data(
                refined_problem,
                SUBDOMAIN_BOUNDS,
            )
        )

        value = subdomain_average_local(
            refined_problem,
            refined_result.solution,
            data=refined_data,
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
        "\nObserved subdomain-average rates:"
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

    for rate in rates:
        assert 1.7 < rate < 2.3


# ==================================================
# Stage 11: Subdomain validation failures
# ==================================================


def test_subdomain_rejects_invalid_bounds(
    problem,
    shared,
):
    try:
        create_subdomain_functional_data(
            problem,
            (
                0.75,
                0.25,
                0.25,
                0.75,
            ),
        )
    except ValueError:
        return

    raise AssertionError(
        "Invalid rectangle bounds were accepted."
    )


def test_subdomain_rejects_nonaligned_rectangle(
    problem,
    shared,
):
    try:
        create_subdomain_functional_data(
            problem,
            (
                0.30,
                0.70,
                0.25,
                0.75,
            ),
        )
    except ValueError:
        return

    raise AssertionError(
        "A non-mesh-aligned rectangle was accepted."
    )


def test_subdomain_requires_bounds_or_data(
    problem,
    shared,
):
    zero = Function(
        problem.V
    )

    try:
        subdomain_average_local(
            problem,
            zero,
        )
    except ValueError:
        return

    raise AssertionError(
        "Missing subdomain data was accepted."
    )


def run_all():
    print(
        "\nPhase 2B: Sub Domain functional evaluation"
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
        "Creating mesh-aligned subdomain data..."
    )

    subdomain_data = (
        create_subdomain_functional_data(
            problem,
            SUBDOMAIN_BOUNDS,
        )
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
        "subdomain_data": subdomain_data,
        "assembled_result": assembled_result,
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
            "Stage 7: Subdomain functional data",
            [
                test_subdomain_data_type,
                test_subdomain_bounds_are_preserved,
                test_subdomain_area_is_correct,
                test_subdomain_has_expected_number_of_cells,
                test_subdomain_data_dimensions,
                test_subdomain_areas_are_positive_and_finite,
            ],
        ),
        (
            "Stage 8: Zero and constant subdomain values",
            [
                test_zero_function_has_zero_subdomain_integral,
                test_zero_function_has_zero_subdomain_average,
                test_constant_one_subdomain_integral_equals_area,
                test_constant_one_subdomain_average_is_one,
            ],
        ),
        (
            "Stage 9: Local/reference subdomain agreement",
            [
                test_interpolated_subdomain_average_agrees,
                test_assembled_solution_subdomain_average_agrees,
                test_subdomain_solution_methods_agree,
            ],
        ),
        (
            "Stage 10: Subdomain functional accuracy",
            [
                test_exact_subdomain_average_is_finite,
                test_subdomain_average_error_decreases,
                test_subdomain_average_observed_rate,
            ],
        ),
        (
            "Stage 11: Subdomain validation",
            [
                test_subdomain_rejects_invalid_bounds,
                test_subdomain_rejects_nonaligned_rectangle,
                test_subdomain_requires_bounds_or_data,
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
        "\nPhase 2B summary"
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
