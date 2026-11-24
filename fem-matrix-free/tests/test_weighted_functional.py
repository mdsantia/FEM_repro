
"""
Manual tests for the smoothly weighted local output.

Run from the project root with:

    python tests/test_weighted_functional.py

or in a notebook with:

    %run tests/test_weighted_functional.py
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

from src.weighted_functional import (
    compare_weighted_average,
    create_weighted_functional_data,
    exact_weighted_average_reference,
    gaussian_weight_values,
    reference_triangle_quadrature,
    weight_integral_reference,
    weighted_average_local,
    weighted_average_reference,
    weighted_integral_local,
    weighted_integral_reference,
)


WEIGHT_CENTER = (
    0.70,
    0.35,
)

WEIGHT_SIGMA = 0.10

LOCAL_QUADRATURE_ORDER = 12

REFERENCE_QUADRATURE_DEGREE = 24

EXACT_REFERENCE_QUADRATURE_DEGREE = 30


# ==================================================
# Stage 1: Reference-triangle quadrature
# ==================================================


def test_reference_quadrature_dimensions(
    problem,
    shared,
):
    points, weights = (
        reference_triangle_quadrature(
            LOCAL_QUADRATURE_ORDER
        )
    )

    expected_points = (
        LOCAL_QUADRATURE_ORDER**2
    )

    assert points.shape == (
        expected_points,
        2,
    )

    assert weights.shape == (
        expected_points,
    )


def test_reference_quadrature_is_finite(
    problem,
    shared,
):
    points, weights = (
        reference_triangle_quadrature(
            LOCAL_QUADRATURE_ORDER
        )
    )

    assert np.isfinite(
        points
    ).all()

    assert np.isfinite(
        weights
    ).all()


def test_reference_quadrature_weights_are_positive(
    problem,
    shared,
):
    _, weights = (
        reference_triangle_quadrature(
            LOCAL_QUADRATURE_ORDER
        )
    )

    assert np.min(
        weights
    ) > 0.0


def test_reference_quadrature_integrates_constant(
    problem,
    shared,
):
    _, weights = (
        reference_triangle_quadrature(
            LOCAL_QUADRATURE_ORDER
        )
    )

    assert abs(
        np.sum(weights) - 0.5
    ) < 1.0e-14


def test_reference_quadrature_integrates_linear_functions(
    problem,
    shared,
):
    points, weights = (
        reference_triangle_quadrature(
            LOCAL_QUADRATURE_ORDER
        )
    )

    xi = points[:, 0]
    eta = points[:, 1]

    integral_xi = np.sum(
        weights * xi
    )

    integral_eta = np.sum(
        weights * eta
    )

    # On the reference triangle:
    #
    # integral xi = integral eta = 1/6.
    np.testing.assert_allclose(
        integral_xi,
        1.0 / 6.0,
        rtol=0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        integral_eta,
        1.0 / 6.0,
        rtol=0.0,
        atol=1.0e-14,
    )


# ==================================================
# Stage 2: Gaussian weight
# ==================================================


def test_gaussian_weight_at_center_is_one(
    problem,
    shared,
):
    value = gaussian_weight_values(
        np.asarray(
            WEIGHT_CENTER
        ),
        center=WEIGHT_CENTER,
        sigma=WEIGHT_SIGMA,
    )

    np.testing.assert_allclose(
        value,
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_gaussian_weight_is_positive(
    problem,
    shared,
):
    points = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ]
    )

    values = gaussian_weight_values(
        points,
        center=WEIGHT_CENTER,
        sigma=WEIGHT_SIGMA,
    )

    assert np.all(
        values > 0.0
    )

    assert np.all(
        values <= 1.0
    )


def test_gaussian_weight_decreases_with_distance(
    problem,
    shared,
):
    center = np.asarray(
        WEIGHT_CENTER
    )

    points = np.vstack(
        (
            center,
            center
            + np.array(
                [0.05, 0.0]
            ),
            center
            + np.array(
                [0.15, 0.0]
            ),
        )
    )

    values = gaussian_weight_values(
        points,
        center=WEIGHT_CENTER,
        sigma=WEIGHT_SIGMA,
    )

    assert values[0] > values[1]
    assert values[1] > values[2]


# ==================================================
# Stage 3: Weighted functional data
# ==================================================


def test_weighted_data_structure(
    problem,
    shared,
):
    data = shared["weighted_data"]

    required_attributes = [
        "center",
        "sigma",
        "quadrature_order",
        "dof_connectivity",
        "local_vectors",
        "element_weight_integrals",
        "weight_integral",
        "num_cells",
        "num_dofs",
    ]

    for attribute in required_attributes:
        assert hasattr(
            data,
            attribute,
        )


def test_weighted_data_dimensions(
    problem,
    shared,
):
    data = shared["weighted_data"]

    assert data.num_dofs == problem.V.dim()

    assert data.dof_connectivity.shape == (
        data.num_cells,
        3,
    )

    assert data.local_vectors.shape == (
        data.num_cells,
        3,
    )

    assert (
        data.element_weight_integrals.shape
        == (data.num_cells,)
    )


def test_weighted_data_is_finite(
    problem,
    shared,
):
    data = shared["weighted_data"]

    assert np.isfinite(
        data.local_vectors
    ).all()

    assert np.isfinite(
        data.element_weight_integrals
    ).all()

    assert np.isfinite(
        data.weight_integral
    )


def test_weight_integrals_are_positive(
    problem,
    shared,
):
    data = shared["weighted_data"]

    assert np.min(
        data.element_weight_integrals
    ) > 0.0

    assert data.weight_integral > 0.0


def test_local_vectors_sum_to_element_weight_integrals(
    problem,
    shared,
):
    data = shared["weighted_data"]

    # Since sum_i phi_i = 1,
    #
    # sum_i integral_T q phi_i
    #     = integral_T q.
    reconstructed = np.sum(
        data.local_vectors,
        axis=1,
    )

    np.testing.assert_allclose(
        reconstructed,
        data.element_weight_integrals,
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def test_local_and_reference_weight_integrals_agree(
    problem,
    shared,
):
    data = shared["weighted_data"]

    reference = weight_integral_reference(
        problem,
        center=WEIGHT_CENTER,
        sigma=WEIGHT_SIGMA,
        quadrature_degree=(
            REFERENCE_QUADRATURE_DEGREE
        ),
    )

    relative_difference = abs(
        data.weight_integral - reference
    ) / abs(reference)

    assert relative_difference < 1.0e-11


# ==================================================
# Stage 4: Zero and constant functions
# ==================================================


def test_zero_function_has_zero_weighted_integral(
    problem,
    shared,
):
    data = shared["weighted_data"]

    zero = Function(
        problem.V
    )

    local = weighted_integral_local(
        problem,
        zero,
        data,
    )

    reference = weighted_integral_reference(
        problem,
        zero,
        center=WEIGHT_CENTER,
        sigma=WEIGHT_SIGMA,
        quadrature_degree=(
            REFERENCE_QUADRATURE_DEGREE
        ),
    )

    assert local == 0.0
    assert reference == 0.0


def test_zero_function_has_zero_weighted_average(
    problem,
    shared,
):
    data = shared["weighted_data"]

    zero = Function(
        problem.V
    )

    local = weighted_average_local(
        problem,
        zero,
        data,
    )

    reference = weighted_average_reference(
        problem,
        zero,
        center=WEIGHT_CENTER,
        sigma=WEIGHT_SIGMA,
        quadrature_degree=(
            REFERENCE_QUADRATURE_DEGREE
        ),
    )

    assert local == 0.0
    assert reference == 0.0


def test_constant_one_weighted_integral_equals_weight_integral(
    problem,
    shared,
):
    data = shared["weighted_data"]

    constant_one = Function(
        problem.V
    )

    constant_one.assign(
        1.0
    )

    local = weighted_integral_local(
        problem,
        constant_one,
        data,
    )

    np.testing.assert_allclose(
        local,
        data.weight_integral,
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def test_constant_one_weighted_average_is_one(
    problem,
    shared,
):
    data = shared["weighted_data"]

    constant_one = Function(
        problem.V
    )

    constant_one.assign(
        1.0
    )

    local = weighted_average_local(
        problem,
        constant_one,
        data,
    )

    reference = weighted_average_reference(
        problem,
        constant_one,
        center=WEIGHT_CENTER,
        sigma=WEIGHT_SIGMA,
        quadrature_degree=(
            REFERENCE_QUADRATURE_DEGREE
        ),
    )

    np.testing.assert_allclose(
        local,
        1.0,
        rtol=0.0,
        atol=1.0e-13,
    )

    np.testing.assert_allclose(
        reference,
        1.0,
        rtol=0.0,
        atol=1.0e-13,
    )


# ==================================================
# Stage 5: Local/reference agreement
# ==================================================


def test_interpolated_weighted_average_agrees(
    problem,
    shared,
):
    data = shared["weighted_data"]

    interpolated = Function(
        problem.V
    ).interpolate(
        problem.u_exact
    )

    comparison = compare_weighted_average(
        problem,
        interpolated,
        data,
        reference_quadrature_degree=(
            REFERENCE_QUADRATURE_DEGREE
        ),
    )

    assert (
        comparison.relative_difference
        < 1.0e-11
    )


def test_assembled_solution_weighted_average_agrees(
    problem,
    shared,
):
    data = shared["weighted_data"]

    comparison = compare_weighted_average(
        problem,
        shared[
            "assembled_result"
        ].solution,
        data,
        reference_quadrature_degree=(
            REFERENCE_QUADRATURE_DEGREE
        ),
    )

    assert (
        comparison.relative_difference
        < 1.0e-11
    )


def test_solution_methods_give_same_weighted_average(
    problem,
    shared,
):
    data = shared["weighted_data"]

    assembled_value = weighted_average_local(
        problem,
        shared[
            "assembled_result"
        ].solution,
        data,
    )

    firedrake_matfree_value = (
        weighted_average_local(
            problem,
            shared[
                "firedrake_matfree_result"
            ].solution,
            data,
        )
    )

    local_precomputed_value = (
        weighted_average_local(
            problem,
            shared[
                "local_precomputed_result"
            ].solution,
            data,
        )
    )

    local_direct_value = (
        weighted_average_local(
            problem,
            shared[
                "local_direct_result"
            ].solution,
            data,
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
# Stage 6: Quadrature convergence
# ==================================================


def test_weighted_quadrature_order_converges(
    problem,
    shared,
):
    reference = weight_integral_reference(
        problem,
        center=WEIGHT_CENTER,
        sigma=WEIGHT_SIGMA,
        quadrature_degree=30,
    )

    orders = [
        4,
        6,
        8,
        10,
        12,
    ]

    errors = []

    for order in orders:
        data = create_weighted_functional_data(
            problem,
            center=WEIGHT_CENTER,
            sigma=WEIGHT_SIGMA,
            quadrature_order=order,
        )

        errors.append(
            abs(
                data.weight_integral
                - reference
            )
        )

    print(
        "\nGaussian weight-integral quadrature errors:"
    )

    for order, error in zip(
        orders,
        errors,
    ):
        print(
            f"  Order {order:2d}: "
            f"{error:.6e}"
        )

    # High-order quadrature should be more accurate
    # than the low-order starting rule.
    assert errors[-1] < errors[0]

    assert (
        errors[-1]
        / abs(reference)
        < 1.0e-11
    )


# ==================================================
# Stage 7: Weighted-output FEM convergence
# ==================================================


def test_weighted_output_error_decreases(
    problem,
    shared,
):
    levels = [
        2,
        3,
        4,
        5,
    ]

    errors = []

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
            create_weighted_functional_data(
                refined_problem,
                center=WEIGHT_CENTER,
                sigma=WEIGHT_SIGMA,
                quadrature_order=(
                    LOCAL_QUADRATURE_ORDER
                ),
            )
        )

        exact_value = (
            exact_weighted_average_reference(
                refined_problem,
                center=WEIGHT_CENTER,
                sigma=WEIGHT_SIGMA,
                quadrature_degree=(
                    EXACT_REFERENCE_QUADRATURE_DEGREE
                ),
            )
        )

        numerical_value = weighted_average_local(
            refined_problem,
            refined_result.solution,
            refined_data,
        )

        errors.append(
            abs(
                exact_value
                - numerical_value
            )
        )

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[3] < errors[2]


def test_weighted_output_observed_rate(
    problem,
    shared,
):
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
            create_weighted_functional_data(
                refined_problem,
                center=WEIGHT_CENTER,
                sigma=WEIGHT_SIGMA,
                quadrature_order=(
                    LOCAL_QUADRATURE_ORDER
                ),
            )
        )

        exact_value = (
            exact_weighted_average_reference(
                refined_problem,
                center=WEIGHT_CENTER,
                sigma=WEIGHT_SIGMA,
                quadrature_degree=(
                    EXACT_REFERENCE_QUADRATURE_DEGREE
                ),
            )
        )

        numerical_value = weighted_average_local(
            refined_problem,
            refined_result.solution,
            refined_data,
        )

        errors.append(
            abs(
                exact_value
                - numerical_value
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
        "\nObserved weighted-output rates:"
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
# Stage 8: Parameter validation
# ==================================================


def test_weighted_data_rejects_invalid_center(
    problem,
    shared,
):
    try:
        create_weighted_functional_data(
            problem,
            center=(0.5,),
            sigma=WEIGHT_SIGMA,
        )
    except ValueError:
        return

    raise AssertionError(
        "An invalid Gaussian center was accepted."
    )


def test_weighted_data_rejects_invalid_sigma(
    problem,
    shared,
):
    try:
        create_weighted_functional_data(
            problem,
            center=WEIGHT_CENTER,
            sigma=0.0,
        )
    except ValueError:
        return

    raise AssertionError(
        "A nonpositive Gaussian sigma was accepted."
    )


def test_weighted_data_rejects_invalid_quadrature_order(
    problem,
    shared,
):
    try:
        create_weighted_functional_data(
            problem,
            center=WEIGHT_CENTER,
            sigma=WEIGHT_SIGMA,
            quadrature_order=0,
        )
    except ValueError:
        return

    raise AssertionError(
        "A nonpositive quadrature order was accepted."
    )


# ==================================================
# Manual runner
# ==================================================


def run_all():
    print(
        "\nPhase 2C: Smoothly weighted local output"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Creating shared level-3 problem..."
    )

    problem = create_problem(
        refinement_level=3
    )

    print(
        "Creating weighted functional data..."
    )

    weighted_data = (
        create_weighted_functional_data(
            problem,
            center=WEIGHT_CENTER,
            sigma=WEIGHT_SIGMA,
            quadrature_order=(
                LOCAL_QUADRATURE_ORDER
            ),
        )
    )

    print(
        "Assembling and solving reference problem..."
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
        "weighted_data": weighted_data,
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
            "Stage 1: Reference-triangle quadrature",
            [
                test_reference_quadrature_dimensions,
                test_reference_quadrature_is_finite,
                test_reference_quadrature_weights_are_positive,
                test_reference_quadrature_integrates_constant,
                test_reference_quadrature_integrates_linear_functions,
            ],
        ),
        (
            "Stage 2: Gaussian weight",
            [
                test_gaussian_weight_at_center_is_one,
                test_gaussian_weight_is_positive,
                test_gaussian_weight_decreases_with_distance,
            ],
        ),
        (
            "Stage 3: Weighted functional data",
            [
                test_weighted_data_structure,
                test_weighted_data_dimensions,
                test_weighted_data_is_finite,
                test_weight_integrals_are_positive,
                test_local_vectors_sum_to_element_weight_integrals,
                test_local_and_reference_weight_integrals_agree,
            ],
        ),
        (
            "Stage 4: Zero and constant functions",
            [
                test_zero_function_has_zero_weighted_integral,
                test_zero_function_has_zero_weighted_average,
                test_constant_one_weighted_integral_equals_weight_integral,
                test_constant_one_weighted_average_is_one,
            ],
        ),
        (
            "Stage 5: Local/reference agreement",
            [
                test_interpolated_weighted_average_agrees,
                test_assembled_solution_weighted_average_agrees,
                test_solution_methods_give_same_weighted_average,
            ],
        ),
        (
            "Stage 6: Quadrature convergence",
            [
                test_weighted_quadrature_order_converges,
            ],
        ),
        (
            "Stage 7: Weighted-output FEM convergence",
            [
                test_weighted_output_error_decreases,
                test_weighted_output_observed_rate,
            ],
        ),
        (
            "Stage 8: Parameter validation",
            [
                test_weighted_data_rejects_invalid_center,
                test_weighted_data_rejects_invalid_sigma,
                test_weighted_data_rejects_invalid_quadrature_order,
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
        "\nPhase 2C summary"
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
        print(
            "\nFailed tests:"
        )

        for test_name, exception in failed:
            print(
                f"  - {test_name}: "
                f"{type(exception).__name__}: "
                f"{exception}"
            )

        raise AssertionError(
            f"{len(failed)} weighted functional tests failed."
        )

    exact_value = exact_weighted_average_reference(
        problem,
        center=WEIGHT_CENTER,
        sigma=WEIGHT_SIGMA,
        quadrature_degree=(
            EXACT_REFERENCE_QUADRATURE_DEGREE
        ),
    )

    numerical_value = weighted_average_local(
        problem,
        assembled_result.solution,
        weighted_data,
    )

    print(
        "\nWeighted-output values"
    )

    print(
        "----------------------"
    )

    print(
        f"Weight center: {WEIGHT_CENTER}"
    )

    print(
        f"Weight sigma:  {WEIGHT_SIGMA}"
    )

    print(
        f"Exact-solution reference: "
        f"{exact_value:.12e}"
    )

    print(
        f"Level-3 FEM output:       "
        f"{numerical_value:.12e}"
    )

    print(
        f"Absolute functional error: "
        f"{abs(exact_value - numerical_value):.12e}"
    )

    print(
        f"\nAll {total} weighted-functional tests passed!"
    )


if __name__ == "__main__":
    run_all()
