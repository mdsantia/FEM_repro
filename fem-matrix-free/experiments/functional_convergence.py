
"""
Phase 2D: Combined output-functional convergence experiment.

The experiment evaluates:

    1. Global average.
    2. Mesh-aligned subdomain average.
    3. Smoothly weighted average.

The outputs are evaluated for solutions obtained with:

    1. Assembled AIJ operator.
    2. Firedrake matrix-free operator.
    3. Independent local operator with precomputed matrices.
    4. Independent local operator with direct element action.

The experiment distinguishes:

    - local-evaluation discrepancy;
    - finite-element functional error;
    - solution-method discrepancy.

Run from the project root with:

    python experiments/functional_convergence.py
"""

from pathlib import Path
import csv
import math
import time

import numpy as np

from src.problem import (
    create_problem,
    l2_error,
    h1_seminorm_error,
)

from src.assembled import (
    DIRECT_SOLVER_PARAMETERS,
    assemble_system,
    compute_relative_residual,
    solve_assembled,
)

from src.firedrake_matfree import (
    solve_firedrake_matfree,
)

from src.local_operator import (
    solve_local_operator,
)

from src.global_functional import (
    compare_global_average,
    create_global_functional_data,
    exact_global_average,
    global_average_local,
)

from src.subdomain_functional import (
    compare_subdomain_average,
    subdomain_average_local,
    exact_subdomain_average,
    create_subdomain_functional_data,
    # exact_subdomain_average,
)

from src.weighted_functional import (
    compare_weighted_average,
    weighted_average_local,
    exact_weighted_average_reference,
    create_weighted_functional_data,
    # exact_weighted_average_reference,
)

DEFAULT_LEVELS = [
    2,
    3,
    4,
    5,
]


SUBDOMAIN_BOUNDS = (
    0.25,
    0.75,
    0.25,
    0.75,
)


WEIGHT_CENTER = (
    0.70,
    0.35,
)


WEIGHT_SIGMA = 0.10


LOCAL_WEIGHT_QUADRATURE_ORDER = 12


REFERENCE_WEIGHT_QUADRATURE_DEGREE = 30


METHODS = (
    "assembled",
    "firedrake_matfree",
    "local_precomputed",
    "local_direct",
)


FUNCTIONALS = (
    "global_average",
    "subdomain_average",
    "weighted_average",
)


def observed_rate(
    coarse_error,
    fine_error,
    coarse_h,
    fine_h,
):
    """
    Compute an experimental convergence rate.

    Return NaN if either error is nonpositive or nonfinite.
    """

    values = np.asarray(
        [
            coarse_error,
            fine_error,
            coarse_h,
            fine_h,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(
        values
    ).all():
        return float("nan")

    if np.any(
        values <= 0.0
    ):
        return float("nan")

    return float(
        math.log(
            coarse_error
            / fine_error
        )
        / math.log(
            coarse_h
            / fine_h
        )
    )


def relative_scalar_difference(
    reference,
    candidate,
):
    """
    Compute a relative scalar difference.
    """

    reference = float(
        reference
    )

    candidate = float(
        candidate
    )

    absolute_difference = abs(
        candidate - reference
    )

    if reference == 0.0:
        return absolute_difference

    return float(
        absolute_difference
        / abs(reference)
    )


def solve_all_methods(
    problem,
):
    """
    Solve the same discrete problem with all four implementations.
    """

    results = {}

    print(
        "    Solving assembled system..."
    )

    start = time.perf_counter()

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

    assembled_elapsed = (
        time.perf_counter()
        - start
    )

    results["assembled"] = {
        "result": assembled_result,
        "elapsed_seconds": (
            assembled_elapsed
        ),
        "iterations": (
            assembled_result.iterations
        ),
        "ksp_residual_norm": (
            assembled_result.residual_norm
        ),
        "assembled_relative_residual": (
            assembled_result.relative_residual
        ),
    }

    print(
        "    Solving Firedrake matrix-free system..."
    )

    start = time.perf_counter()

    firedrake_matfree_result = (
        solve_firedrake_matfree(
            problem
        )
    )

    firedrake_matfree_elapsed = (
        time.perf_counter()
        - start
    )

    results["firedrake_matfree"] = {
        "result": firedrake_matfree_result,
        "elapsed_seconds": (
            firedrake_matfree_elapsed
        ),
        "iterations": (
            firedrake_matfree_result.iterations
        ),
        "ksp_residual_norm": getattr(
            firedrake_matfree_result,
            "ksp_residual_norm",
            np.nan,
        ),
        "assembled_relative_residual": (
            compute_relative_residual(
                assembled_system,
                firedrake_matfree_result.solution,
            )
        ),
    }

    print(
        "    Solving precomputed local system..."
    )

    start = time.perf_counter()

    local_precomputed_result = (
        solve_local_operator(
            problem,
            precompute_matrices=True,
        )
    )

    local_precomputed_elapsed = (
        time.perf_counter()
        - start
    )

    results["local_precomputed"] = {
        "result": (
            local_precomputed_result
        ),
        "elapsed_seconds": (
            local_precomputed_elapsed
        ),
        "iterations": (
            local_precomputed_result.iterations
        ),
        "ksp_residual_norm": (
            local_precomputed_result
            .ksp_residual_norm
        ),
        "assembled_relative_residual": (
            compute_relative_residual(
                assembled_system,
                local_precomputed_result.solution,
            )
        ),
    }

    print(
        "    Solving direct local system..."
    )

    start = time.perf_counter()

    local_direct_result = (
        solve_local_operator(
            problem,
            precompute_matrices=False,
        )
    )

    local_direct_elapsed = (
        time.perf_counter()
        - start
    )

    results["local_direct"] = {
        "result": local_direct_result,
        "elapsed_seconds": (
            local_direct_elapsed
        ),
        "iterations": (
            local_direct_result.iterations
        ),
        "ksp_residual_norm": (
            local_direct_result
            .ksp_residual_norm
        ),
        "assembled_relative_residual": (
            compute_relative_residual(
                assembled_system,
                local_direct_result.solution,
            )
        ),
    }

    return (
        assembled_system,
        results,
    )


def create_functional_data(
    problem,
):
    """
    Construct all local functional data for one mesh.
    """

    global_data = (
        create_global_functional_data(
            problem
        )
    )

    subdomain_data = (
        create_subdomain_functional_data(
            problem,
            SUBDOMAIN_BOUNDS,
        )
    )

    weighted_data = (
        create_weighted_functional_data(
            problem,
            center=WEIGHT_CENTER,
            sigma=WEIGHT_SIGMA,
            quadrature_order=(
                LOCAL_WEIGHT_QUADRATURE_ORDER
            ),
        )
    )

    return {
        "global_average": global_data,
        "subdomain_average": (
            subdomain_data
        ),
        "weighted_average": (
            weighted_data
        ),
    }


def create_exact_values(
    problem,
):
    """
    Construct exact or high-order-reference functional values.
    """

    return {
        "global_average": (
            exact_global_average()
        ),
        "subdomain_average": (
            exact_subdomain_average(
                SUBDOMAIN_BOUNDS
            )
        ),
        "weighted_average": (
            exact_weighted_average_reference(
                problem,
                center=WEIGHT_CENTER,
                sigma=WEIGHT_SIGMA,
                quadrature_degree=(
                    REFERENCE_WEIGHT_QUADRATURE_DEGREE
                ),
            )
        ),
    }


def evaluate_functional(
    functional_name,
    problem,
    solution,
    functional_data,
):
    """
    Evaluate one functional by its local formula.
    """

    if (
        functional_name
        == "global_average"
    ):
        return global_average_local(
            problem,
            solution,
            data=functional_data,
        )

    if (
        functional_name
        == "subdomain_average"
    ):
        return subdomain_average_local(
            problem,
            solution,
            data=functional_data,
        )

    if (
        functional_name
        == "weighted_average"
    ):
        return weighted_average_local(
            problem,
            solution,
            functional_data,
        )

    raise ValueError(
        f"Unknown functional: "
        f"{functional_name}"
    )


def evaluate_reference_discrepancy(
    functional_name,
    problem,
    solution,
    functional_data,
):
    """
    Compare the local functional formula with its Firedrake reference.
    """

    if (
        functional_name
        == "global_average"
    ):
        comparison = (
            compare_global_average(
                problem,
                solution,
                data=functional_data,
            )
        )

    elif (
        functional_name
        == "subdomain_average"
    ):
        comparison = (
            compare_subdomain_average(
                problem,
                solution,
                data=functional_data,
            )
        )

    elif (
        functional_name
        == "weighted_average"
    ):
        comparison = (
            compare_weighted_average(
                problem,
                solution,
                functional_data,
                reference_quadrature_degree=(
                    REFERENCE_WEIGHT_QUADRATURE_DEGREE
                ),
            )
        )

    else:
        raise ValueError(
            f"Unknown functional: "
            f"{functional_name}"
        )

    return (
        comparison.absolute_difference,
        comparison.relative_difference,
    )


def run_level(
    level,
):
    """
    Run all solution and functional evaluations at one mesh level.
    """

    print(
        f"\nRefinement level {level}"
    )

    print(
        "-" * 40
    )

    problem = create_problem(
        refinement_level=level
    )

    (
        assembled_system,
        method_results,
    ) = solve_all_methods(
        problem
    )

    print(
        "    Constructing local functional data..."
    )

    functional_data = (
        create_functional_data(
            problem
        )
    )

    exact_values = create_exact_values(
        problem
    )

    assembled_solution = (
        method_results[
            "assembled"
        ][
            "result"
        ].solution
    )

    rows = []

    method_solution_values = {
        method: (
            method_results[
                method
            ][
                "result"
            ].solution.dat.data_ro.copy()
        )
        for method in METHODS
    }

    assembled_values = (
        method_solution_values[
            "assembled"
        ]
    )

    for method in METHODS:

        result = method_results[
            method
        ][
            "result"
        ]

        solution = result.solution

        l2_value = l2_error(
            problem,
            solution,
        )

        h1_value = h1_seminorm_error(
            problem,
            solution,
        )

        solution_relative_difference = (
            relative_scalar_or_vector_difference(
                assembled_values,
                method_solution_values[
                    method
                ],
            )
        )

        for functional_name in FUNCTIONALS:

            data = functional_data[
                functional_name
            ]

            local_value = evaluate_functional(
                functional_name,
                problem,
                solution,
                data,
            )

            exact_value = exact_values[
                functional_name
            ]

            functional_error = abs(
                exact_value
                - local_value
            )

            (
                evaluation_absolute_difference,
                evaluation_relative_difference,
            ) = evaluate_reference_discrepancy(
                functional_name,
                problem,
                solution,
                data,
            )

            assembled_functional_value = (
                evaluate_functional(
                    functional_name,
                    problem,
                    assembled_solution,
                    data,
                )
            )

            method_functional_difference = (
                relative_scalar_difference(
                    assembled_functional_value,
                    local_value,
                )
            )

            rows.append(
                {
                    "level": level,
                    "h": problem.h,
                    "ndofs": (
                        problem.V.dim()
                    ),
                    "num_cells": (
                        data.num_cells
                    ),
                    "method": method,
                    "functional": (
                        functional_name
                    ),
                    "functional_value": (
                        local_value
                    ),
                    "exact_value": (
                        exact_value
                    ),
                    "functional_error": (
                        functional_error
                    ),
                    "functional_rate": (
                        np.nan
                    ),
                    "evaluation_absolute_difference": (
                        evaluation_absolute_difference
                    ),
                    "evaluation_relative_difference": (
                        evaluation_relative_difference
                    ),
                    "method_functional_relative_difference": (
                        method_functional_difference
                    ),
                    "solution_relative_difference": (
                        solution_relative_difference
                    ),
                    "l2_error": (
                        l2_value
                    ),
                    "h1_seminorm_error": (
                        h1_value
                    ),
                    "iterations": (
                        method_results[
                            method
                        ][
                            "iterations"
                        ]
                    ),
                    "ksp_residual_norm": (
                        method_results[
                            method
                        ][
                            "ksp_residual_norm"
                        ]
                    ),
                    "assembled_relative_residual": (
                        method_results[
                            method
                        ][
                            "assembled_relative_residual"
                        ]
                    ),
                    "solve_elapsed_seconds": (
                        method_results[
                            method
                        ][
                            "elapsed_seconds"
                        ]
                    ),
                }
            )

    return rows


def relative_scalar_or_vector_difference(
    reference,
    candidate,
):
    """
    Compute relative Euclidean difference between solution vectors.
    """

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    candidate = np.asarray(
        candidate,
        dtype=np.float64,
    )

    difference = np.linalg.norm(
        candidate - reference
    )

    denominator = np.linalg.norm(
        reference
    )

    if denominator == 0.0:
        return float(
            difference
        )

    return float(
        difference / denominator
    )


def add_functional_rates(
    rows,
):
    """
    Add successive-mesh functional convergence rates.

    Rates are computed independently for each method and functional.
    """

    groups = {}

    for row in rows:

        key = (
            row["method"],
            row["functional"],
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row
        )

    for group_rows in groups.values():

        group_rows.sort(
            key=lambda row: row[
                "level"
            ]
        )

        for index in range(
            1,
            len(group_rows),
        ):

            coarse = group_rows[
                index - 1
            ]

            fine = group_rows[
                index
            ]

            fine[
                "functional_rate"
            ] = observed_rate(
                coarse[
                    "functional_error"
                ],
                fine[
                    "functional_error"
                ],
                coarse["h"],
                fine["h"],
            )

    return rows


def write_csv(
    rows,
    output_path,
):
    """
    Write the detailed experiment rows to CSV.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "level",
        "h",
        "ndofs",
        "num_cells",
        "method",
        "functional",
        "functional_value",
        "exact_value",
        "functional_error",
        "functional_rate",
        "evaluation_absolute_difference",
        "evaluation_relative_difference",
        "method_functional_relative_difference",
        "solution_relative_difference",
        "l2_error",
        "h1_seminorm_error",
        "iterations",
        "ksp_residual_norm",
        "assembled_relative_residual",
        "solve_elapsed_seconds",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"\nSaved detailed results to: "
        f"{output_path}"
    )


def print_assembled_summary(
    rows,
):
    """
    Print the principal convergence table for the assembled solution.

    Other methods should have nearly identical functional values and
    are summarized separately by their discrepancies.
    """

    assembled_rows = [
        row
        for row in rows
        if row["method"]
        == "assembled"
    ]

    print(
        "\nFunctional convergence summary"
    )

    print(
        "=" * 112
    )

    header = (
        f"{'Level':>5} "
        f"{'DoFs':>8} "
        f"{'Functional':>20} "
        f"{'Value':>15} "
        f"{'Exact':>15} "
        f"{'Error':>13} "
        f"{'Rate':>8} "
        f"{'Local/ref diff':>15}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    assembled_rows.sort(
        key=lambda row: (
            row["level"],
            row["functional"],
        )
    )

    for row in assembled_rows:

        rate = row[
            "functional_rate"
        ]

        if np.isfinite(
            rate
        ):
            rate_text = (
                f"{rate:.4f}"
            )
        else:
            rate_text = "-"

        print(
            f"{row['level']:5d} "
            f"{row['ndofs']:8d} "
            f"{row['functional']:>20} "
            f"{row['functional_value']:15.7e} "
            f"{row['exact_value']:15.7e} "
            f"{row['functional_error']:13.5e} "
            f"{rate_text:>8} "
            f"{row['evaluation_relative_difference']:15.5e}"
        )

    print(
        "=" * 112
    )


def print_method_agreement_summary(
    rows,
):
    """
    Print maximum differences from the assembled solution.
    """

    print(
        "\nSolution-method agreement summary"
    )

    print(
        "=" * 95
    )

    header = (
        f"{'Method':>22} "
        f"{'Functional':>20} "
        f"{'max output diff':>18} "
        f"{'max solution diff':>18} "
        f"{'max residual':>14}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for method in METHODS:

        for functional_name in FUNCTIONALS:

            selected = [
                row
                for row in rows
                if (
                    row["method"] == method
                    and row["functional"]
                    == functional_name
                )
            ]

            max_output_difference = max(
                row[
                    "method_functional_relative_difference"
                ]
                for row in selected
            )

            max_solution_difference = max(
                row[
                    "solution_relative_difference"
                ]
                for row in selected
            )

            max_residual = max(
                row[
                    "assembled_relative_residual"
                ]
                for row in selected
            )

            print(
                f"{method:>22} "
                f"{functional_name:>20} "
                f"{max_output_difference:18.6e} "
                f"{max_solution_difference:18.6e} "
                f"{max_residual:14.6e}"
            )

    print(
        "=" * 95
    )


def check_phase_status(
    rows,
    evaluation_target=1.0e-10,
    solution_target=1.0e-9,
    residual_target=1.0e-9,
):
    """
    Check the numerical acceptance criteria for Phase 2D.
    """

    evaluation_failures = [
        row
        for row in rows
        if row[
            "evaluation_relative_difference"
        ] > evaluation_target
    ]

    solution_failures = [
        row
        for row in rows
        if row[
            "solution_relative_difference"
        ] > solution_target
    ]

    residual_failures = [
        row
        for row in rows
        if row[
            "assembled_relative_residual"
        ] > residual_target
    ]

    finite_failures = [
        row
        for row in rows
        if not np.isfinite(
            [
                row[
                    "functional_value"
                ],
                row[
                    "exact_value"
                ],
                row[
                    "functional_error"
                ],
                row[
                    "evaluation_relative_difference"
                ],
            ]
        ).all()
    ]

    passed = not (
        evaluation_failures
        or solution_failures
        or residual_failures
        or finite_failures
    )

    print(
        "\nPhase 2D status"
    )

    print(
        "----------------"
    )

    print(
        f"Functional-evaluation failures: "
        f"{len(evaluation_failures)}"
    )

    print(
        f"Solution-agreement failures:     "
        f"{len(solution_failures)}"
    )

    print(
        f"Residual failures:               "
        f"{len(residual_failures)}"
    )

    print(
        f"Nonfinite rows:                  "
        f"{len(finite_failures)}"
    )

    if passed:
        print(
            "\nPHASE 2D STATUS: PASS"
        )
    else:
        print(
            "\nPHASE 2D STATUS: REVIEW REQUIRED"
        )

    return passed


def run_experiment(
    levels=None,
    output_path=(
        "results/functional_convergence.csv"
    ),
):
    """
    Run the complete Phase 2D functional experiment.
    """

    if levels is None:
        levels = DEFAULT_LEVELS

    print(
        "Phase 2D: Combined functional-convergence experiment"
    )

    print(
        "----------------------------------------------------"
    )

    print(
        f"Levels: {levels}"
    )

    print(
        f"Subdomain: {SUBDOMAIN_BOUNDS}"
    )

    print(
        f"Weight center: {WEIGHT_CENTER}"
    )

    print(
        f"Weight sigma: {WEIGHT_SIGMA}"
    )

    rows = []

    for level in levels:

        level_rows = run_level(
            level
        )

        rows.extend(
            level_rows
        )

    rows = add_functional_rates(
        rows
    )

    print_assembled_summary(
        rows
    )

    print_method_agreement_summary(
        rows
    )

    write_csv(
        rows,
        output_path,
    )

    passed = check_phase_status(
        rows
    )

    return rows, passed


if __name__ == "__main__":
    run_experiment()
