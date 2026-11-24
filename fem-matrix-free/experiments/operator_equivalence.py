
"""
Operator-equivalence experiment.

For each refinement level and seeded input vector, compare:

    1. Firedrake assembled AIJ action.
    2. Independent local action with precomputed element matrices.
    3. Independent direct gradient-factorized element action.

The experiment reports both:

    - the full constrained operator difference;
    - the free-DoF difference for zero-boundary inputs.

Run from the project root with:

    python experiments/operator_equivalence.py
"""

from pathlib import Path
import csv
import time

import numpy as np

from src.problem import create_problem

from src.assembled import (
    apply_assembled_operator,
    assemble_system,
    enforce_homogeneous_boundary,
)

from src.local_operator import (
    LocalP1Operator,
)

from src.diagnostics import (
    restricted_vector_difference,
    vector_difference,
)


DEFAULT_LEVELS = [
    2,
    3,
    4,
    5,
]

DEFAULT_SEEDS = [
    2026,
    2027,
    2028,
    2029,
    2030,
    2031,
    2032,
    2033,
    2034,
    2035,
]

ROUND_OFF_TARGET = 1.0e-11


def time_operator_application(
    operator_function,
    vector,
    repetitions=5,
):
    """
    Return the minimum elapsed application time.

    Timing is secondary in this experiment. The minimum is recorded
    only as a basic diagnostic and should not be interpreted as a
    performance comparison between NumPy and Firedrake.
    """

    elapsed = []

    for _ in range(repetitions):

        start = time.perf_counter()

        output = operator_function(
            vector
        )

        stop = time.perf_counter()

        elapsed.append(
            stop - start
        )

    return output, float(
        min(elapsed)
    )


def result_row(
    level,
    problem,
    system,
    seed,
    comparison,
    candidate_name,
    diagnostics,
    elapsed_seconds,
):
    """
    Create one flat experiment-record dictionary.
    """

    return {
        "level": level,
        "h": problem.h,
        "ndofs": system.ndofs,
        "num_cells": (
            2
            * (2 ** level)
            * (2 ** level)
        ),
        "seed": seed,
        "comparison": comparison,
        "candidate": candidate_name,
        "absolute_l2": diagnostics.absolute_l2,
        "relative_l2": diagnostics.relative_l2,
        "absolute_linf": diagnostics.absolute_linf,
        "reference_l2": diagnostics.reference_l2,
        "elapsed_seconds": elapsed_seconds,
        "passes_roundoff_target": (
            diagnostics.relative_l2
            <= ROUND_OFF_TARGET
        ),
    }


def run_level(
    level,
    seeds,
    timing_repetitions=5,
):
    """
    Run all operator comparisons for one refinement level.
    """

    print(
        f"\nPreparing refinement level {level}..."
    )

    problem = create_problem(
        refinement_level=level
    )

    system = assemble_system(
        problem
    )

    precomputed_operator = LocalP1Operator(
        problem,
        precompute_matrices=True,
    )

    direct_operator = LocalP1Operator(
        problem,
        precompute_matrices=False,
    )

    rows = []

    for seed in seeds:

        rng = np.random.default_rng(
            seed
        )

        # ------------------------------------------
        # Full constrained operator comparison
        # ------------------------------------------

        vector = rng.standard_normal(
            system.ndofs
        )

        assembled_action, assembled_time = (
            time_operator_application(
                lambda x: apply_assembled_operator(
                    system,
                    x,
                ),
                vector,
                repetitions=timing_repetitions,
            )
        )

        precomputed_action, precomputed_time = (
            time_operator_application(
                precomputed_operator.apply,
                vector,
                repetitions=timing_repetitions,
            )
        )

        direct_action, direct_time = (
            time_operator_application(
                direct_operator.apply,
                vector,
                repetitions=timing_repetitions,
            )
        )

        precomputed_diagnostics = (
            vector_difference(
                assembled_action,
                precomputed_action,
            )
        )

        direct_diagnostics = (
            vector_difference(
                assembled_action,
                direct_action,
            )
        )

        local_mode_diagnostics = (
            vector_difference(
                precomputed_action,
                direct_action,
            )
        )

        rows.append(
            result_row(
                level=level,
                problem=problem,
                system=system,
                seed=seed,
                comparison="full_constrained",
                candidate_name=(
                    "local_precomputed"
                ),
                diagnostics=(
                    precomputed_diagnostics
                ),
                elapsed_seconds=(
                    precomputed_time
                ),
            )
        )

        rows.append(
            result_row(
                level=level,
                problem=problem,
                system=system,
                seed=seed,
                comparison="full_constrained",
                candidate_name=(
                    "local_direct"
                ),
                diagnostics=(
                    direct_diagnostics
                ),
                elapsed_seconds=(
                    direct_time
                ),
            )
        )

        rows.append(
            result_row(
                level=level,
                problem=problem,
                system=system,
                seed=seed,
                comparison="local_mode_difference",
                candidate_name=(
                    "direct_vs_precomputed"
                ),
                diagnostics=(
                    local_mode_diagnostics
                ),
                elapsed_seconds=np.nan,
            )
        )

        # ------------------------------------------
        # Free-DoF comparison
        # ------------------------------------------

        zero_boundary_vector = (
            enforce_homogeneous_boundary(
                system,
                vector,
            )
        )

        assembled_free_action = (
            apply_assembled_operator(
                system,
                zero_boundary_vector,
            )
        )

        precomputed_free_action = (
            precomputed_operator.apply(
                zero_boundary_vector
            )
        )

        direct_free_action = (
            direct_operator.apply(
                zero_boundary_vector
            )
        )

        precomputed_free_diagnostics = (
            restricted_vector_difference(
                assembled_free_action,
                precomputed_free_action,
                system.free_dofs,
            )
        )

        direct_free_diagnostics = (
            restricted_vector_difference(
                assembled_free_action,
                direct_free_action,
                system.free_dofs,
            )
        )

        rows.append(
            result_row(
                level=level,
                problem=problem,
                system=system,
                seed=seed,
                comparison="free_dofs",
                candidate_name=(
                    "local_precomputed"
                ),
                diagnostics=(
                    precomputed_free_diagnostics
                ),
                elapsed_seconds=np.nan,
            )
        )

        rows.append(
            result_row(
                level=level,
                problem=problem,
                system=system,
                seed=seed,
                comparison="free_dofs",
                candidate_name=(
                    "local_direct"
                ),
                diagnostics=(
                    direct_free_diagnostics
                ),
                elapsed_seconds=np.nan,
            )
        )

        # The assembled application time is recorded
        # separately because the candidate rows store
        # candidate timings.
        rows.append(
            {
                "level": level,
                "h": problem.h,
                "ndofs": system.ndofs,
                "num_cells": (
                    precomputed_operator.num_cells
                ),
                "seed": seed,
                "comparison": "timing_only",
                "candidate": "assembled_aij",
                "absolute_l2": np.nan,
                "relative_l2": np.nan,
                "absolute_linf": np.nan,
                "reference_l2": np.nan,
                "elapsed_seconds": (
                    assembled_time
                ),
                "passes_roundoff_target": "",
            }
        )

    return rows


def summarize_rows(rows):
    """
    Print maximum discrepancies for each level and candidate.
    """

    print(
        "\nOperator-equivalence summary"
    )
    print(
        "=" * 104
    )

    header = (
        f"{'Level':>5} "
        f"{'DoFs':>9} "
        f"{'Comparison':>20} "
        f"{'Candidate':>22} "
        f"{'max rel L2':>14} "
        f"{'max abs Linf':>14} "
        f"{'Status':>10}"
    )

    print(header)
    print(
        "-" * len(header)
    )

    keys = sorted(
        {
            (
                row["level"],
                row["comparison"],
                row["candidate"],
            )
            for row in rows
            if row["comparison"]
            != "timing_only"
        }
    )

    for (
        level,
        comparison,
        candidate,
    ) in keys:

        selected = [
            row
            for row in rows
            if (
                row["level"] == level
                and row["comparison"] == comparison
                and row["candidate"] == candidate
            )
        ]

        max_relative_l2 = max(
            row["relative_l2"]
            for row in selected
        )

        max_absolute_linf = max(
            row["absolute_linf"]
            for row in selected
        )

        ndofs = selected[0]["ndofs"]

        status = (
            "PASS"
            if max_relative_l2
            <= ROUND_OFF_TARGET
            else "REVIEW"
        )

        print(
            f"{level:5d} "
            f"{ndofs:9d} "
            f"{comparison:>20} "
            f"{candidate:>22} "
            f"{max_relative_l2:14.6e} "
            f"{max_absolute_linf:14.6e} "
            f"{status:>10}"
        )

    print(
        "=" * 104
    )


def write_results_csv(
    rows,
    output_path,
):
    """
    Write all per-vector results to CSV.
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
        "seed",
        "comparison",
        "candidate",
        "absolute_l2",
        "relative_l2",
        "absolute_linf",
        "reference_l2",
        "elapsed_seconds",
        "passes_roundoff_target",
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


def check_experiment_status(rows):
    """
    Report whether all numerical comparisons meet the target.

    This is an experiment-level status report, not a unit-test
    assertion. All rows are preserved even when review is needed.
    """

    numerical_rows = [
        row
        for row in rows
        if row["comparison"]
        != "timing_only"
    ]

    failed = [
        row
        for row in numerical_rows
        if not row[
            "passes_roundoff_target"
        ]
    ]

    if not failed:

        print(
            "\nPHASE 1 STATUS: PASS"
        )

        print(
            "All tested operator differences are below "
            f"{ROUND_OFF_TARGET:.1e}."
        )

        return True

    print(
        "\nPHASE 1 STATUS: REVIEW REQUIRED"
    )

    print(
        f"{len(failed)} comparison rows exceed "
        f"{ROUND_OFF_TARGET:.1e}."
    )

    worst = max(
        failed,
        key=lambda row: row["relative_l2"],
    )

    print(
        "Worst comparison:"
    )

    print(
        f"  level      = {worst['level']}"
    )

    print(
        f"  seed       = {worst['seed']}"
    )

    print(
        f"  comparison = {worst['comparison']}"
    )

    print(
        f"  candidate  = {worst['candidate']}"
    )

    print(
        f"  relative L2 difference = "
        f"{worst['relative_l2']:.6e}"
    )

    return False


def run_experiment(
    levels=None,
    seeds=None,
    output_path=(
        "results/operator_equivalence.csv"
    ),
    timing_repetitions=5,
):
    """
    Run the complete Phase 1 experiment.
    """

    if levels is None:
        levels = DEFAULT_LEVELS

    if seeds is None:
        seeds = DEFAULT_SEEDS

    print(
        "Phase 1: Operator-equivalence experiment"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Levels: {levels}"
    )

    print(
        f"Seeds:  {seeds}"
    )

    print(
        f"Relative L2 target: "
        f"{ROUND_OFF_TARGET:.1e}"
    )

    rows = []

    for level in levels:

        level_rows = run_level(
            level=level,
            seeds=seeds,
            timing_repetitions=(
                timing_repetitions
            ),
        )

        rows.extend(
            level_rows
        )

    summarize_rows(
        rows
    )

    write_results_csv(
        rows,
        output_path,
    )

    passed = check_experiment_status(
        rows
    )

    return rows, passed


if __name__ == "__main__":

    run_experiment()
