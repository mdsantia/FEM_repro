
"""
Create consolidated finite-element convergence and diagnostic
performance tables from:

    results/functional_convergence.csv

The source CSV contains one row per:

    mesh level x solution method x functional.

Solution-level quantities such as the L2 error, H1-seminorm error,
iteration count, residual, and elapsed solve time are therefore repeated
for the three functionals. This script validates those repeated values
and reduces the data to one row per mesh level and solution method.

Outputs
-------
results/solution_convergence.csv
    Full convergence table for all four solution methods.

results/solution_convergence_assembled.csv
    Compact convergence table for the assembled reference solution.

results/solution_convergence.md
    Markdown convergence tables suitable for the report.

results/solution_convergence.tex
    LaTeX convergence table for the assembled reference solution.

results/solver_performance.csv
    Diagnostic timing, iteration, and residual table.

results/solver_performance.md
    Markdown diagnostic-performance table.

Important timing limitation
---------------------------
The assembled reference uses preonly with LU, whereas the other methods
use Jacobi-preconditioned CG. The custom local operators execute through
Python and NumPy. Elapsed times therefore diagnose the current code
paths but are not a controlled assembled-versus-matrix-free performance
benchmark.

Run from the project root with:

    python experiments/solution_convergence_table.py

or from a notebook with:

    %run experiments/solution_convergence_table.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "results/functional_convergence.csv"
)


FULL_OUTPUT_PATH = Path(
    "results/solution_convergence.csv"
)


ASSEMBLED_OUTPUT_PATH = Path(
    "results/solution_convergence_assembled.csv"
)


MARKDOWN_OUTPUT_PATH = Path(
    "results/solution_convergence.md"
)


LATEX_OUTPUT_PATH = Path(
    "results/solution_convergence.tex"
)


PERFORMANCE_OUTPUT_PATH = Path(
    "results/solver_performance.csv"
)


PERFORMANCE_MARKDOWN_OUTPUT_PATH = Path(
    "results/solver_performance.md"
)


METHOD_ORDER = [
    "assembled",
    "firedrake_matfree",
    "local_precomputed",
    "local_direct",
]


METHOD_LABELS = {
    "assembled": "Assembled AIJ",
    "firedrake_matfree": (
        "Firedrake matfree"
    ),
    "local_precomputed": (
        "Local precomputed"
    ),
    "local_direct": (
        "Local direct"
    ),
}


SOLVER_LABELS = {
    "assembled": "preonly + LU",
    "firedrake_matfree": (
        "CG + Jacobi"
    ),
    "local_precomputed": (
        "CG + Jacobi"
    ),
    "local_direct": (
        "CG + Jacobi"
    ),
}


REQUIRED_COLUMNS = {
    "level",
    "h",
    "ndofs",
    "num_cells",
    "method",
    "l2_error",
    "h1_seminorm_error",
    "iterations",
    "ksp_residual_norm",
    "assembled_relative_residual",
    "solve_elapsed_seconds",
}


def load_results():
    """
    Load and validate the Phase 2D CSV.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. "
            "Run experiments/functional_convergence.py first."
        )

    data = pd.read_csv(
        INPUT_PATH
    )

    missing = (
        REQUIRED_COLUMNS
        - set(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            "The convergence CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            "The convergence CSV contains no data."
        )

    unknown_methods = (
        set(
            data["method"].unique()
        )
        - set(
            METHOD_ORDER
        )
    )

    if unknown_methods:
        raise ValueError(
            "The convergence CSV contains unknown methods: "
            f"{sorted(unknown_methods)}"
        )

    return data


def verify_repeated_solution_data(
    data,
):
    """
    Verify that solution-level values are constant across functionals.

    Every level-method pair should have identical solution errors,
    residuals, iteration counts, and elapsed times in all functional
    rows.
    """

    repeated_columns = [
        "h",
        "ndofs",
        "num_cells",
        "l2_error",
        "h1_seminorm_error",
        "iterations",
        "ksp_residual_norm",
        "assembled_relative_residual",
        "solve_elapsed_seconds",
    ]

    failures = []

    for (
        level,
        method,
    ), group in data.groupby(
        [
            "level",
            "method",
        ]
    ):
        for column in repeated_columns:
            values = group[
                column
            ].dropna().to_numpy()

            if values.size == 0:
                continue

            if np.issubdtype(
                values.dtype,
                np.number,
            ):
                consistent = np.allclose(
                    values,
                    values[0],
                    rtol=1.0e-13,
                    atol=1.0e-15,
                )
            else:
                consistent = np.all(
                    values
                    == values[0]
                )

            if not consistent:
                failures.append(
                    (
                        level,
                        method,
                        column,
                    )
                )

    if failures:
        details = "\n".join(
            (
                f"  level={level}, "
                f"method={method}, "
                f"column={column}"
            )
            for (
                level,
                method,
                column,
            ) in failures
        )

        raise ValueError(
            "Solution-level data are not constant across "
            f"functional rows:\n{details}"
        )


def reduce_to_solution_rows(
    data,
):
    """
    Return one row per mesh level and solution method.
    """

    columns = [
        "level",
        "h",
        "ndofs",
        "num_cells",
        "method",
        "l2_error",
        "h1_seminorm_error",
        "iterations",
        "ksp_residual_norm",
        "assembled_relative_residual",
        "solve_elapsed_seconds",
    ]

    reduced = (
        data[
            columns
        ]
        .drop_duplicates(
            subset=[
                "level",
                "method",
            ]
        )
        .copy()
    )

    expected_rows = (
        data["level"].nunique()
        * len(
            METHOD_ORDER
        )
    )

    if len(reduced) != expected_rows:
        raise ValueError(
            "The reduced convergence table does not contain "
            "exactly one row per level and method."
        )

    reduced["method"] = pd.Categorical(
        reduced["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    reduced = reduced.sort_values(
        [
            "method",
            "level",
        ]
    ).reset_index(
        drop=True
    )

    return reduced


def observed_rate(
    error_coarse,
    error_fine,
    h_coarse,
    h_fine,
):
    """
    Compute the experimental convergence rate

        log(error_coarse / error_fine)
        --------------------------------.
        log(h_coarse / h_fine)
    """

    values = np.asarray(
        [
            error_coarse,
            error_fine,
            h_coarse,
            h_fine,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(
        values
    ).all():
        return np.nan

    if np.any(
        values <= 0.0
    ):
        return np.nan

    if h_coarse == h_fine:
        return np.nan

    return float(
        np.log(
            error_coarse
            / error_fine
        )
        / np.log(
            h_coarse
            / h_fine
        )
    )


def add_convergence_rates(
    reduced,
):
    """
    Add L2 and H1-seminorm rates independently for each method.
    """

    reduced = reduced.copy()

    reduced["l2_rate"] = np.nan
    reduced["h1_rate"] = np.nan

    for method in METHOD_ORDER:
        method_indices = (
            reduced[
                reduced["method"]
                == method
            ]
            .sort_values(
                "level"
            )
            .index
            .to_list()
        )

        for position in range(
            1,
            len(method_indices),
        ):
            coarse_index = (
                method_indices[
                    position - 1
                ]
            )

            fine_index = (
                method_indices[
                    position
                ]
            )

            coarse = reduced.loc[
                coarse_index
            ]

            fine = reduced.loc[
                fine_index
            ]

            reduced.loc[
                fine_index,
                "l2_rate",
            ] = observed_rate(
                coarse["l2_error"],
                fine["l2_error"],
                coarse["h"],
                fine["h"],
            )

            reduced.loc[
                fine_index,
                "h1_rate",
            ] = observed_rate(
                coarse[
                    "h1_seminorm_error"
                ],
                fine[
                    "h1_seminorm_error"
                ],
                coarse["h"],
                fine["h"],
            )

    return reduced


def validate_convergence(
    reduced,
    l2_rate_bounds=(
        1.7,
        2.3,
    ),
    h1_rate_bounds=(
        0.8,
        1.2,
    ),
):
    """
    Check that observed rates lie in broad P1 verification ranges.
    """

    failures = []

    for _, row in reduced.iterrows():
        if np.isfinite(
            row["l2_rate"]
        ):
            if not (
                l2_rate_bounds[0]
                <= row["l2_rate"]
                <= l2_rate_bounds[1]
            ):
                failures.append(
                    (
                        str(
                            row["method"]
                        ),
                        int(
                            row["level"]
                        ),
                        "L2",
                        float(
                            row["l2_rate"]
                        ),
                    )
                )

        if np.isfinite(
            row["h1_rate"]
        ):
            if not (
                h1_rate_bounds[0]
                <= row["h1_rate"]
                <= h1_rate_bounds[1]
            ):
                failures.append(
                    (
                        str(
                            row["method"]
                        ),
                        int(
                            row["level"]
                        ),
                        "H1 seminorm",
                        float(
                            row["h1_rate"]
                        ),
                    )
                )

    return failures


def format_rate(
    value,
):
    """
    Format a convergence rate for text and report tables.
    """

    if not np.isfinite(
        value
    ):
        return "–"

    return f"{value:.4f}"


def create_performance_table(
    reduced,
):
    """
    Create one diagnostic performance row per level and method.

    Important
    ---------
    The assembled reference uses preonly with LU. The remaining
    implementations use Jacobi-preconditioned CG.

    Elapsed times characterize the current implementation paths.
    They do not constitute a controlled comparison of equivalent
    linear solvers.
    """

    performance = reduced[
        [
            "level",
            "h",
            "ndofs",
            "num_cells",
            "method",
            "iterations",
            "ksp_residual_norm",
            "assembled_relative_residual",
            "solve_elapsed_seconds",
        ]
    ].copy()

    performance[
        "solver_type"
    ] = performance[
        "method"
    ].astype(
        "object"
    ).map(
        SOLVER_LABELS
    )

    performance[
        "method_label"
    ] = performance[
        "method"
    ].astype(
        "object"
    ).map(
        METHOD_LABELS
    )

    if performance[
        "solver_type"
    ].isna().any():
        raise ValueError(
            "Could not assign solver labels to all methods."
        )

    if performance[
        "method_label"
    ].isna().any():
        raise ValueError(
            "Could not assign method labels to all methods."
        )

    performance[
        "time_per_dof"
    ] = (
        performance[
            "solve_elapsed_seconds"
        ]
        / performance[
            "ndofs"
        ]
    )

    performance[
        "time_per_cell"
    ] = (
        performance[
            "solve_elapsed_seconds"
        ]
        / performance[
            "num_cells"
        ]
    )

    # Ratios are computed relative to the assembled elapsed time
    # at the same refinement level.
    #
    # These are implementation-time ratios, not algorithm-independent
    # speedups.
    assembled_times = (
        performance[
            performance["method"]
            == "assembled"
        ][
            [
                "level",
                "solve_elapsed_seconds",
            ]
        ]
        .rename(
            columns={
                "solve_elapsed_seconds":
                "assembled_elapsed_seconds",
            }
        )
    )

    performance = performance.merge(
        assembled_times,
        on="level",
        how="left",
        validate="many_to_one",
    )

    if performance[
        "assembled_elapsed_seconds"
    ].isna().any():
        raise ValueError(
            "Missing assembled timing for at least one level."
        )

    if np.any(
        performance[
            "assembled_elapsed_seconds"
        ] <= 0.0
    ):
        raise ValueError(
            "Assembled elapsed times must be positive."
        )

    performance[
        "time_ratio_to_assembled"
    ] = (
        performance[
            "solve_elapsed_seconds"
        ]
        / performance[
            "assembled_elapsed_seconds"
        ]
    )

    performance = performance.drop(
        columns=[
            "assembled_elapsed_seconds",
        ]
    )

    performance["method"] = pd.Categorical(
        performance["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return performance.sort_values(
        [
            "level",
            "method",
        ]
    ).reset_index(
        drop=True
    )


def print_assembled_table(
    assembled,
):
    """
    Print the principal assembled-reference convergence table.
    """

    print(
        "\nFinite-element solution convergence"
    )

    print(
        "=" * 104
    )

    header = (
        f"{'Level':>5} "
        f"{'h':>10} "
        f"{'DoFs':>8} "
        f"{'L2 error':>14} "
        f"{'L2 rate':>9} "
        f"{'H1 seminorm':>14} "
        f"{'H1 rate':>9} "
        f"{'Residual':>14}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for _, row in assembled.iterrows():
        print(
            f"{int(row['level']):5d} "
            f"{row['h']:10.5f} "
            f"{int(row['ndofs']):8d} "
            f"{row['l2_error']:14.6e} "
            f"{format_rate(row['l2_rate']):>9} "
            f"{row['h1_seminorm_error']:14.6e} "
            f"{format_rate(row['h1_rate']):>9} "
            f"{row['assembled_relative_residual']:14.6e}"
        )

    print(
        "=" * 104
    )


def print_method_summary(
    reduced,
):
    """
    Print convergence-rate ranges across all solution methods.
    """

    print(
        "\nConvergence-rate range across solution methods"
    )

    print(
        "=" * 69
    )

    header = (
        f"{'Level':>5} "
        f"{'min L2 rate':>14} "
        f"{'max L2 rate':>14} "
        f"{'min H1 rate':>14} "
        f"{'max H1 rate':>14}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for level, group in reduced.groupby(
        "level",
        observed=True,
    ):
        l2_values = group[
            "l2_rate"
        ].dropna()

        h1_values = group[
            "h1_rate"
        ].dropna()

        if (
            l2_values.empty
            or h1_values.empty
        ):
            continue

        print(
            f"{int(level):5d} "
            f"{l2_values.min():14.6f} "
            f"{l2_values.max():14.6f} "
            f"{h1_values.min():14.6f} "
            f"{h1_values.max():14.6f}"
        )

    print(
        "=" * 69
    )


def print_performance_table(
    performance,
):
    """
    Print the diagnostic implementation-performance table.
    """

    print(
        "\nDiagnostic solver-performance comparison"
    )

    print(
        "=" * 143
    )

    header = (
        f"{'Level':>5} "
        f"{'DoFs':>8} "
        f"{'Method':>22} "
        f"{'Solver':>15} "
        f"{'Iterations':>10} "
        f"{'Time (s)':>12} "
        f"{'Time/DoF':>12} "
        f"{'Time/cell':>12} "
        f"{'Time ratio':>12} "
        f"{'Rel. residual':>14}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for _, row in performance.iterrows():
        iterations = row[
            "iterations"
        ]

        if pd.isna(
            iterations
        ):
            iteration_text = "–"
        else:
            iteration_text = str(
                int(
                    iterations
                )
            )

        print(
            f"{int(row['level']):5d} "
            f"{int(row['ndofs']):8d} "
            f"{row['method_label']:>22} "
            f"{row['solver_type']:>15} "
            f"{iteration_text:>10} "
            f"{row['solve_elapsed_seconds']:12.6e} "
            f"{row['time_per_dof']:12.6e} "
            f"{row['time_per_cell']:12.6e} "
            f"{row['time_ratio_to_assembled']:12.4f} "
            f"{row['assembled_relative_residual']:14.6e}"
        )

    print(
        "=" * 143
    )

    print(
        "\nTiming qualification:"
    )

    print(
        "  The assembled reference uses preonly with LU."
    )

    print(
        "  The matrix-free implementations use CG with Jacobi."
    )

    print(
        "  The custom local operators use Python, NumPy, "
        "and numpy.add.at."
    )

    print(
        "  The recorded times characterize the present code paths."
    )

    print(
        "  They do not measure intrinsic assembled-versus-"
        "matrix-free performance."
    )


def markdown_table(
    data,
    include_method=False,
):
    """
    Construct a GitHub-flavored Markdown convergence table.
    """

    if include_method:
        lines = [
            (
                "| Method | Level | $h$ | DoFs | "
                "$L^2$ error | Rate | "
                "$H^1$ seminorm error | Rate | "
                "Relative residual |"
            ),
            (
                "|---|---:|---:|---:|---:|---:|"
                "---:|---:|---:|"
            ),
        ]
    else:
        lines = [
            (
                "| Level | $h$ | DoFs | "
                "$L^2$ error | Rate | "
                "$H^1$ seminorm error | Rate | "
                "Relative residual |"
            ),
            (
                "|---:|---:|---:|---:|---:|---:|"
                "---:|---:|"
            ),
        ]

    for _, row in data.iterrows():
        common_values = [
            str(
                int(
                    row["level"]
                )
            ),
            f"{row['h']:.5f}",
            str(
                int(
                    row["ndofs"]
                )
            ),
            f"{row['l2_error']:.6e}",
            format_rate(
                row["l2_rate"]
            ),
            (
                f"{row['h1_seminorm_error']:.6e}"
            ),
            format_rate(
                row["h1_rate"]
            ),
            (
                f"{row['assembled_relative_residual']:.6e}"
            ),
        ]

        if include_method:
            method_name = METHOD_LABELS[
                str(
                    row["method"]
                )
            ]

            values = [
                method_name,
                *common_values,
            ]
        else:
            values = common_values

        lines.append(
            "| "
            + " | ".join(
                values
            )
            + " |"
        )

    return "\n".join(
        lines
    )


def performance_markdown_table(
    performance,
):
    """
    Construct a Markdown diagnostic-performance table.
    """

    lines = [
        (
            "| Level | DoFs | Method | Solver | Iterations | "
            "Time (s) | Time/DoF (s) | Time/cell (s) | "
            "Ratio to assembled | Relative residual |"
        ),
        (
            "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for _, row in performance.iterrows():
        iterations = row[
            "iterations"
        ]

        if pd.isna(
            iterations
        ):
            iteration_text = "–"
        else:
            iteration_text = str(
                int(
                    iterations
                )
            )

        lines.append(
            "| "
            + " | ".join(
                [
                    str(
                        int(
                            row["level"]
                        )
                    ),
                    str(
                        int(
                            row["ndofs"]
                        )
                    ),
                    str(
                        row["method_label"]
                    ),
                    str(
                        row["solver_type"]
                    ),
                    iteration_text,
                    (
                        f"{row['solve_elapsed_seconds']:.6e}"
                    ),
                    (
                        f"{row['time_per_dof']:.6e}"
                    ),
                    (
                        f"{row['time_per_cell']:.6e}"
                    ),
                    (
                        f"{row['time_ratio_to_assembled']:.4f}"
                    ),
                    (
                        f"{row['assembled_relative_residual']:.6e}"
                    ),
                ]
            )
            + " |"
        )

    return "\n".join(
        lines
    )


def write_markdown(
    assembled,
    reduced,
):
    """
    Write assembled and all-method convergence tables to Markdown.
    """

    assembled_table = markdown_table(
        assembled
    )

    all_method_table = markdown_table(
        reduced,
        include_method=True,
    )

    content = (
        "# Finite-Element Solution Convergence\n\n"
        "The table below reports the discretization errors of the "
        "assembled reference solution. The other three operator "
        "implementations produce the same errors and rates to the "
        "displayed precision because their coefficient vectors agree "
        "with the assembled solution to solver tolerance.\n\n"
        "For continuous $P_1$ finite elements on quasi-uniform affine "
        "triangular meshes, the expected behavior is\n\n"
        "$$\n"
        "\\|u-u_h\\|_{L^2(\\Omega)}=O(h^2),\n"
        "\\qquad\n"
        "|u-u_h|_{H^1(\\Omega)}=O(h).\n"
        "$$\n\n"
        "## Assembled reference\n\n"
        f"{assembled_table}\n\n"
        "## All solution methods\n\n"
        f"{all_method_table}\n\n"
        "## Interpretation\n\n"
        "- The $L^2$ rates approach two.\n"
        "- The $H^1$-seminorm rates approach one.\n"
        "- All four solution paths preserve the expected $P_1$ "
        "convergence.\n"
        "- Differences among methods are algebraic-solver effects "
        "far below the finite-element discretization error.\n"
    )

    MARKDOWN_OUTPUT_PATH.write_text(
        content,
        encoding="utf-8",
    )


def write_latex(
    assembled,
):
    """
    Write a compact LaTeX table for the assembled reference.
    """

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Finite-element solution errors and observed "
            r"convergence rates.}"
        ),
        r"\label{tab:solution-convergence}",
        r"\begin{tabular}{rrrrrrr}",
        r"\hline",
        (
            r"Level & DoFs & $h$ & "
            r"$L^2$ error & Rate & "
            r"$H^1$ seminorm error & Rate \\"
        ),
        r"\hline",
    ]

    for _, row in assembled.iterrows():
        lines.append(
            (
                f"{int(row['level'])} & "
                f"{int(row['ndofs'])} & "
                f"{row['h']:.5f} & "
                f"{row['l2_error']:.6e} & "
                f"{format_rate(row['l2_rate'])} & "
                f"{row['h1_seminorm_error']:.6e} & "
                f"{format_rate(row['h1_rate'])} "
                r"\\"
            )
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    LATEX_OUTPUT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def write_performance_outputs(
    performance,
):
    """
    Save the diagnostic performance table as CSV and Markdown.
    """

    output = performance.copy()

    output["method"] = output[
        "method"
    ].astype(
        "object"
    )

    output.to_csv(
        PERFORMANCE_OUTPUT_PATH,
        index=False,
    )

    table = performance_markdown_table(
        performance
    )

    content = (
        "# Diagnostic Solver-Performance Comparison\n\n"
        "The table records elapsed times, iteration counts, and "
        "assembled-system residuals produced by the current "
        "implementations.\n\n"
        "## Important limitation\n\n"
        "This table is not a controlled assembled-versus-matrix-free "
        "performance benchmark:\n\n"
        "- the assembled reference uses `preonly` with LU;\n"
        "- the other methods use CG with Jacobi preconditioning;\n"
        "- the independent local operators execute through Python "
        "and NumPy;\n"
        "- code generation, setup, and one-run timing effects may "
        "be included;\n"
        "- the custom scatter uses `numpy.add.at`.\n\n"
        "The results should therefore be interpreted as implementation "
        "diagnostics.\n\n"
        f"{table}\n"
    )

    PERFORMANCE_MARKDOWN_OUTPUT_PATH.write_text(
        content,
        encoding="utf-8",
    )


def write_outputs(
    reduced,
    assembled,
):
    """
    Save convergence CSV, Markdown, and LaTeX outputs.
    """

    FULL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = reduced.copy()

    output["method"] = output[
        "method"
    ].astype(
        "object"
    )

    assembled_output = (
        assembled.copy()
    )

    assembled_output["method"] = (
        assembled_output[
            "method"
        ].astype(
            "object"
        )
    )

    output.to_csv(
        FULL_OUTPUT_PATH,
        index=False,
    )

    assembled_output.to_csv(
        ASSEMBLED_OUTPUT_PATH,
        index=False,
    )

    write_markdown(
        assembled_output,
        output,
    )

    write_latex(
        assembled_output,
    )

    print(
        "\nSaved convergence outputs:"
    )

    for path in [
        FULL_OUTPUT_PATH,
        ASSEMBLED_OUTPUT_PATH,
        MARKDOWN_OUTPUT_PATH,
        LATEX_OUTPUT_PATH,
    ]:
        print(
            f"  {path}"
        )


def run_experiment():
    """
    Build, validate, print, and save the convergence and diagnostic
    performance tables.
    """

    print(
        "Combined finite-element solution convergence table"
    )

    print(
        "--------------------------------------------------"
    )

    data = load_results()

    verify_repeated_solution_data(
        data
    )

    reduced = reduce_to_solution_rows(
        data
    )

    reduced = add_convergence_rates(
        reduced
    )

    assembled = (
        reduced[
            reduced["method"]
            == "assembled"
        ]
        .copy()
        .sort_values(
            "level"
        )
    )

    performance = create_performance_table(
        reduced
    )

    print_assembled_table(
        assembled
    )

    print_method_summary(
        reduced
    )

    print_performance_table(
        performance
    )

    failures = validate_convergence(
        reduced
    )

    print(
        "\nConvergence validation"
    )

    print(
        "----------------------"
    )

    if failures:
        print(
            "STATUS: REVIEW REQUIRED"
        )

        for (
            method,
            level,
            norm_name,
            rate,
        ) in failures:
            print(
                f"  method={method}, "
                f"level={level}, "
                f"norm={norm_name}, "
                f"rate={rate:.6f}"
            )
    else:
        print(
            "STATUS: PASS"
        )

        print(
            "All observed L2 and H1-seminorm rates "
            "lie in the verification ranges."
        )

    write_outputs(
        reduced,
        assembled,
    )

    write_performance_outputs(
        performance
    )

    print(
        "\nSaved performance outputs:"
    )

    print(
        f"  {PERFORMANCE_OUTPUT_PATH}"
    )

    print(
        f"  {PERFORMANCE_MARKDOWN_OUTPUT_PATH}"
    )

    return (
        reduced,
        assembled,
        performance,
        failures,
    )


if __name__ == "__main__":
    run_experiment()
