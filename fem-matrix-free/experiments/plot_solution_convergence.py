
"""
Create finite-element solution convergence and solver diagnostic plots.

Inputs
------
results/solution_convergence.csv
results/solver_performance.csv

Outputs
-------
results/plots/solution_errors.png
results/plots/solution_rates.png
results/plots/solution_error_vs_dofs.png
results/plots/cg_iterations.png
results/plots/solver_residuals.png
results/plots/solver_times_diagnostic.png
results/plots/solver_time_per_dof_diagnostic.png

Interpretation
--------------
The solution-error and rate plots are finite-element verification
results.

The iteration, residual, and timing plots are implementation
diagnostics. In particular, the timing plot is not a controlled
assembled-versus-matrix-free benchmark because:

- the assembled reference uses preonly with LU;
- the remaining methods use CG with Jacobi;
- the custom local operators execute through Python and NumPy;
- setup and first-use overhead may be included.

Run from the project root with:

    python experiments/plot_solution_convergence.py

or from a notebook with:

    %run experiments/plot_solution_convergence.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONVERGENCE_PATH = Path(
    "results/solution_convergence.csv"
)


PERFORMANCE_PATH = Path(
    "results/solver_performance.csv"
)


OUTPUT_DIRECTORY = Path(
    "results/plots"
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


METHOD_MARKERS = {
    "assembled": "o",
    "firedrake_matfree": "s",
    "local_precomputed": "^",
    "local_direct": "D",
}


METHOD_LINESTYLES = {
    "assembled": "-",
    "firedrake_matfree": "--",
    "local_precomputed": "-.",
    "local_direct": ":",
}


def load_data():
    """
    Load and validate convergence and performance data.
    """

    if not CONVERGENCE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CONVERGENCE_PATH}. "
            "Run experiments/solution_convergence_table.py first."
        )

    if not PERFORMANCE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PERFORMANCE_PATH}. "
            "Run experiments/solution_convergence_table.py first."
        )

    convergence = pd.read_csv(
        CONVERGENCE_PATH
    )

    performance = pd.read_csv(
        PERFORMANCE_PATH
    )

    convergence_required = {
        "level",
        "h",
        "ndofs",
        "method",
        "l2_error",
        "h1_seminorm_error",
        "l2_rate",
        "h1_rate",
        "assembled_relative_residual",
    }

    performance_required = {
        "level",
        "h",
        "ndofs",
        "num_cells",
        "method",
        "iterations",
        "solve_elapsed_seconds",
        "time_per_dof",
        "time_per_cell",
        "time_ratio_to_assembled",
        "assembled_relative_residual",
    }

    missing_convergence = (
        convergence_required
        - set(
            convergence.columns
        )
    )

    missing_performance = (
        performance_required
        - set(
            performance.columns
        )
    )

    if missing_convergence:
        raise ValueError(
            "The convergence CSV is missing columns: "
            f"{sorted(missing_convergence)}"
        )

    if missing_performance:
        raise ValueError(
            "The performance CSV is missing columns: "
            f"{sorted(missing_performance)}"
        )

    unknown_convergence_methods = (
        set(
            convergence["method"].unique()
        )
        - set(
            METHOD_ORDER
        )
    )

    unknown_performance_methods = (
        set(
            performance["method"].unique()
        )
        - set(
            METHOD_ORDER
        )
    )

    if unknown_convergence_methods:
        raise ValueError(
            "Unknown convergence methods: "
            f"{sorted(unknown_convergence_methods)}"
        )

    if unknown_performance_methods:
        raise ValueError(
            "Unknown performance methods: "
            f"{sorted(unknown_performance_methods)}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        convergence,
        performance,
    )


def assembled_rows(
    convergence,
):
    """
    Return assembled-reference rows ordered from coarse to fine.
    """

    data = (
        convergence[
            convergence["method"]
            == "assembled"
        ]
        .sort_values(
            "level"
        )
        .copy()
    )

    if data.empty:
        raise ValueError(
            "No assembled convergence rows were found."
        )

    return data


def save_figure(
    figure,
    filename,
):
    """
    Save and close one figure.
    """

    output_path = (
        OUTPUT_DIRECTORY
        / filename
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {output_path}"
    )


# ==================================================
# Finite-element convergence plots
# ==================================================


def plot_solution_errors(
    convergence,
):
    """
    Plot L2 and H1-seminorm errors against mesh size.
    """

    data = assembled_rows(
        convergence
    )

    h_values = data[
        "h"
    ].to_numpy(
        dtype=np.float64
    )

    l2_values = data[
        "l2_error"
    ].to_numpy(
        dtype=np.float64
    )

    h1_values = data[
        "h1_seminorm_error"
    ].to_numpy(
        dtype=np.float64
    )

    if len(data) < 2:
        raise ValueError(
            "At least two refinement levels are required."
        )

    # Anchor both reference slopes at the second mesh.
    anchor_index = 1

    l2_coefficient = (
        l2_values[
            anchor_index
        ]
        / h_values[
            anchor_index
        ]**2
    )

    h1_coefficient = (
        h1_values[
            anchor_index
        ]
        / h_values[
            anchor_index
        ]
    )

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            5.1,
        )
    )

    axis.loglog(
        h_values,
        l2_values,
        marker="o",
        linewidth=2,
        label=r"$L^2$ error",
    )

    axis.loglog(
        h_values,
        h1_values,
        marker="s",
        linewidth=2,
        label=(
            r"$H^1$ seminorm error"
        ),
    )

    axis.loglog(
        h_values,
        l2_coefficient
        * h_values**2,
        linestyle="--",
        color="black",
        label=r"$O(h^2)$ reference",
    )

    axis.loglog(
        h_values,
        h1_coefficient
        * h_values,
        linestyle=":",
        color="black",
        label=r"$O(h)$ reference",
    )

    axis.invert_xaxis()

    axis.set_xlabel(
        r"Mesh size $h$"
    )

    axis.set_ylabel(
        "Error"
    )

    axis.set_title(
        "Finite-element solution convergence"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    save_figure(
        figure,
        "solution_errors.png",
    )


def plot_observed_rates(
    convergence,
):
    """
    Plot observed L2 and H1-seminorm convergence rates.
    """

    data = assembled_rows(
        convergence
    )

    data = data[
        data["l2_rate"].notna()
        & data["h1_rate"].notna()
    ].copy()

    if data.empty:
        raise ValueError(
            "No finite convergence-rate rows were found."
        )

    x_positions = np.arange(
        len(data)
    )

    width = 0.34

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            4.8,
        )
    )

    axis.bar(
        x_positions
        - width / 2.0,
        data["l2_rate"],
        width=width,
        label=r"$L^2$ rate",
    )

    axis.bar(
        x_positions
        + width / 2.0,
        data["h1_rate"],
        width=width,
        label=(
            r"$H^1$ seminorm rate"
        ),
    )

    axis.axhline(
        2.0,
        linestyle="--",
        color="black",
        label=(
            r"Expected $L^2$ rate"
        ),
    )

    axis.axhline(
        1.0,
        linestyle=":",
        color="black",
        label=(
            r"Expected $H^1$ rate"
        ),
    )

    axis.set_xticks(
        x_positions,
        [
            (
                f"Level "
                f"{int(level) - 1}"
                f"→{int(level)}"
            )
            for level in data[
                "level"
            ]
        ],
    )

    axis.set_ylabel(
        "Observed rate"
    )

    axis.set_ylim(
        0.0,
        2.15,
    )

    axis.set_title(
        "Observed solution convergence rates"
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    axis.legend(
        ncol=2,
        fontsize=8,
    )

    save_figure(
        figure,
        "solution_rates.png",
    )


def plot_error_vs_dofs(
    convergence,
):
    """
    Plot solution error against degrees of freedom.
    """

    data = assembled_rows(
        convergence
    )

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            5.0,
        )
    )

    axis.loglog(
        data["ndofs"],
        data["l2_error"],
        marker="o",
        linewidth=2,
        label=r"$L^2$ error",
    )

    axis.loglog(
        data["ndofs"],
        data[
            "h1_seminorm_error"
        ],
        marker="s",
        linewidth=2,
        label=(
            r"$H^1$ seminorm error"
        ),
    )

    axis.set_xlabel(
        "Degrees of freedom"
    )

    axis.set_ylabel(
        "Error"
    )

    axis.set_title(
        "Solution error versus problem size"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    save_figure(
        figure,
        "solution_error_vs_dofs.png",
    )


# ==================================================
# Solver diagnostic plots
# ==================================================


def plot_iterations(
    performance,
):
    """
    Plot CG iterations for matrix-free implementations.

    The assembled LU solve is excluded because its reported iteration
    count is not comparable with CG iterations.
    """

    iterative_methods = [
        "firedrake_matfree",
        "local_precomputed",
        "local_direct",
    ]

    iterative = performance[
        performance["method"].isin(
            iterative_methods
        )
    ]

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            4.8,
        )
    )

    for method in iterative_methods:
        group = iterative[
            iterative["method"]
            == method
        ].sort_values(
            "ndofs"
        )

        axis.plot(
            group["ndofs"],
            group["iterations"],
            marker=METHOD_MARKERS[
                method
            ],
            linestyle=(
                METHOD_LINESTYLES[
                    method
                ]
            ),
            linewidth=2,
            label=METHOD_LABELS[
                method
            ],
        )

    axis.set_xscale(
        "log"
    )

    axis.set_xlabel(
        "Degrees of freedom"
    )

    axis.set_ylabel(
        "CG iterations"
    )

    axis.set_title(
        "Jacobi-preconditioned CG iteration growth"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    save_figure(
        figure,
        "cg_iterations.png",
    )


def plot_residuals(
    performance,
):
    """
    Plot residuals measured using the assembled reference system.
    """

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            4.8,
        )
    )

    for method in METHOD_ORDER:
        group = performance[
            performance["method"]
            == method
        ].sort_values(
            "ndofs"
        )

        axis.loglog(
            group["ndofs"],
            group[
                "assembled_relative_residual"
            ],
            marker=METHOD_MARKERS[
                method
            ],
            linestyle=(
                METHOD_LINESTYLES[
                    method
                ]
            ),
            linewidth=2,
            label=METHOD_LABELS[
                method
            ],
        )

    axis.set_xlabel(
        "Degrees of freedom"
    )

    axis.set_ylabel(
        "Assembled relative residual"
    )

    axis.set_title(
        "Algebraic residual under mesh refinement"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend(
        fontsize=8
    )

    save_figure(
        figure,
        "solver_residuals.png",
    )


def plot_solver_times(
    performance,
):
    """
    Plot recorded solve times.

    This is a diagnostic plot, not a controlled performance benchmark.
    """

    figure, axis = plt.subplots(
        figsize=(
            7.4,
            4.9,
        )
    )

    for method in METHOD_ORDER:
        group = performance[
            performance["method"]
            == method
        ].sort_values(
            "ndofs"
        )

        axis.loglog(
            group["ndofs"],
            group[
                "solve_elapsed_seconds"
            ],
            marker=METHOD_MARKERS[
                method
            ],
            linestyle=(
                METHOD_LINESTYLES[
                    method
                ]
            ),
            linewidth=2,
            label=METHOD_LABELS[
                method
            ],
        )

    axis.set_xlabel(
        "Degrees of freedom"
    )

    axis.set_ylabel(
        "Recorded solve elapsed time (s)"
    )

    axis.set_title(
        "Solver timings, diagnostic only"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend(
        fontsize=8
    )

    save_figure(
        figure,
        "solver_times_diagnostic.png",
    )


def plot_time_per_dof(
    performance,
):
    """
    Plot recorded solve time normalized by degrees of freedom.
    """

    figure, axis = plt.subplots(
        figsize=(
            7.4,
            4.9,
        )
    )

    for method in METHOD_ORDER:
        group = performance[
            performance["method"]
            == method
        ].sort_values(
            "ndofs"
        )

        axis.loglog(
            group["ndofs"],
            group["time_per_dof"],
            marker=METHOD_MARKERS[
                method
            ],
            linestyle=(
                METHOD_LINESTYLES[
                    method
                ]
            ),
            linewidth=2,
            label=METHOD_LABELS[
                method
            ],
        )

    axis.set_xlabel(
        "Degrees of freedom"
    )

    axis.set_ylabel(
        "Recorded solve time per DoF (s)"
    )

    axis.set_title(
        "Normalized solver timing, diagnostic only"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend(
        fontsize=8
    )

    save_figure(
        figure,
        "solver_time_per_dof_diagnostic.png",
    )


def run_all_plots():
    """
    Generate all solution-convergence and solver diagnostic plots.
    """

    (
        convergence,
        performance,
    ) = load_data()

    plot_solution_errors(
        convergence
    )

    plot_observed_rates(
        convergence
    )

    plot_error_vs_dofs(
        convergence
    )

    plot_iterations(
        performance
    )

    plot_residuals(
        performance
    )

    plot_solver_times(
        performance
    )

    plot_time_per_dof(
        performance
    )

    print(
        "\nSaved all solution-convergence plots to:"
    )

    print(
        f"  {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    run_all_plots()
