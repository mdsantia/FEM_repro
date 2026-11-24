
"""
Create output-functional convergence and validation plots.

Input
-----
results/functional_convergence.csv

Outputs
-------
results/plots/functional_errors.png
results/plots/functional_rates.png
results/plots/functional_values.png
results/plots/functional_evaluation_agreement.png
results/plots/functional_method_agreement.png

The principal report figures are:

1. functional_errors.png
2. functional_rates.png
3. functional_evaluation_agreement.png

Run from the project root with:

    python experiments/plot_functional_convergence.py

or from a notebook with:

    %run experiments/plot_functional_convergence.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "results/functional_convergence.csv"
)


OUTPUT_DIRECTORY = Path(
    "results/plots"
)


FUNCTIONAL_ORDER = [
    "global_average",
    "subdomain_average",
    "weighted_average",
]


FUNCTIONAL_LABELS = {
    "global_average": (
        "Global average"
    ),
    "subdomain_average": (
        "Subdomain average"
    ),
    "weighted_average": (
        "Weighted average"
    ),
}


FUNCTIONAL_MARKERS = {
    "global_average": "o",
    "subdomain_average": "s",
    "weighted_average": "^",
}


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


def load_results():
    """
    Load and validate functional-convergence results.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. "
            "Run experiments/functional_convergence.py first."
        )

    data = pd.read_csv(
        INPUT_PATH
    )

    required = {
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
        "assembled_relative_residual",
    }

    missing = (
        required
        - set(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            "The functional-convergence CSV is missing columns: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            "The functional-convergence CSV contains no data."
        )

    unknown_functionals = (
        set(
            data[
                "functional"
            ].unique()
        )
        - set(
            FUNCTIONAL_ORDER
        )
    )

    unknown_methods = (
        set(
            data[
                "method"
            ].unique()
        )
        - set(
            METHOD_ORDER
        )
    )

    if unknown_functionals:
        raise ValueError(
            "Unknown functionals: "
            f"{sorted(unknown_functionals)}"
        )

    if unknown_methods:
        raise ValueError(
            "Unknown methods: "
            f"{sorted(unknown_methods)}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    return data


def assembled_rows(
    data,
):
    """
    Return assembled-reference functional rows.
    """

    assembled = (
        data[
            data["method"]
            == "assembled"
        ]
        .copy()
        .sort_values(
            [
                "functional",
                "level",
            ]
        )
    )

    if assembled.empty:
        raise ValueError(
            "No assembled functional rows were found."
        )

    return assembled


# ==================================================
# Principal functional-convergence figures
# ==================================================


def plot_functional_errors(
    data,
):
    """
    Plot all three functional errors against mesh size.
    """

    assembled = assembled_rows(
        data
    )

    figure, axis = plt.subplots(
        figsize=(
            7.3,
            5.1,
        )
    )

    for functional in FUNCTIONAL_ORDER:
        group = assembled[
            assembled[
                "functional"
            ]
            == functional
        ].sort_values(
            "h",
            ascending=False,
        )

        axis.loglog(
            group["h"],
            group[
                "functional_error"
            ],
            marker=FUNCTIONAL_MARKERS[
                functional
            ],
            linewidth=2,
            label=FUNCTIONAL_LABELS[
                functional
            ],
        )

    global_rows = assembled[
        assembled["functional"]
        == "global_average"
    ].sort_values(
        "h",
        ascending=False,
    )

    if len(global_rows) < 2:
        raise ValueError(
            "At least two levels are required for the reference slope."
        )

    anchor = global_rows.iloc[
        1
    ]

    coefficient = (
        anchor[
            "functional_error"
        ]
        / anchor["h"]**2
    )

    guide_h = np.asarray(
        sorted(
            assembled[
                "h"
            ].unique(),
            reverse=True,
        ),
        dtype=np.float64,
    )

    axis.loglog(
        guide_h,
        coefficient
        * guide_h**2,
        linestyle="--",
        color="black",
        label=r"$O(h^2)$ reference",
    )

    axis.invert_xaxis()

    axis.set_xlabel(
        r"Mesh size $h$"
    )

    axis.set_ylabel(
        "Absolute functional error"
    )

    axis.set_title(
        "Output-functional convergence"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    save_figure(
        figure,
        "functional_errors.png",
    )


def plot_observed_rates(
    data,
):
    """
    Plot observed convergence rates for the three functionals.
    """

    assembled = assembled_rows(
        data
    )

    rate_rows = assembled[
        assembled[
            "functional_rate"
        ].notna()
    ].copy()

    levels = sorted(
        rate_rows[
            "level"
        ].unique()
    )

    if not levels:
        raise ValueError(
            "No finite functional convergence rates were found."
        )

    x_positions = np.arange(
        len(levels)
    )

    width = 0.24

    figure, axis = plt.subplots(
        figsize=(
            7.3,
            4.8,
        )
    )

    for (
        functional_index,
        functional,
    ) in enumerate(
        FUNCTIONAL_ORDER
    ):
        group = rate_rows[
            rate_rows[
                "functional"
            ]
            == functional
        ].sort_values(
            "level"
        )

        axis.bar(
            (
                x_positions
                + (
                    functional_index
                    - 1
                )
                * width
            ),
            group[
                "functional_rate"
            ],
            width=width,
            label=FUNCTIONAL_LABELS[
                functional
            ],
        )

    axis.axhline(
        2.0,
        linestyle="--",
        color="black",
        label="Expected rate 2",
    )

    axis.set_xticks(
        x_positions,
        [
            (
                f"Level "
                f"{int(level) - 1}"
                f"→{int(level)}"
            )
            for level in levels
        ],
    )

    axis.set_ylabel(
        "Observed rate"
    )

    axis.set_ylim(
        1.85,
        2.03,
    )

    axis.set_title(
        "Observed functional convergence rates"
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
        "functional_rates.png",
    )


def plot_functional_values(
    data,
):
    """
    Plot computed functional values approaching exact references.
    """

    assembled = assembled_rows(
        data
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            12.8,
            3.9,
        ),
        sharex=True,
    )

    for axis, functional in zip(
        axes,
        FUNCTIONAL_ORDER,
    ):
        group = assembled[
            assembled[
                "functional"
            ]
            == functional
        ].sort_values(
            "ndofs"
        )

        axis.plot(
            group["ndofs"],
            group[
                "functional_value"
            ],
            marker=FUNCTIONAL_MARKERS[
                functional
            ],
            linewidth=2,
            label="FEM output",
        )

        axis.axhline(
            group[
                "exact_value"
            ].iloc[0],
            linestyle="--",
            color="black",
            label=(
                "Exact/reference output"
            ),
        )

        axis.set_xscale(
            "log"
        )

        axis.set_xlabel(
            "Degrees of freedom"
        )

        axis.set_title(
            FUNCTIONAL_LABELS[
                functional
            ]
        )

        axis.grid(
            True,
            alpha=0.3,
        )

    axes[0].set_ylabel(
        "Functional value"
    )

    axes[0].legend(
        fontsize=8
    )

    figure.suptitle(
        "Functional values approaching exact or high-order references"
    )

    save_figure(
        figure,
        "functional_values.png",
    )


# ==================================================
# Functional-evaluation and method-agreement diagnostics
# ==================================================


def plot_evaluation_agreement(
    data,
):
    """
    Plot the local-versus-reference functional-evaluation discrepancy.

    The maximum discrepancy over all solution methods is plotted at
    every refinement level.
    """

    grouped = (
        data.groupby(
            [
                "level",
                "ndofs",
                "functional",
            ],
            as_index=False,
        )[
            "evaluation_relative_difference"
        ].max()
    )

    figure, axis = plt.subplots(
        figsize=(
            7.3,
            4.9,
        )
    )

    for functional in FUNCTIONAL_ORDER:
        group = grouped[
            grouped[
                "functional"
            ]
            == functional
        ].sort_values(
            "ndofs"
        )

        values = group[
            "evaluation_relative_difference"
        ].to_numpy(
            dtype=np.float64
        )

        # Plot exact zeros at a visible floor.
        values = np.where(
            values > 0.0,
            values,
            1.0e-18,
        )

        axis.semilogy(
            group["ndofs"],
            values,
            marker=FUNCTIONAL_MARKERS[
                functional
            ],
            linewidth=2,
            label=FUNCTIONAL_LABELS[
                functional
            ],
        )

    axis.axhline(
        np.finfo(
            np.float64
        ).eps,
        linestyle="--",
        color="black",
        label=(
            "Double-precision epsilon"
        ),
    )

    axis.set_xlabel(
        "Degrees of freedom"
    )

    axis.set_ylabel(
        "Maximum relative local/reference difference"
    )

    axis.set_title(
        "Local functional evaluation agrees to roundoff"
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
        "functional_evaluation_agreement.png",
    )


def plot_method_agreement(
    data,
):
    """
    Plot functional-output agreement with the assembled solution.

    Exact assembled-reference zeros are omitted from the logarithmic
    plot. Only nonassembled methods are shown.
    """

    nonassembled = data[
        data["method"]
        != "assembled"
    ].copy()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            12.0,
            4.6,
        ),
    )

    # --------------------------------------------------
    # Left: functional output differences
    # --------------------------------------------------

    for method in [
        "firedrake_matfree",
        "local_precomputed",
        "local_direct",
    ]:
        method_rows = nonassembled[
            nonassembled[
                "method"
            ]
            == method
        ]

        maximum_by_level = (
            method_rows.groupby(
                [
                    "level",
                    "ndofs",
                ],
                as_index=False,
            )[
                "method_functional_relative_difference"
            ].max()
            .sort_values(
                "ndofs"
            )
        )

        values = maximum_by_level[
            "method_functional_relative_difference"
        ].to_numpy(
            dtype=np.float64
        )

        values = np.where(
            values > 0.0,
            values,
            1.0e-18,
        )

        axes[0].loglog(
            maximum_by_level[
                "ndofs"
            ],
            values,
            marker=METHOD_MARKERS[
                method
            ],
            linewidth=2,
            label=METHOD_LABELS[
                method
            ],
        )

    axes[0].set_xlabel(
        "Degrees of freedom"
    )

    axes[0].set_ylabel(
        "Maximum relative functional-output difference"
    )

    axes[0].set_title(
        "Functional agreement with assembled solution"
    )

    axes[0].grid(
        True,
        which="both",
        alpha=0.3,
    )

    axes[0].legend(
        fontsize=8
    )

    # --------------------------------------------------
    # Right: coefficient-vector differences
    # --------------------------------------------------

    solution_rows = (
        nonassembled.drop_duplicates(
            subset=[
                "level",
                "method",
            ]
        )
    )

    for method in [
        "firedrake_matfree",
        "local_precomputed",
        "local_direct",
    ]:
        group = solution_rows[
            solution_rows[
                "method"
            ]
            == method
        ].sort_values(
            "ndofs"
        )

        axes[1].loglog(
            group["ndofs"],
            group[
                "solution_relative_difference"
            ],
            marker=METHOD_MARKERS[
                method
            ],
            linewidth=2,
            label=METHOD_LABELS[
                method
            ],
        )

    axes[1].set_xlabel(
        "Degrees of freedom"
    )

    axes[1].set_ylabel(
        "Relative solution-vector difference"
    )

    axes[1].set_title(
        "Coefficient-vector agreement"
    )

    axes[1].grid(
        True,
        which="both",
        alpha=0.3,
    )

    axes[1].legend(
        fontsize=8
    )

    save_figure(
        figure,
        "functional_method_agreement.png",
    )


def run_all_plots():
    """
    Generate all functional-convergence and validation plots.
    """

    data = load_results()

    plot_functional_errors(
        data
    )

    plot_observed_rates(
        data
    )

    plot_functional_values(
        data
    )

    plot_evaluation_agreement(
        data
    )

    plot_method_agreement(
        data
    )

    print(
        "\nSaved all functional-convergence plots to:"
    )

    print(
        f"  {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    run_all_plots()
