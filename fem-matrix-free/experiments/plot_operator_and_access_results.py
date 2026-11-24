
"""
Create operator-equivalence and logical access-model plots.

Inputs
------
results/operator_equivalence.csv
results/access_model.csv

Outputs
-------
Operator-equivalence figures:

    results/plots/operator_max_relative_difference.png
    results/plots/operator_seed_distribution.png
    results/plots/operator_max_entrywise_difference.png
    results/plots/operator_timings_diagnostic.png

Access-model figures:

    results/plots/access_total_scalar_reads.png
    results/plots/access_fraction.png
    results/plots/access_persistent_storage.png
    results/plots/access_accumulation_structure.png
    results/plots/access_arithmetic_vs_reads.png

Important interpretation
------------------------
The access-model plots report logical representation counts, not measured
cache or memory traffic.

The operator timing plot is diagnostic only. The custom local operators
are NumPy implementations and are not optimized reproductions of
compiled matrix-free kernels.

Run from the project root with:

    python experiments/plot_operator_and_access_results.py

or from a notebook with:

    %run experiments/plot_operator_and_access_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OPERATOR_INPUT = Path(
    "results/operator_equivalence.csv"
)


ACCESS_INPUT = Path(
    "results/access_model.csv"
)


OUTPUT_DIRECTORY = Path(
    "results/plots"
)


COMPARISON_LABELS = {
    (
        "full_constrained",
        "local_precomputed",
    ): (
        "Precomputed, constrained"
    ),

    (
        "full_constrained",
        "local_direct",
    ): (
        "Direct, constrained"
    ),

    (
        "free_dofs",
        "local_precomputed",
    ): (
        "Precomputed, free DoFs"
    ),

    (
        "free_dofs",
        "local_direct",
    ): (
        "Direct, free DoFs"
    ),

    (
        "local_mode_difference",
        "direct_vs_precomputed",
    ): (
        "Direct vs precomputed"
    ),
}


OPERATION_LABELS = {
    "operator_precomputed": (
        "Precomputed operator"
    ),
    "operator_direct": (
        "Direct operator"
    ),
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


OPERATION_ORDER = [
    "operator_precomputed",
    "operator_direct",
    "global_average",
    "subdomain_average",
    "weighted_average",
]


OPERATION_MARKERS = {
    "operator_precomputed": "o",
    "operator_direct": "s",
    "global_average": "^",
    "subdomain_average": "D",
    "weighted_average": "v",
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


def load_operator_results():
    """
    Load and validate operator-equivalence results.
    """

    if not OPERATOR_INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {OPERATOR_INPUT}. "
            "Run experiments/operator_equivalence.py first."
        )

    data = pd.read_csv(
        OPERATOR_INPUT
    )

    required = {
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
        "elapsed_seconds",
    }

    missing = (
        required
        - set(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            "The operator-equivalence CSV is missing columns: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            "The operator-equivalence CSV contains no data."
        )

    return data


def load_access_results():
    """
    Load and validate the corrected access-model CSV.
    """

    if not ACCESS_INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {ACCESS_INPUT}. "
            "Run experiments/access_model.py first."
        )

    data = pd.read_csv(
        ACCESS_INPUT
    )

    required = {
        "level",
        "h",
        "ndofs",
        "total_cells",
        "operation",
        "accessed_cells",
        "access_fraction",
        "storage_cells",
        "storage_fraction",
        "total_scalar_reads",
        "total_vector_scatter_adds",
        "total_scalar_reduction_contributions",
        "total_logical_multiplications",
        "total_logical_additions",
        "total_logical_arithmetic_operations",
        "persistent_float_storage_bytes",
        "persistent_index_storage_bytes",
        "persistent_total_local_storage_bytes",
    }

    missing = (
        required
        - set(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            "The access-model CSV is missing columns: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            "The access-model CSV contains no data."
        )

    unknown_operations = (
        set(
            data[
                "operation"
            ].unique()
        )
        - set(
            OPERATION_ORDER
        )
    )

    if unknown_operations:
        raise ValueError(
            "The access-model CSV contains unknown operations: "
            f"{sorted(unknown_operations)}"
        )

    return data


def comparison_label(
    comparison,
    candidate,
):
    """
    Return a readable comparison label.
    """

    key = (
        comparison,
        candidate,
    )

    if key not in COMPARISON_LABELS:
        raise KeyError(
            f"Unknown operator comparison: {key}"
        )

    return COMPARISON_LABELS[
        key
    ]


def ordered_access_data(
    data,
):
    """
    Return access rows in a stable operation order.
    """

    ordered = data.copy()

    ordered["operation"] = pd.Categorical(
        ordered["operation"],
        categories=OPERATION_ORDER,
        ordered=True,
    )

    return ordered.sort_values(
        [
            "level",
            "operation",
        ]
    )


# ==================================================
# Operator-equivalence plots
# ==================================================


def plot_operator_max_relative_difference(
    data,
):
    """
    Plot maximum relative operator differences over random seeds.
    """

    numerical = data[
        data["comparison"]
        != "timing_only"
    ].copy()

    grouped = (
        numerical.groupby(
            [
                "level",
                "ndofs",
                "comparison",
                "candidate",
            ],
            as_index=False,
        )[
            "relative_l2"
        ].max()
    )

    figure, axis = plt.subplots(
        figsize=(
            8.0,
            5.2,
        )
    )

    for (
        comparison,
        candidate,
    ), group in grouped.groupby(
        [
            "comparison",
            "candidate",
        ]
    ):
        group = group.sort_values(
            "ndofs"
        )

        values = group[
            "relative_l2"
        ].to_numpy(
            dtype=np.float64
        )

        # Exact zeros cannot be placed on a logarithmic axis.
        values = np.where(
            values > 0.0,
            values,
            1.0e-18,
        )

        axis.semilogy(
            group["ndofs"],
            values,
            marker="o",
            linewidth=2,
            label=comparison_label(
                comparison,
                candidate,
            ),
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
        "Maximum relative operator difference"
    )

    axis.set_title(
        "Operator equivalence across mesh refinement"
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
        "operator_max_relative_difference.png",
    )


def plot_operator_seed_distribution(
    data,
):
    """
    Plot seed-level relative differences for principal comparisons.
    """

    numerical = data[
        data["comparison"].isin(
            [
                "full_constrained",
                "local_mode_difference",
            ]
        )
    ].copy()

    numerical["series"] = [
        comparison_label(
            comparison,
            candidate,
        )
        for comparison, candidate in zip(
            numerical["comparison"],
            numerical["candidate"],
        )
    ]

    series_order = [
        "Precomputed, constrained",
        "Direct, constrained",
        "Direct vs precomputed",
    ]

    levels = sorted(
        numerical[
            "level"
        ].unique()
    )

    figure, axes = plt.subplots(
        1,
        len(levels),
        figsize=(
            3.6 * len(levels),
            4.5,
        ),
        sharey=True,
    )

    if len(levels) == 1:
        axes = [
            axes
        ]

    rng = np.random.default_rng(
        2026
    )

    for axis, level in zip(
        axes,
        levels,
    ):
        level_data = numerical[
            numerical["level"]
            == level
        ]

        for (
            series_index,
            series,
        ) in enumerate(
            series_order
        ):
            values = level_data[
                level_data["series"]
                == series
            ][
                "relative_l2"
            ].to_numpy(
                dtype=np.float64
            )

            values = np.where(
                values > 0.0,
                values,
                1.0e-18,
            )

            jitter = rng.uniform(
                -0.08,
                0.08,
                size=values.size,
            )

            axis.scatter(
                (
                    np.full(
                        values.size,
                        series_index,
                    )
                    + jitter
                ),
                values,
                s=28,
                alpha=0.8,
            )

        axis.axhline(
            np.finfo(
                np.float64
            ).eps,
            linestyle="--",
            color="black",
            linewidth=1,
        )

        axis.set_yscale(
            "log"
        )

        axis.set_xticks(
            range(
                len(
                    series_order
                )
            ),
            [
                "Precomputed",
                "Direct",
                (
                    "Direct vs\n"
                    "precomputed"
                ),
            ],
            rotation=20,
        )

        axis.set_title(
            f"Level {level}"
        )

        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

    axes[0].set_ylabel(
        "Relative operator difference"
    )

    figure.suptitle(
        "Operator differences across seeded vectors"
    )

    save_figure(
        figure,
        "operator_seed_distribution.png",
    )


def plot_operator_entrywise_difference(
    data,
):
    """
    Plot maximum absolute entrywise operator differences.
    """

    numerical = data[
        data["comparison"]
        != "timing_only"
    ].copy()

    grouped = (
        numerical.groupby(
            [
                "level",
                "ndofs",
                "comparison",
                "candidate",
            ],
            as_index=False,
        )[
            "absolute_linf"
        ].max()
    )

    figure, axis = plt.subplots(
        figsize=(
            8.0,
            5.2,
        )
    )

    for (
        comparison,
        candidate,
    ), group in grouped.groupby(
        [
            "comparison",
            "candidate",
        ]
    ):
        group = group.sort_values(
            "ndofs"
        )

        values = group[
            "absolute_linf"
        ].to_numpy(
            dtype=np.float64
        )

        values = np.where(
            values > 0.0,
            values,
            1.0e-18,
        )

        axis.semilogy(
            group["ndofs"],
            values,
            marker="o",
            linewidth=2,
            label=comparison_label(
                comparison,
                candidate,
            ),
        )

    axis.set_xlabel(
        "Degrees of freedom"
    )

    axis.set_ylabel(
        (
            "Maximum absolute "
            r"$\ell^\infty$ difference"
        )
    )

    axis.set_title(
        "Maximum entrywise operator discrepancy"
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
        "operator_max_entrywise_difference.png",
    )


def plot_operator_timings(
    data,
):
    """
    Plot median operator-application timing by mesh size.

    Timing rows represent:
    - assembled AIJ application;
    - precomputed local action;
    - direct local action.
    """

    timed = data[
        data["elapsed_seconds"].notna()
    ].copy()

    timing_labels = {
        "assembled_aij": (
            "Assembled AIJ"
        ),
        "local_precomputed": (
            "Local precomputed"
        ),
        "local_direct": (
            "Local direct"
        ),
    }

    grouped = (
        timed.groupby(
            [
                "level",
                "ndofs",
                "candidate",
            ],
            as_index=False,
        )[
            "elapsed_seconds"
        ].median()
    )

    figure, axis = plt.subplots(
        figsize=(
            7.4,
            4.8,
        )
    )

    for candidate in [
        "assembled_aij",
        "local_precomputed",
        "local_direct",
    ]:
        group = grouped[
            grouped["candidate"]
            == candidate
        ].sort_values(
            "ndofs"
        )

        if group.empty:
            continue

        axis.loglog(
            group["ndofs"],
            group["elapsed_seconds"],
            marker="o",
            linewidth=2,
            label=timing_labels[
                candidate
            ],
        )

    axis.set_xlabel(
        "Degrees of freedom"
    )

    axis.set_ylabel(
        "Median application time (s)"
    )

    axis.set_title(
        "Operator-application timings, diagnostic only"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    save_figure(
        figure,
        "operator_timings_diagnostic.png",
    )


# ==================================================
# Access-model plots
# ==================================================


def plot_total_scalar_reads(
    data,
):
    """
    Plot total logical scalar reads as the mesh grows.
    """

    ordered = ordered_access_data(
        data
    )

    figure, axis = plt.subplots(
        figsize=(
            7.6,
            5.0,
        )
    )

    for operation in OPERATION_ORDER:
        group = ordered[
            ordered["operation"]
            == operation
        ].sort_values(
            "total_cells"
        )

        axis.loglog(
            group["total_cells"],
            group["total_scalar_reads"],
            marker=OPERATION_MARKERS[
                operation
            ],
            linewidth=2,
            label=OPERATION_LABELS[
                operation
            ],
        )

    axis.set_xlabel(
        "Total mesh cells"
    )

    axis.set_ylabel(
        "Logical scalar reads per evaluation"
    )

    axis.set_title(
        "Logical scalar-access growth"
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
        "access_total_scalar_reads.png",
    )


def plot_access_fraction(
    data,
):
    """
    Plot the fraction of mesh cells accessed at the finest level.
    """

    finest_level = int(
        data["level"].max()
    )

    finest = ordered_access_data(
        data[
            data["level"]
            == finest_level
        ]
    )

    labels = [
        OPERATION_LABELS[
            str(operation)
        ]
        for operation in finest[
            "operation"
        ]
    ]

    figure, axis = plt.subplots(
        figsize=(
            7.8,
            4.8,
        )
    )

    axis.bar(
        labels,
        finest[
            "access_fraction"
        ],
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_ylabel(
        "Fraction of mesh cells accessed"
    )

    axis.set_title(
        (
            "Cell-access fraction at "
            f"level {finest_level}"
        )
    )

    axis.tick_params(
        axis="x",
        rotation=20,
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    save_figure(
        figure,
        "access_fraction.png",
    )


def plot_persistent_storage(
    data,
):
    """
    Plot operation-specific persistent storage at the finest level.
    """

    finest_level = int(
        data["level"].max()
    )

    finest = ordered_access_data(
        data[
            data["level"]
            == finest_level
        ]
    )

    labels = [
        OPERATION_LABELS[
            str(operation)
        ]
        for operation in finest[
            "operation"
        ]
    ]

    x_positions = np.arange(
        len(labels)
    )

    float_bytes = finest[
        "persistent_float_storage_bytes"
    ].to_numpy(
        dtype=np.float64
    )

    index_bytes = finest[
        "persistent_index_storage_bytes"
    ].to_numpy(
        dtype=np.float64
    )

    figure, axis = plt.subplots(
        figsize=(
            8.2,
            5.0,
        )
    )

    axis.bar(
        x_positions,
        float_bytes,
        label=(
            "Floating-point local data"
        ),
    )

    axis.bar(
        x_positions,
        index_bytes,
        bottom=float_bytes,
        label=(
            "Integer connectivity/index data"
        ),
    )

    axis.set_xticks(
        x_positions,
        labels,
        rotation=20,
    )

    axis.set_ylabel(
        "Persistent operation-specific storage (bytes)"
    )

    axis.set_title(
        (
            "Logical persistent local storage at "
            f"level {finest_level}"
        )
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    save_figure(
        figure,
        "access_persistent_storage.png",
    )


def plot_accumulation_structure(
    data,
):
    """
    Contrast vector scatter-adds with scalar reductions.
    """

    finest_level = int(
        data["level"].max()
    )

    finest = ordered_access_data(
        data[
            data["level"]
            == finest_level
        ]
    )

    labels = [
        OPERATION_LABELS[
            str(operation)
        ]
        for operation in finest[
            "operation"
        ]
    ]

    x_positions = np.arange(
        len(labels)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(
            8.2,
            5.0,
        )
    )

    axis.bar(
        x_positions
        - width / 2.0,
        finest[
            "total_vector_scatter_adds"
        ],
        width=width,
        label=(
            "Vector scatter-adds"
        ),
    )

    axis.bar(
        x_positions
        + width / 2.0,
        finest[
            "total_scalar_reduction_contributions"
        ],
        width=width,
        label=(
            "Scalar reduction contributions"
        ),
    )

    axis.set_xticks(
        x_positions,
        labels,
        rotation=20,
    )

    axis.set_ylabel(
        "Total accumulation contributions"
    )

    axis.set_title(
        (
            "Accumulation structure at "
            f"level {finest_level}"
        )
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    save_figure(
        figure,
        "access_accumulation_structure.png",
    )


def plot_arithmetic_vs_reads(
    data,
):
    """
    Plot logical arithmetic against logical scalar reads.
    """

    finest_level = int(
        data["level"].max()
    )

    finest = ordered_access_data(
        data[
            data["level"]
            == finest_level
        ]
    )

    figure, axis = plt.subplots(
        figsize=(
            7.4,
            5.2,
        )
    )

    for _, row in finest.iterrows():
        operation = str(
            row["operation"]
        )

        scalar_reads = float(
            row[
                "total_scalar_reads"
            ]
        )

        arithmetic = float(
            row[
                "total_logical_arithmetic_operations"
            ]
        )

        axis.scatter(
            scalar_reads,
            arithmetic,
            s=75,
            marker=OPERATION_MARKERS[
                operation
            ],
        )

        axis.annotate(
            OPERATION_LABELS[
                operation
            ],
            (
                scalar_reads,
                arithmetic,
            ),
            xytext=(
                6,
                5,
            ),
            textcoords=(
                "offset points"
            ),
            fontsize=8,
        )

    axis.set_xscale(
        "log"
    )

    axis.set_yscale(
        "log"
    )

    axis.set_xlabel(
        "Logical scalar reads"
    )

    axis.set_ylabel(
        "Logical arithmetic operations"
    )

    axis.set_title(
        (
            "Arithmetic versus local-data access at "
            f"level {finest_level}"
        )
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    save_figure(
        figure,
        "access_arithmetic_vs_reads.png",
    )


def run_all_plots():
    """
    Generate all operator-equivalence and access-model figures.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    operator_data = (
        load_operator_results()
    )

    access_data = (
        load_access_results()
    )

    plot_operator_max_relative_difference(
        operator_data
    )

    plot_operator_seed_distribution(
        operator_data
    )

    plot_operator_entrywise_difference(
        operator_data
    )

    plot_operator_timings(
        operator_data
    )

    plot_total_scalar_reads(
        access_data
    )

    plot_access_fraction(
        access_data
    )

    plot_persistent_storage(
        access_data
    )

    plot_accumulation_structure(
        access_data
    )

    plot_arithmetic_vs_reads(
        access_data
    )

    print(
        "\nSaved all operator-equivalence and access-model plots to:"
    )

    print(
        f"  {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    run_all_plots()
