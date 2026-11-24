
"""
Phase 3: Logical local-access and persistent-storage model.

This experiment counts the mathematical data accessed by the current
operator and functional representations.

The counts are logical representation counts. They are not hardware
measurements and do not represent:

- cache or DRAM traffic;
- SIMD instructions;
- fused multiply-add instructions;
- NumPy temporary arrays;
- Python object overhead;
- PETSc internal storage;
- Firedrake mesh storage;
- MPI communication.

Storage conventions
-------------------
1. Floating-point storage includes persistent operation-specific
   local data.

2. Index storage includes operation-specific connectivity and,
   where applicable, selected-cell indices.

3. Global input and output vectors are excluded.

4. Shared mesh and Firedrake function-space storage are excluded.

5. The logical arithmetic counts describe the written formulas,
   not compiled hardware instruction counts.

Outputs
-------
results/access_model.csv

Run from the project root with:

    python experiments/access_model.py

or from a notebook with:

    %run experiments/access_model.py
"""

from pathlib import Path
import csv

from src.problem import create_problem
from src.local_operator import LocalP1Operator

from src.global_functional import (
    create_global_functional_data,
)

from src.subdomain_functional import (
    create_subdomain_functional_data,
)

from src.weighted_functional import (
    create_weighted_functional_data,
)


LEVELS = [
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


WEIGHT_QUADRATURE_ORDER = 12


FLOAT_BYTES = 8
INDEX_BYTES = 8


def _validate_nonnegative_integer(
    value,
    name,
):
    """
    Validate a nonnegative integer model parameter.
    """

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer."
        )

    integer_value = int(
        value
    )

    if (
        integer_value < 0
        or integer_value != value
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer."
        )

    return integer_value


def make_row(
    level,
    operator,
    operation,
    accessed_cells,
    storage_cells,
    integer_reads_per_accessed_cell,
    scalar_reads_per_accessed_cell,
    scalar_outputs_per_accessed_cell,
    vector_scatter_adds_per_accessed_cell,
    scalar_reduction_contributions_per_accessed_cell,
    persistent_float_scalars_per_stored_cell,
    persistent_integer_indices_per_stored_cell,
    logical_multiplications_per_accessed_cell,
    logical_additions_per_accessed_cell,
):
    """
    Construct one row of the logical access model.

    Parameters
    ----------
    accessed_cells:
        Number of cells visited during one evaluation.

    storage_cells:
        Number of cells carrying operation-specific persistent data.

    integer_reads_per_accessed_cell:
        Logical connectivity/index reads during one cell evaluation.

    scalar_reads_per_accessed_cell:
        Logical floating-point reads during one cell evaluation.
        These include gathered coefficients and operation-specific
        local data.

    scalar_outputs_per_accessed_cell:
        Number of local scalar values produced by one cell evaluation.

    vector_scatter_adds_per_accessed_cell:
        Number of updates into a global output vector.

    scalar_reduction_contributions_per_accessed_cell:
        Number of local scalar contributions entering a global
        reduction.

    persistent_float_scalars_per_stored_cell:
        Operation-specific persistent floating-point values per
        storage cell.

    persistent_integer_indices_per_stored_cell:
        Operation-specific persistent integer indices per storage cell.

    logical_multiplications_per_accessed_cell:
        Mathematical multiplication count for the represented formula.

    logical_additions_per_accessed_cell:
        Mathematical addition count for the represented formula.
    """

    total_cells = int(
        operator.num_cells
    )

    accessed_cells = _validate_nonnegative_integer(
        accessed_cells,
        "accessed_cells",
    )

    storage_cells = _validate_nonnegative_integer(
        storage_cells,
        "storage_cells",
    )

    if accessed_cells > total_cells:
        raise ValueError(
            "accessed_cells cannot exceed total_cells."
        )

    if storage_cells > total_cells:
        raise ValueError(
            "storage_cells cannot exceed total_cells."
        )

    count_parameters = {
        "integer_reads_per_accessed_cell":
            integer_reads_per_accessed_cell,

        "scalar_reads_per_accessed_cell":
            scalar_reads_per_accessed_cell,

        "scalar_outputs_per_accessed_cell":
            scalar_outputs_per_accessed_cell,

        "vector_scatter_adds_per_accessed_cell":
            vector_scatter_adds_per_accessed_cell,

        "scalar_reduction_contributions_per_accessed_cell":
            scalar_reduction_contributions_per_accessed_cell,

        "persistent_float_scalars_per_stored_cell":
            persistent_float_scalars_per_stored_cell,

        "persistent_integer_indices_per_stored_cell":
            persistent_integer_indices_per_stored_cell,

        "logical_multiplications_per_accessed_cell":
            logical_multiplications_per_accessed_cell,

        "logical_additions_per_accessed_cell":
            logical_additions_per_accessed_cell,
    }

    validated = {
        name: _validate_nonnegative_integer(
            value,
            name,
        )
        for name, value in count_parameters.items()
    }

    persistent_float_storage_bytes = (
        storage_cells
        * validated[
            "persistent_float_scalars_per_stored_cell"
        ]
        * FLOAT_BYTES
    )

    persistent_index_storage_bytes = (
        storage_cells
        * validated[
            "persistent_integer_indices_per_stored_cell"
        ]
        * INDEX_BYTES
    )

    return {
        "level": int(level),
        "h": float(
            operator.problem.h
        ),
        "ndofs": int(
            operator.num_dofs
        ),
        "total_cells": total_cells,
        "operation": operation,

        "accessed_cells": accessed_cells,
        "access_fraction": (
            accessed_cells
            / total_cells
        ),

        "storage_cells": storage_cells,
        "storage_fraction": (
            storage_cells
            / total_cells
        ),

        "integer_reads_per_accessed_cell": validated[
            "integer_reads_per_accessed_cell"
        ],

        "scalar_reads_per_accessed_cell": validated[
            "scalar_reads_per_accessed_cell"
        ],

        "scalar_outputs_per_accessed_cell": validated[
            "scalar_outputs_per_accessed_cell"
        ],

        "vector_scatter_adds_per_accessed_cell": validated[
            "vector_scatter_adds_per_accessed_cell"
        ],

        "scalar_reduction_contributions_per_accessed_cell": validated[
            "scalar_reduction_contributions_per_accessed_cell"
        ],

        "persistent_float_scalars_per_stored_cell": validated[
            "persistent_float_scalars_per_stored_cell"
        ],

        "persistent_integer_indices_per_stored_cell": validated[
            "persistent_integer_indices_per_stored_cell"
        ],

        "logical_multiplications_per_accessed_cell": validated[
            "logical_multiplications_per_accessed_cell"
        ],

        "logical_additions_per_accessed_cell": validated[
            "logical_additions_per_accessed_cell"
        ],

        "total_integer_reads": (
            accessed_cells
            * validated[
                "integer_reads_per_accessed_cell"
            ]
        ),

        "total_scalar_reads": (
            accessed_cells
            * validated[
                "scalar_reads_per_accessed_cell"
            ]
        ),

        "total_scalar_outputs": (
            accessed_cells
            * validated[
                "scalar_outputs_per_accessed_cell"
            ]
        ),

        "total_vector_scatter_adds": (
            accessed_cells
            * validated[
                "vector_scatter_adds_per_accessed_cell"
            ]
        ),

        "total_scalar_reduction_contributions": (
            accessed_cells
            * validated[
                "scalar_reduction_contributions_per_accessed_cell"
            ]
        ),

        "total_logical_multiplications": (
            accessed_cells
            * validated[
                "logical_multiplications_per_accessed_cell"
            ]
        ),

        "total_logical_additions": (
            accessed_cells
            * validated[
                "logical_additions_per_accessed_cell"
            ]
        ),

        "total_logical_arithmetic_operations": (
            accessed_cells
            * (
                validated[
                    "logical_multiplications_per_accessed_cell"
                ]
                + validated[
                    "logical_additions_per_accessed_cell"
                ]
            )
        ),

        "persistent_float_storage_bytes": (
            persistent_float_storage_bytes
        ),

        "persistent_index_storage_bytes": (
            persistent_index_storage_bytes
        ),

        "persistent_total_local_storage_bytes": (
            persistent_float_storage_bytes
            + persistent_index_storage_bytes
        ),
    }


def run_level(
    level,
):
    """
    Generate the access-model rows for one refinement level.
    """

    problem = create_problem(
        refinement_level=level
    )

    # This object supplies verified geometry and cell counts.
    # The operation rows below describe both supported local modes.
    operator = LocalP1Operator(
        problem,
        precompute_matrices=False,
    )

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
                WEIGHT_QUADRATURE_ORDER
            ),
        )
    )

    rows = []

    # ==================================================
    # Precomputed element-matrix operator
    # ==================================================
    #
    # Reads:
    #   3 gathered input coefficients
    #   9 entries of A_T
    #
    # Persistent local data:
    #   9 floating-point matrix entries
    #   3 cell-to-DoF indices
    #
    # Arithmetic:
    #   3 x 3 matrix-vector multiplication
    #   9 multiplications and 6 additions
    #
    # Accumulation:
    #   3 vector scatter-adds

    rows.append(
        make_row(
            level=level,
            operator=operator,
            operation=(
                "operator_precomputed"
            ),

            accessed_cells=(
                operator.num_cells
            ),

            storage_cells=(
                operator.num_cells
            ),

            integer_reads_per_accessed_cell=3,

            scalar_reads_per_accessed_cell=12,

            scalar_outputs_per_accessed_cell=3,

            vector_scatter_adds_per_accessed_cell=3,

            scalar_reduction_contributions_per_accessed_cell=0,

            persistent_float_scalars_per_stored_cell=9,

            persistent_integer_indices_per_stored_cell=3,

            logical_multiplications_per_accessed_cell=9,

            logical_additions_per_accessed_cell=6,
        )
    )

    # ==================================================
    # Direct gradient-factorized operator
    # ==================================================
    #
    # Reads:
    #   3 gathered input coefficients
    #   6 gradient components
    #   1 triangle area
    #
    # Persistent local data:
    #   6 gradient components
    #   1 triangle area
    #   3 cell-to-DoF indices
    #
    # Arithmetic:
    #   15 multiplications and 7 additions
    #
    # Accumulation:
    #   3 vector scatter-adds

    rows.append(
        make_row(
            level=level,
            operator=operator,
            operation=(
                "operator_direct"
            ),

            accessed_cells=(
                operator.num_cells
            ),

            storage_cells=(
                operator.num_cells
            ),

            integer_reads_per_accessed_cell=3,

            scalar_reads_per_accessed_cell=10,

            scalar_outputs_per_accessed_cell=3,

            vector_scatter_adds_per_accessed_cell=3,

            scalar_reduction_contributions_per_accessed_cell=0,

            persistent_float_scalars_per_stored_cell=7,

            persistent_integer_indices_per_stored_cell=3,

            logical_multiplications_per_accessed_cell=15,

            logical_additions_per_accessed_cell=7,
        )
    )

    # ==================================================
    # Global average
    # ==================================================
    #
    # Reads:
    #   3 gathered solution coefficients
    #   1 triangle area
    #
    # Persistent local data:
    #   1 triangle area
    #   3 cell-to-DoF indices
    #
    # Accumulation:
    #   1 scalar reduction contribution

    rows.append(
        make_row(
            level=level,
            operator=operator,
            operation=(
                "global_average"
            ),

            accessed_cells=(
                global_data.num_cells
            ),

            storage_cells=(
                global_data.num_cells
            ),

            integer_reads_per_accessed_cell=3,

            scalar_reads_per_accessed_cell=4,

            scalar_outputs_per_accessed_cell=1,

            vector_scatter_adds_per_accessed_cell=0,

            scalar_reduction_contributions_per_accessed_cell=1,

            persistent_float_scalars_per_stored_cell=1,

            persistent_integer_indices_per_stored_cell=3,

            logical_multiplications_per_accessed_cell=2,

            logical_additions_per_accessed_cell=2,
        )
    )

    # ==================================================
    # Mesh-aligned subdomain average
    # ==================================================
    #
    # The current representation stores copied connectivity and
    # areas only for selected cells.
    #
    # Persistent integer data:
    #   3 selected cell-to-DoF indices
    #   1 selected-cell index
    #
    # The selected-cell index is included as persistent storage.
    # It is not counted as an additional evaluation-stage read because
    # the selected data are already stored in compact selected arrays.

    rows.append(
        make_row(
            level=level,
            operator=operator,
            operation=(
                "subdomain_average"
            ),

            accessed_cells=(
                subdomain_data
                .num_selected_cells
            ),

            storage_cells=(
                subdomain_data
                .num_selected_cells
            ),

            integer_reads_per_accessed_cell=3,

            scalar_reads_per_accessed_cell=4,

            scalar_outputs_per_accessed_cell=1,

            vector_scatter_adds_per_accessed_cell=0,

            scalar_reduction_contributions_per_accessed_cell=1,

            persistent_float_scalars_per_stored_cell=1,

            persistent_integer_indices_per_stored_cell=4,

            logical_multiplications_per_accessed_cell=2,

            logical_additions_per_accessed_cell=2,
        )
    )

    # ==================================================
    # Gaussian-weighted average
    # ==================================================
    #
    # Evaluation reads:
    #   3 gathered solution coefficients
    #   3 precomputed local functional-vector entries
    #
    # Persistent local data:
    #   3 local functional-vector entries
    #   3 cell-to-DoF indices
    #
    # Quadrature work required to construct the local vectors is
    # setup cost and is not included in the evaluation-stage counts.

    rows.append(
        make_row(
            level=level,
            operator=operator,
            operation=(
                "weighted_average"
            ),

            accessed_cells=(
                weighted_data.num_cells
            ),

            storage_cells=(
                weighted_data.num_cells
            ),

            integer_reads_per_accessed_cell=3,

            scalar_reads_per_accessed_cell=6,

            scalar_outputs_per_accessed_cell=1,

            vector_scatter_adds_per_accessed_cell=0,

            scalar_reduction_contributions_per_accessed_cell=1,

            persistent_float_scalars_per_stored_cell=3,

            persistent_integer_indices_per_stored_cell=3,

            logical_multiplications_per_accessed_cell=3,

            logical_additions_per_accessed_cell=2,
        )
    )

    return rows


def print_summary(
    rows,
):
    """
    Print a compact access and storage summary.
    """

    print(
        "\nPhase 3: Logical local-access and storage model"
    )

    print(
        "=" * 151
    )

    header = (
        f"{'Level':>5} "
        f"{'DoFs':>8} "
        f"{'Cells':>8} "
        f"{'Operation':>23} "
        f"{'Accessed':>9} "
        f"{'Access frac':>11} "
        f"{'Scalar reads':>13} "
        f"{'Scatters':>10} "
        f"{'Reductions':>11} "
        f"{'Float bytes':>12} "
        f"{'Index bytes':>12} "
        f"{'Total bytes':>12}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for row in rows:
        print(
            f"{row['level']:5d} "
            f"{row['ndofs']:8d} "
            f"{row['total_cells']:8d} "
            f"{row['operation']:>23} "
            f"{row['accessed_cells']:9d} "
            f"{row['access_fraction']:11.3f} "
            f"{row['total_scalar_reads']:13d} "
            f"{row['total_vector_scatter_adds']:10d} "
            f"{row['total_scalar_reduction_contributions']:11d} "
            f"{row['persistent_float_storage_bytes']:12d} "
            f"{row['persistent_index_storage_bytes']:12d} "
            f"{row['persistent_total_local_storage_bytes']:12d}"
        )

    print(
        "=" * 151
    )

    print(
        "\nInterpretation:"
    )

    print(
        "  Scalar reads and arithmetic counts are logical "
        "representation counts."
    )

    print(
        "  Scatter-add counts describe global vector updates."
    )

    print(
        "  Reduction counts describe local scalar contributions."
    )

    print(
        "  Storage excludes global vectors, mesh objects, Python "
        "overhead, PETSc internals, and temporary arrays."
    )


def write_csv(
    rows,
    output_path=(
        "results/access_model.csv"
    ),
):
    """
    Write the access-model rows to CSV.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "No access-model rows were generated."
        )

    fieldnames = list(
        rows[0].keys()
    )

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
        f"\nSaved access-model results to: "
        f"{output_path}"
    )


def run_experiment(
    levels=None,
    output_path=(
        "results/access_model.csv"
    ),
):
    """
    Run the complete logical access-model experiment.
    """

    if levels is None:
        levels = LEVELS

    rows = []

    for level in levels:
        rows.extend(
            run_level(
                level
            )
        )

    print_summary(
        rows
    )

    write_csv(
        rows,
        output_path=output_path,
    )

    return rows


if __name__ == "__main__":
    run_experiment()
