
"""
Measure first-call time and process memory in isolated subprocesses.

Why subprocess isolation is required
------------------------------------
Firedrake, PETSc, Python, NumPy, and system allocators may retain memory
after objects are destroyed. Peak resident memory is also monotone
within a process. Measuring all methods in one notebook kernel would
therefore bias later methods.

This script launches a fresh subprocess for every:

    repetition x refinement level x solution method

Multiple repetitions are used because process startup, JIT-cache access,
filesystem caching, CPU scheduling, and allocator behavior can produce
substantial variation for these short experiments.

The worker records:

- problem-construction time;
- first complete method-call time;
- resident-set size before and after each stage;
- absolute peak resident-set size after the solve;
- solution-vector payload;
- actual NumPy array payload for local operators;
- available PETSc matrix metadata;
- solver convergence information.

Timing interpretation
---------------------
The field

    setup_and_first_solve_seconds

measures the first complete method call in a fresh process. Depending on
the method, this call may include:

- operator construction;
- right-hand-side assembly;
- matrix or preconditioner assembly;
- Firedrake/PETSc object construction;
- first-use kernel compilation;
- the linear solve.

It is not a pure steady-state solve time.

Memory interpretation
---------------------
Process RSS includes Python, Firedrake, PETSc, shared libraries,
allocator behavior, JIT infrastructure, and the finite-element objects.
RSS should therefore be treated as a coarse process diagnostic.

The NumPy payload is more precise for comparing the two custom local
representations. It counts retained NumPy array data but excludes Python
object overhead.

Outputs
-------
results/setup_and_memory.csv
    Raw subprocess measurements.

results/setup_and_memory.json
    Raw subprocess measurements in JSON format.

results/setup_and_memory_summary.csv
    Median and interquartile-range summaries.

results/setup_and_memory.md
    Human-readable summary and qualifications.

Run from the project root with:

    python experiments/setup_and_memory.py

or from a notebook with:

    %run experiments/setup_and_memory.py
"""

from pathlib import Path

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time


# ==================================================
# Project import path
# ==================================================


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


PROJECT_ROOT_STRING = str(
    PROJECT_ROOT
)


if (
    PROJECT_ROOT_STRING
    not in sys.path
):
    sys.path.insert(
        0,
        PROJECT_ROOT_STRING,
    )


import numpy as np
import pandas as pd


# ==================================================
# Output paths
# ==================================================


OUTPUT_CSV = Path(
    "results/setup_and_memory.csv"
)


OUTPUT_JSON = Path(
    "results/setup_and_memory.json"
)


SUMMARY_CSV = Path(
    "results/setup_and_memory_summary.csv"
)


OUTPUT_MARKDOWN = Path(
    "results/setup_and_memory.md"
)


# ==================================================
# Experiment configuration
# ==================================================


LEVELS = [
    2,
    3,
    4,
    5,
]


METHODS = [
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


# Three repetitions are sufficient for a first median/IQR diagnostic.
# Increase to five for final archived measurements.
DEFAULT_REPETITIONS = 3


RESULT_PREFIX = (
    "SETUP_MEMORY_RESULT="
)


# ==================================================
# Process-memory helpers
# ==================================================


def current_rss_bytes():
    """
    Return current resident-set size.

    On Linux, /proc/self/status is preferred. If /proc is unavailable,
    psutil is used when installed.
    """

    status_path = Path(
        "/proc/self/status"
    )

    if status_path.exists():
        try:
            for line in status_path.read_text(
                encoding="utf-8"
            ).splitlines():
                if line.startswith(
                    "VmRSS:"
                ):
                    kilobytes = int(
                        line.split()[1]
                    )

                    return int(
                        kilobytes
                        * 1024
                    )

        except Exception:
            pass

    try:
        import psutil

        process = psutil.Process(
            os.getpid()
        )

        return int(
            process.memory_info().rss
        )

    except Exception:
        return None


def peak_rss_bytes():
    """
    Return absolute peak process RSS.

    On Linux, ru_maxrss is reported in KiB. On macOS, ru_maxrss is
    commonly reported in bytes.

    The returned value includes imports, shared libraries, runtime
    initialization, and all computations performed by this worker.
    """

    maximum_rss = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    if platform.system() == "Darwin":
        return int(
            maximum_rss
        )

    return int(
        maximum_rss
        * 1024
    )


def safe_difference(
    final_value,
    initial_value,
):
    """
    Return final_value - initial_value when both values are available.
    """

    if (
        final_value is None
        or initial_value is None
    ):
        return None

    return int(
        final_value
        - initial_value
    )


def finite_positive_or_none(
    value,
):
    """
    Return a positive finite float, otherwise None.
    """

    if value is None:
        return None

    try:
        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None

    if (
        not np.isfinite(value)
        or value <= 0.0
    ):
        return None

    return value


# ==================================================
# Representation-payload helpers
# ==================================================


def array_payload_bytes(
    arrays,
):
    """
    Sum NumPy array payload sizes.

    Python object overhead, allocator metadata, and temporary arrays
    are excluded.
    """

    total = 0

    for array in arrays:
        if array is None:
            continue

        if hasattr(
            array,
            "nbytes",
        ):
            total += int(
                array.nbytes
            )

    return int(
        total
    )


def local_operator_payload(
    operator,
):
    """
    Return the actual retained NumPy payload of LocalP1Operator.

    This payload includes the arrays retained by the current Python
    implementation, including coordinate data and, in precomputed mode,
    the element stiffness matrices.
    """

    arrays = [
        getattr(
            operator,
            "dof_connectivity",
            None,
        ),
        getattr(
            operator,
            "boundary_dofs",
            None,
        ),
        getattr(
            operator,
            "dof_coordinates",
            None,
        ),
        getattr(
            operator,
            "cell_coordinates",
            None,
        ),
        getattr(
            operator,
            "areas",
            None,
        ),
        getattr(
            operator,
            "gradients",
            None,
        ),
        getattr(
            operator,
            "element_matrices",
            None,
        ),
    ]

    return array_payload_bytes(
        arrays
    )


def petsc_matrix_information(
    matrix,
):
    """
    Extract PETSc matrix metadata when the installed build exposes it.

    A zero or nonfinite memory value is treated as unavailable. An
    assembled matrix cannot physically require zero bytes.
    """

    information = {
        "petsc_matrix_memory_bytes":
            None,

        "petsc_nz_used":
            None,

        "petsc_nz_allocated":
            None,
    }

    if matrix is None:
        return information

    try:
        info = matrix.getInfo()

    except Exception:
        return information

    memory = finite_positive_or_none(
        info.get(
            "memory"
        )
    )

    nonzeros_used = finite_positive_or_none(
        info.get(
            "nz_used"
        )
    )

    nonzeros_allocated = finite_positive_or_none(
        info.get(
            "nz_allocated"
        )
    )

    information[
        "petsc_matrix_memory_bytes"
    ] = memory

    information[
        "petsc_nz_used"
    ] = nonzeros_used

    information[
        "petsc_nz_allocated"
    ] = nonzeros_allocated

    return information


# ==================================================
# Worker measurement
# ==================================================


def worker_measure(
    level,
    method,
    repetition,
):
    """
    Run one isolated level-method measurement.
    """

    # Import Firedrake-dependent project modules only in the isolated
    # worker. Import initialization is not included in the problem or
    # method timers but is included in process RSS and absolute peak RSS.
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
        LocalP1Operator,
        solve_local_operator,
    )

    if method not in METHODS:
        raise ValueError(
            f"Unknown method: {method}"
        )

    gc.collect()

    initial_rss = (
        current_rss_bytes()
    )

    # --------------------------------------------------
    # Shared problem construction
    # --------------------------------------------------

    problem_start = (
        time.perf_counter()
    )

    problem = create_problem(
        refinement_level=level
    )

    problem_stop = (
        time.perf_counter()
    )

    after_problem_rss = (
        current_rss_bytes()
    )

    # --------------------------------------------------
    # First complete method call
    # --------------------------------------------------

    method_start = (
        time.perf_counter()
    )

    petsc_information = (
        petsc_matrix_information(
            None
        )
    )

    local_numpy_payload_bytes = None

    iterations = None
    convergence_reason = None
    ksp_residual_norm = None
    assembled_relative_residual = None

    local_payload_mode = None

    if method == "assembled":
        system = assemble_system(
            problem
        )

        result = solve_assembled(
            problem,
            system=system,
            solver_parameters=(
                DIRECT_SOLVER_PARAMETERS
            ),
        )

        petsc_matrix = None

        try:
            petsc_matrix = (
                system.A.petscmat
            )

        except Exception:
            try:
                petsc_matrix = (
                    system.matrix.petscmat
                )

            except Exception:
                petsc_matrix = None

        petsc_information = (
            petsc_matrix_information(
                petsc_matrix
            )
        )

        iterations = getattr(
            result,
            "iterations",
            None,
        )

        convergence_reason = getattr(
            result,
            "converged_reason",
            None,
        )

        ksp_residual_norm = getattr(
            result,
            "residual_norm",
            None,
        )

        assembled_relative_residual = getattr(
            result,
            "relative_residual",
            None,
        )

        solution = result.solution

    elif method == "firedrake_matfree":
        result = solve_firedrake_matfree(
            problem
        )

        iterations = getattr(
            result,
            "iterations",
            None,
        )

        convergence_reason = getattr(
            result,
            "converged_reason",
            None,
        )

        ksp_residual_norm = getattr(
            result,
            "ksp_residual_norm",
            None,
        )

        solution = result.solution

    elif method == "local_precomputed":
        result = solve_local_operator(
            problem,
            precompute_matrices=True,
        )

        iterations = getattr(
            result,
            "iterations",
            None,
        )

        convergence_reason = getattr(
            result,
            "converged_reason",
            None,
        )

        ksp_residual_norm = getattr(
            result,
            "ksp_residual_norm",
            None,
        )

        solution = result.solution

        local_payload_mode = True

    elif method == "local_direct":
        result = solve_local_operator(
            problem,
            precompute_matrices=False,
        )

        iterations = getattr(
            result,
            "iterations",
            None,
        )

        convergence_reason = getattr(
            result,
            "converged_reason",
            None,
        )

        ksp_residual_norm = getattr(
            result,
            "ksp_residual_norm",
            None,
        )

        solution = result.solution

        local_payload_mode = False

    else:
        raise RuntimeError(
            "Unreachable method branch."
        )

    method_stop = (
        time.perf_counter()
    )

    after_solve_rss = (
        current_rss_bytes()
    )

    absolute_peak_after_solve = (
        peak_rss_bytes()
    )

    solution_payload_bytes = int(
        solution.dat.data_ro.nbytes
    )

    # --------------------------------------------------
    # Actual local NumPy payload
    # --------------------------------------------------
    #
    # solve_local_operator does not return its internal LocalP1Operator.
    # Construct a representative object after stopping the method timer.
    # This does not affect setup_and_first_solve_seconds or the recorded
    # solve-stage RSS and peak values.

    if method in (
        "local_precomputed",
        "local_direct",
    ):
        representative_operator = (
            LocalP1Operator(
                problem,
                precompute_matrices=(
                    local_payload_mode
                ),
            )
        )

        local_numpy_payload_bytes = (
            local_operator_payload(
                representative_operator
            )
        )

        del representative_operator

        gc.collect()

    record = {
        "repetition": int(
            repetition
        ),

        "level": int(
            level
        ),

        "h": float(
            problem.h
        ),

        "ndofs": int(
            problem.V.dim()
        ),

        "method": method,

        "method_label": (
            METHOD_LABELS[
                method
            ]
        ),

        "problem_setup_seconds": (
            problem_stop
            - problem_start
        ),

        "setup_and_first_solve_seconds": (
            method_stop
            - method_start
        ),

        "initial_rss_bytes": (
            initial_rss
        ),

        "after_problem_rss_bytes": (
            after_problem_rss
        ),

        "after_solve_rss_bytes": (
            after_solve_rss
        ),

        "problem_rss_delta_bytes": (
            safe_difference(
                after_problem_rss,
                initial_rss,
            )
        ),

        "method_rss_delta_bytes": (
            safe_difference(
                after_solve_rss,
                after_problem_rss,
            )
        ),

        "total_rss_delta_bytes": (
            safe_difference(
                after_solve_rss,
                initial_rss,
            )
        ),

        # This is an absolute process peak, not operator-only memory.
        "absolute_peak_rss_bytes": (
            absolute_peak_after_solve
        ),

        "solution_payload_bytes": (
            solution_payload_bytes
        ),

        "local_numpy_payload_bytes": (
            local_numpy_payload_bytes
        ),

        "iterations": (
            int(
                iterations
            )
            if iterations is not None
            else None
        ),

        "convergence_reason": (
            int(
                convergence_reason
            )
            if convergence_reason is not None
            else None
        ),

        "ksp_residual_norm": (
            float(
                ksp_residual_norm
            )
            if ksp_residual_norm is not None
            else None
        ),

        "assembled_relative_residual": (
            float(
                assembled_relative_residual
            )
            if assembled_relative_residual is not None
            else None
        ),

        **petsc_information,

        "python_version": (
            sys.version
        ),

        "platform": (
            platform.platform()
        ),
    }

    return record


def run_worker(
    level,
    method,
    repetition,
):
    """
    Execute one worker and emit one identifiable JSON result line.
    """

    record = worker_measure(
        level=level,
        method=method,
        repetition=repetition,
    )

    print(
        RESULT_PREFIX
        + json.dumps(
            record,
            default=str,
        )
    )


# ==================================================
# Parent subprocess orchestration
# ==================================================


def extract_worker_result(
    stdout,
):
    """
    Extract the JSON record from worker stdout.
    """

    matching_lines = [
        line
        for line in stdout.splitlines()
        if line.startswith(
            RESULT_PREFIX
        )
    ]

    if not matching_lines:
        raise RuntimeError(
            "The worker did not produce a measurement record.\n"
            f"Worker output:\n{stdout}"
        )

    payload = matching_lines[-1][
        len(
            RESULT_PREFIX
        ):
    ]

    return json.loads(
        payload
    )


def launch_worker(
    level,
    method,
    repetition,
):
    """
    Launch one isolated worker subprocess.

    The repository root is the subprocess working directory and is
    explicitly prepended to PYTHONPATH.
    """

    script_path = Path(
        __file__
    ).resolve()

    project_root = (
        script_path.parents[1]
    )

    command = [
        sys.executable,
        str(
            script_path
        ),
        "--worker",
        "--level",
        str(
            level
        ),
        "--method",
        method,
        "--repetition",
        str(
            repetition
        ),
    ]

    environment = (
        os.environ.copy()
    )

    existing_pythonpath = (
        environment.get(
            "PYTHONPATH",
            "",
        )
    )

    if existing_pythonpath:
        environment[
            "PYTHONPATH"
        ] = (
            str(
                project_root
            )
            + os.pathsep
            + existing_pythonpath
        )

    else:
        environment[
            "PYTHONPATH"
        ] = str(
            project_root
        )

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=project_root,
        env=environment,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Worker process failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Working directory: {project_root}\n"
            f"PYTHONPATH: "
            f"{environment.get('PYTHONPATH')}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return extract_worker_result(
        completed.stdout
    )


# ==================================================
# Aggregation
# ==================================================


def percentile_25(
    values,
):
    """
    Return the first quartile.
    """

    return float(
        np.nanpercentile(
            values,
            25.0,
        )
    )


def percentile_75(
    values,
):
    """
    Return the third quartile.
    """

    return float(
        np.nanpercentile(
            values,
            75.0,
        )
    )


def first_nonmissing(
    values,
):
    """
    Return the first nonmissing value.
    """

    valid = values.dropna()

    if valid.empty:
        return np.nan

    return valid.iloc[0]


def build_summary(
    data,
):
    """
    Build median and interquartile-range summaries.

    Payload and solver fields should normally be deterministic across
    repetitions. Timing and RSS fields are summarized statistically.
    """

    summary = (
        data.groupby(
            [
                "level",
                "h",
                "ndofs",
                "method",
                "method_label",
            ],
            as_index=False,
        )
        .agg(
            repetitions=(
                "repetition",
                "count",
            ),

            problem_setup_median_seconds=(
                "problem_setup_seconds",
                "median",
            ),

            problem_setup_q1_seconds=(
                "problem_setup_seconds",
                percentile_25,
            ),

            problem_setup_q3_seconds=(
                "problem_setup_seconds",
                percentile_75,
            ),

            setup_and_first_solve_median_seconds=(
                "setup_and_first_solve_seconds",
                "median",
            ),

            setup_and_first_solve_q1_seconds=(
                "setup_and_first_solve_seconds",
                percentile_25,
            ),

            setup_and_first_solve_q3_seconds=(
                "setup_and_first_solve_seconds",
                percentile_75,
            ),

            total_rss_delta_median_bytes=(
                "total_rss_delta_bytes",
                "median",
            ),

            total_rss_delta_q1_bytes=(
                "total_rss_delta_bytes",
                percentile_25,
            ),

            total_rss_delta_q3_bytes=(
                "total_rss_delta_bytes",
                percentile_75,
            ),

            absolute_peak_rss_median_bytes=(
                "absolute_peak_rss_bytes",
                "median",
            ),

            solution_payload_bytes=(
                "solution_payload_bytes",
                first_nonmissing,
            ),

            local_numpy_payload_bytes=(
                "local_numpy_payload_bytes",
                first_nonmissing,
            ),

            petsc_matrix_memory_bytes=(
                "petsc_matrix_memory_bytes",
                first_nonmissing,
            ),

            petsc_nz_used=(
                "petsc_nz_used",
                first_nonmissing,
            ),

            petsc_nz_allocated=(
                "petsc_nz_allocated",
                first_nonmissing,
            ),

            iterations=(
                "iterations",
                first_nonmissing,
            ),

            convergence_reason=(
                "convergence_reason",
                first_nonmissing,
            ),

            ksp_residual_norm=(
                "ksp_residual_norm",
                first_nonmissing,
            ),

            assembled_relative_residual=(
                "assembled_relative_residual",
                first_nonmissing,
            ),
        )
    )

    method_order = {
        method: index
        for index, method
        in enumerate(
            METHODS
        )
    }

    summary[
        "_method_order"
    ] = summary[
        "method"
    ].map(
        method_order
    )

    summary = summary.sort_values(
        [
            "level",
            "_method_order",
        ]
    ).drop(
        columns=[
            "_method_order",
        ]
    ).reset_index(
        drop=True
    )

    return summary


def add_local_payload_comparison(
    summary,
):
    """
    Add direct-versus-precomputed local payload diagnostics.

    For P1 triangles, the expected precomputed-minus-direct difference
    is nine float64 element-matrix entries per cell.
    """

    summary = (
        summary.copy()
    )

    summary[
        "expected_matrix_payload_difference_bytes"
    ] = np.nan

    summary[
        "observed_matrix_payload_difference_bytes"
    ] = np.nan

    summary[
        "payload_difference_matches_expected"
    ] = pd.NA

    for level in sorted(
        summary[
            "level"
        ].unique()
    ):
        direct_mask = (
            (
                summary["level"]
                == level
            )
            & (
                summary["method"]
                == "local_direct"
            )
        )

        precomputed_mask = (
            (
                summary["level"]
                == level
            )
            & (
                summary["method"]
                == "local_precomputed"
            )
        )

        if (
            not direct_mask.any()
            or not precomputed_mask.any()
        ):
            continue

        direct_payload = summary.loc[
            direct_mask,
            "local_numpy_payload_bytes",
        ].iloc[0]

        precomputed_payload = summary.loc[
            precomputed_mask,
            "local_numpy_payload_bytes",
        ].iloc[0]

        ndofs = int(
            summary.loc[
                direct_mask,
                "ndofs",
            ].iloc[0]
        )

        # For a UnitSquareMesh with N = sqrt(ndofs)-1 subdivisions,
        # the number of triangles is 2*N^2.
        subdivisions = int(
            round(
                np.sqrt(
                    ndofs
                )
                - 1
            )
        )

        num_cells = int(
            2
            * subdivisions**2
        )

        expected_difference = int(
            num_cells
            * 9
            * np.dtype(
                np.float64
            ).itemsize
        )

        if (
            pd.isna(
                direct_payload
            )
            or pd.isna(
                precomputed_payload
            )
        ):
            observed_difference = np.nan
            matches = pd.NA

        else:
            observed_difference = int(
                precomputed_payload
                - direct_payload
            )

            matches = bool(
                observed_difference
                == expected_difference
            )

        local_mask = (
            summary["level"]
            == level
        ) & summary[
            "method"
        ].isin(
            [
                "local_direct",
                "local_precomputed",
            ]
        )

        summary.loc[
            local_mask,
            "expected_matrix_payload_difference_bytes",
        ] = expected_difference

        summary.loc[
            local_mask,
            "observed_matrix_payload_difference_bytes",
        ] = observed_difference

        summary.loc[
            local_mask,
            "payload_difference_matches_expected",
        ] = matches

    return summary


# ==================================================
# Formatting and reporting
# ==================================================


def format_bytes(
    value,
):
    """
    Format bytes using binary units.

    Missing, NaN, infinite, and negative values are displayed as an
    en dash.
    """

    if value is None:
        return "–"

    try:
        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return "–"

    if (
        not np.isfinite(value)
        or value < 0.0
    ):
        return "–"

    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
    ]

    unit_index = 0

    while (
        abs(value) >= 1024.0
        and unit_index
        < len(units) - 1
    ):
        value /= 1024.0
        unit_index += 1

    return (
        f"{value:.2f} "
        f"{units[unit_index]}"
    )


def print_summary(
    summary,
):
    """
    Print median setup-time and memory diagnostics.
    """

    print(
        "\nIsolated setup and memory summary"
    )

    print(
        "=" * 136
    )

    header = (
        f"{'Level':>5} "
        f"{'DoFs':>8} "
        f"{'Method':>22} "
        f"{'Problem median':>15} "
        f"{'Setup+solve':>15} "
        f"{'RSS delta':>12} "
        f"{'Absolute peak':>13} "
        f"{'NumPy payload':>14} "
        f"{'PETSc memory':>13}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for _, row in summary.iterrows():
        print(
            f"{int(row['level']):5d} "
            f"{int(row['ndofs']):8d} "
            f"{row['method_label']:>22} "
            f"{row['problem_setup_median_seconds']:15.6e} "
            f"{row['setup_and_first_solve_median_seconds']:15.6e} "
            f"{format_bytes(row['total_rss_delta_median_bytes']):>12} "
            f"{format_bytes(row['absolute_peak_rss_median_bytes']):>13} "
            f"{format_bytes(row['local_numpy_payload_bytes']):>14} "
            f"{format_bytes(row['petsc_matrix_memory_bytes']):>13}"
        )

    print(
        "=" * 136
    )

    print(
        "\nMeasurement qualifications:"
    )

    print(
        "  Every repetition ran in a fresh subprocess."
    )

    print(
        "  Timing columns report medians; the summary CSV also "
        "contains first and third quartiles."
    )

    print(
        "  Problem-setup variations among methods reflect worker "
        "startup and runtime noise, not method-specific setup."
    )

    print(
        "  RSS includes the complete process and should not be used "
        "for fine representation-level comparisons."
    )

    print(
        "  Absolute peak RSS includes module imports and runtime "
        "initialization."
    )

    print(
        "  NumPy payload is the preferred empirical comparison for "
        "the two local representations."
    )

    print(
        "  PETSc matrix memory is shown only when PETSc provides a "
        "positive finite value."
    )


def print_local_payload_validation(
    summary,
):
    """
    Print direct-versus-precomputed NumPy payload validation.
    """

    local_rows = summary[
        summary["method"].isin(
            [
                "local_direct",
                "local_precomputed",
            ]
        )
    ]

    print(
        "\nLocal NumPy payload validation"
    )

    print(
        "=" * 91
    )

    header = (
        f"{'Level':>5} "
        f"{'Direct':>14} "
        f"{'Precomputed':>14} "
        f"{'Observed diff':>14} "
        f"{'Expected diff':>14} "
        f"{'Match':>8}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for level in sorted(
        local_rows[
            "level"
        ].unique()
    ):
        level_rows = local_rows[
            local_rows["level"]
            == level
        ]

        direct = level_rows[
            level_rows["method"]
            == "local_direct"
        ].iloc[0]

        precomputed = level_rows[
            level_rows["method"]
            == "local_precomputed"
        ].iloc[0]

        observed = direct[
            "observed_matrix_payload_difference_bytes"
        ]

        expected = direct[
            "expected_matrix_payload_difference_bytes"
        ]

        matches = direct[
            "payload_difference_matches_expected"
        ]

        print(
            f"{int(level):5d} "
            f"{format_bytes(direct['local_numpy_payload_bytes']):>14} "
            f"{format_bytes(precomputed['local_numpy_payload_bytes']):>14} "
            f"{format_bytes(observed):>14} "
            f"{format_bytes(expected):>14} "
            f"{str(matches):>8}"
        )

    print(
        "=" * 91
    )


def markdown_summary_table(
    summary,
):
    """
    Construct a Markdown summary table.
    """

    lines = [
        (
            "| Level | DoFs | Method | Repetitions | "
            "Problem setup median (s) | Setup + first solve median (s) | "
            "RSS delta median | Absolute peak RSS | "
            "Local NumPy payload | PETSc matrix memory |"
        ),
        (
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for _, row in summary.iterrows():
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
                        int(
                            row["repetitions"]
                        )
                    ),
                    (
                        f"{row['problem_setup_median_seconds']:.6e}"
                    ),
                    (
                        f"{row['setup_and_first_solve_median_seconds']:.6e}"
                    ),
                    format_bytes(
                        row[
                            "total_rss_delta_median_bytes"
                        ]
                    ),
                    format_bytes(
                        row[
                            "absolute_peak_rss_median_bytes"
                        ]
                    ),
                    format_bytes(
                        row[
                            "local_numpy_payload_bytes"
                        ]
                    ),
                    format_bytes(
                        row[
                            "petsc_matrix_memory_bytes"
                        ]
                    ),
                ]
            )
            + " |"
        )

    return "\n".join(
        lines
    )


def markdown_payload_table(
    summary,
):
    """
    Construct a Markdown local-payload validation table.
    """

    lines = [
        (
            "| Level | Cells | Direct payload | Precomputed payload | "
            "Observed difference | Expected matrix difference | Match |"
        ),
        (
            "|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for level in sorted(
        summary[
            "level"
        ].unique()
    ):
        direct_rows = summary[
            (
                summary["level"]
                == level
            )
            & (
                summary["method"]
                == "local_direct"
            )
        ]

        precomputed_rows = summary[
            (
                summary["level"]
                == level
            )
            & (
                summary["method"]
                == "local_precomputed"
            )
        ]

        if (
            direct_rows.empty
            or precomputed_rows.empty
        ):
            continue

        direct = direct_rows.iloc[0]
        precomputed = (
            precomputed_rows.iloc[0]
        )

        subdivisions = int(
            round(
                np.sqrt(
                    direct["ndofs"]
                )
                - 1
            )
        )

        num_cells = int(
            2
            * subdivisions**2
        )

        lines.append(
            "| "
            + " | ".join(
                [
                    str(
                        int(
                            level
                        )
                    ),
                    str(
                        num_cells
                    ),
                    format_bytes(
                        direct[
                            "local_numpy_payload_bytes"
                        ]
                    ),
                    format_bytes(
                        precomputed[
                            "local_numpy_payload_bytes"
                        ]
                    ),
                    format_bytes(
                        direct[
                            "observed_matrix_payload_difference_bytes"
                        ]
                    ),
                    format_bytes(
                        direct[
                            "expected_matrix_payload_difference_bytes"
                        ]
                    ),
                    str(
                        direct[
                            "payload_difference_matches_expected"
                        ]
                    ),
                ]
            )
            + " |"
        )

    return "\n".join(
        lines
    )


# ==================================================
# Output generation
# ==================================================


def write_outputs(
    records,
):
    """
    Write raw and summarized diagnostics.
    """

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.DataFrame(
        records
    )

    method_order = {
        method: index
        for index, method
        in enumerate(
            METHODS
        )
    }

    data[
        "_method_order"
    ] = data[
        "method"
    ].map(
        method_order
    )

    data = data.sort_values(
        [
            "level",
            "_method_order",
            "repetition",
        ]
    ).drop(
        columns=[
            "_method_order",
        ]
    ).reset_index(
        drop=True
    )

    summary = build_summary(
        data
    )

    summary = add_local_payload_comparison(
        summary
    )

    data.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            data.to_dict(
                orient="records"
            ),
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_table = (
        markdown_summary_table(
            summary
        )
    )

    payload_table = (
        markdown_payload_table(
            summary
        )
    )

    markdown_content = (
        "# Isolated Setup and Memory Diagnostics\n\n"
        "Every repetition and level-method combination was executed "
        "in a fresh subprocess to prevent retained Python, Firedrake, "
        "PETSc, NumPy, and allocator state from contaminating later "
        "measurements.\n\n"
        "## Summary\n\n"
        f"{summary_table}\n\n"
        "## Local representation payload validation\n\n"
        "The expected difference between the precomputed and direct "
        "local payloads is nine double-precision element-matrix entries "
        "per triangle:\n\n"
        "$$\n"
        "N_T \\times 9 \\times 8\\text{ bytes}.\n"
        "$$\n\n"
        f"{payload_table}\n\n"
        "## Important limitations\n\n"
        "- `setup_and_first_solve_seconds` is the first complete method "
        "call. It may include construction, assembly, JIT compilation, "
        "preconditioner creation, and the linear solve.\n"
        "- Timing medians summarize fresh-process repetitions. The "
        "summary CSV also records the first and third quartiles.\n"
        "- Problem setup is mathematically identical across methods. "
        "Differences in measured problem-setup time therefore represent "
        "worker startup, caching, scheduling, and runtime variation.\n"
        "- RSS measures the complete process, including loaded "
        "libraries and allocator behavior. RSS is too coarse for fine "
        "operator-storage comparisons at the tested problem sizes.\n"
        "- Absolute peak RSS includes imports and runtime initialization "
        "and is not operator-only memory.\n"
        "- Local NumPy payload counts retained array data only and "
        "excludes Python object overhead.\n"
        "- PETSc matrix memory is reported only when PETSc provides a "
        "positive finite value. Zero and nonfinite values are treated "
        "as unavailable.\n"
        "- These diagnostics are not a controlled performance "
        "benchmark.\n"
    )

    OUTPUT_MARKDOWN.write_text(
        markdown_content,
        encoding="utf-8",
    )

    print_summary(
        summary
    )

    print_local_payload_validation(
        summary
    )

    print(
        "\nSaved setup-and-memory outputs:"
    )

    for path in [
        OUTPUT_CSV,
        OUTPUT_JSON,
        SUMMARY_CSV,
        OUTPUT_MARKDOWN,
    ]:
        print(
            f"  {path}"
        )

    return (
        data,
        summary,
    )


def run_parent(
    levels=None,
    methods=None,
    repetitions=(
        DEFAULT_REPETITIONS
    ),
):
    """
    Launch all isolated measurements.
    """

    if levels is None:
        levels = LEVELS

    if methods is None:
        methods = METHODS

    if (
        isinstance(
            repetitions,
            bool,
        )
        or repetitions <= 0
    ):
        raise ValueError(
            "repetitions must be a positive integer."
        )

    records = []

    print(
        "Running isolated setup and memory measurements"
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"Repetitions per level-method pair: "
        f"{repetitions}"
    )

    for repetition in range(
        1,
        repetitions + 1,
    ):
        print(
            f"\nRepetition {repetition}"
        )

        for level in levels:
            for method in methods:
                print(
                    f"  Level {level}, "
                    f"method {method}..."
                )

                record = launch_worker(
                    level=level,
                    method=method,
                    repetition=repetition,
                )

                records.append(
                    record
                )

    return write_outputs(
        records
    )


# ==================================================
# Command-line interface
# ==================================================


def parse_arguments():
    """
    Parse parent and worker arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Measure FEM first-call time and process memory."
        )
    )

    parser.add_argument(
        "--worker",
        action="store_true",
        help=(
            "Run one isolated worker measurement."
        ),
    )

    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help=(
            "Refinement level for worker mode."
        ),
    )

    parser.add_argument(
        "--method",
        choices=METHODS,
        default=None,
        help=(
            "Solution method for worker mode."
        ),
    )

    parser.add_argument(
        "--repetition",
        type=int,
        default=1,
        help=(
            "Repetition identifier for worker mode."
        ),
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=(
            "Fresh subprocess repetitions per level-method pair "
            "in parent mode."
        ),
    )

    return parser.parse_args()


def main():
    """
    Execute parent or worker mode.
    """

    arguments = parse_arguments()

    if arguments.worker:
        if arguments.level is None:
            raise ValueError(
                "--level is required in worker mode."
            )

        if arguments.method is None:
            raise ValueError(
                "--method is required in worker mode."
            )

        run_worker(
            level=arguments.level,
            method=arguments.method,
            repetition=(
                arguments.repetition
            ),
        )

    else:
        run_parent(
            repetitions=(
                arguments.repetitions
            )
        )


if __name__ == "__main__":
    main()
