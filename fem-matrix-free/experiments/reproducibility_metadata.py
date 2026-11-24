
"""
Collect software, platform, solver, and experiment metadata.

Outputs
-------
results/reproducibility_metadata.json
results/reproducibility_metadata.md
results/pip_freeze.txt

The metadata record includes:

- operating system and processor information;
- Python, NumPy, pandas, Firedrake, PETSc, and mpi4py versions;
- thread-related environment variables;
- solver parameters;
- mesh and functional parameters;
- Git commit and working-tree status;
- a pip package snapshot.

Run from the project root with:

    python experiments/reproducibility_metadata.py

or from a notebook with:

    %run experiments/reproducibility_metadata.py
"""

from datetime import datetime, timezone
from pathlib import Path

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys

import numpy as np
import pandas as pd

import firedrake
from firedrake.petsc import PETSc


JSON_OUTPUT = Path(
    "results/reproducibility_metadata.json"
)


MARKDOWN_OUTPUT = Path(
    "results/reproducibility_metadata.md"
)


PIP_FREEZE_OUTPUT = Path(
    "results/pip_freeze.txt"
)


ASSEMBLED_SOLVER_PARAMETERS = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
}


FIREDRAKE_MATFREE_SOLVER_PARAMETERS = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-12,
    "ksp_atol": 1.0e-14,
    "ksp_max_it": 1000,
    "pc_type": "jacobi",
}


LOCAL_OPERATOR_SOLVER_PARAMETERS = {
    "operator_type": "python",
    "preconditioner_matrix_type": "aij",
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-12,
    "ksp_atol": 1.0e-14,
    "ksp_max_it": 1000,
    "pc_type": "jacobi",
}


EXPERIMENT_PARAMETERS = {
    "problem": "Manufactured Poisson problem",
    "domain": "(0, 1)^2",
    "exact_solution": (
        "sin(pi*x) * sin(pi*y)"
    ),
    "forcing": (
        "2*pi^2*sin(pi*x)*sin(pi*y)"
    ),
    "element": (
        "Continuous Lagrange P1 triangle"
    ),
    "boundary_condition": (
        "Homogeneous Dirichlet"
    ),
    "execution_mode": "Serial",
    "refinement_levels": [
        2,
        3,
        4,
        5,
    ],
    "operator_random_seeds": list(
        range(
            2026,
            2036,
        )
    ),
    "subdomain_bounds": [
        0.25,
        0.75,
        0.25,
        0.75,
    ],
    "gaussian_center": [
        0.70,
        0.35,
    ],
    "gaussian_sigma": 0.10,
    "weighted_local_quadrature_order": 12,
    "weighted_reference_quadrature_degree": 30,
    "floating_point_type": "float64",
    "operator_equivalence_target": 1.0e-11,
    "solution_agreement_target": 1.0e-9,
    "residual_target": 1.0e-9,
}


THREAD_ENVIRONMENT_VARIABLES = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PETSC_OPTIONS",
]


def command_output(
    command,
    cwd=None,
):
    """
    Return command output without failing metadata collection.
    """

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )

        return completed.stdout.strip()

    except Exception:
        return None


def package_version(
    package_name,
    imported_package=None,
):
    """
    Return an installed package version when available.
    """

    if imported_package is not None:
        imported_version = getattr(
            imported_package,
            "__version__",
            None,
        )

        if imported_version is not None:
            return str(
                imported_version
            )

    try:
        return importlib.metadata.version(
            package_name
        )

    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def cpu_model():
    """
    Return the Linux CPU model when available.
    """

    cpu_information = Path(
        "/proc/cpuinfo"
    )

    if not cpu_information.exists():
        return platform.processor()

    try:
        for line in cpu_information.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.lower().startswith(
                "model name"
            ):
                return line.split(
                    ":",
                    maxsplit=1,
                )[1].strip()

    except Exception:
        pass

    return platform.processor()


def available_memory_bytes():
    """
    Return detected physical memory on POSIX systems.
    """

    try:
        page_size = os.sysconf(
            "SC_PAGE_SIZE"
        )

        physical_pages = os.sysconf(
            "SC_PHYS_PAGES"
        )

        return int(
            page_size
            * physical_pages
        )

    except (
        AttributeError,
        OSError,
        ValueError,
    ):
        return None


def collect_git_metadata():
    """
    Collect Git commit and working-tree information.
    """

    repository_root = Path.cwd()

    commit = command_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=repository_root,
    )

    branch = command_output(
        [
            "git",
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ],
        cwd=repository_root,
    )

    status = command_output(
        [
            "git",
            "status",
            "--short",
        ],
        cwd=repository_root,
    )

    remote = command_output(
        [
            "git",
            "remote",
            "get-url",
            "origin",
        ],
        cwd=repository_root,
    )

    return {
        "commit": commit,
        "branch": branch,
        "remote": remote,
        "working_tree_clean": (
            status == ""
            if status is not None
            else None
        ),
        "working_tree_status": status,
    }


def collect_petsc_metadata():
    """
    Collect PETSc version and configuration information.
    """

    try:
        version = PETSc.Sys.getVersion()
    except Exception:
        version = None

    try:
        version_info = PETSc.Sys.getVersionInfo()
    except Exception:
        version_info = None

    try:
        scalar_type = str(
            PETSc.ScalarType
        )
    except Exception:
        scalar_type = None

    try:
        integer_type = str(
            PETSc.IntType
        )
    except Exception:
        integer_type = None

    return {
        "version": version,
        "version_info": version_info,
        "scalar_type": scalar_type,
        "integer_type": integer_type,
    }


def collect_environment_metadata():
    """
    Collect thread and PETSc environment variables.
    """

    return {
        variable: os.environ.get(
            variable
        )
        for variable
        in THREAD_ENVIRONMENT_VARIABLES
    }


def collect_metadata():
    """
    Construct the complete reproducibility record.
    """

    metadata = {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "project": {
            "working_directory": (
                str(
                    Path.cwd().resolve()
                )
            ),
            "python_executable": (
                sys.executable
            ),
        },

        "platform": {
            "hostname": (
                socket.gethostname()
            ),
            "platform": (
                platform.platform()
            ),
            "system": (
                platform.system()
            ),
            "release": (
                platform.release()
            ),
            "machine": (
                platform.machine()
            ),
            "processor": (
                platform.processor()
            ),
            "cpu_model": (
                cpu_model()
            ),
            "logical_cpu_count": (
                os.cpu_count()
            ),
            "physical_memory_bytes": (
                available_memory_bytes()
            ),
        },

        "python": {
            "version": (
                sys.version
            ),
            "version_info": {
                "major": (
                    sys.version_info.major
                ),
                "minor": (
                    sys.version_info.minor
                ),
                "micro": (
                    sys.version_info.micro
                ),
            },
        },

        "software": {
            "firedrake_version": (
                package_version(
                    "firedrake",
                    firedrake,
                )
            ),
            "numpy_version": (
                np.__version__
            ),
            "pandas_version": (
                pd.__version__
            ),
            "matplotlib_version": (
                package_version(
                    "matplotlib"
                )
            ),
            "mpi4py_version": (
                package_version(
                    "mpi4py"
                )
            ),
            "petsc4py_version": (
                package_version(
                    "petsc4py"
                )
            ),
            "pyop2_version": (
                package_version(
                    "pyop2"
                )
            ),
            "ufl_version": (
                package_version(
                    "fenics-ufl"
                )
            ),
        },

        "petsc": (
            collect_petsc_metadata()
        ),

        "environment": (
            collect_environment_metadata()
        ),

        "solver_configurations": {
            "assembled": (
                ASSEMBLED_SOLVER_PARAMETERS
            ),
            "firedrake_matfree": (
                FIREDRAKE_MATFREE_SOLVER_PARAMETERS
            ),
            "local_precomputed": (
                {
                    **LOCAL_OPERATOR_SOLVER_PARAMETERS,
                    "precompute_matrices": True,
                }
            ),
            "local_direct": (
                {
                    **LOCAL_OPERATOR_SOLVER_PARAMETERS,
                    "precompute_matrices": False,
                }
            ),
        },

        "experiment": (
            EXPERIMENT_PARAMETERS
        ),

        "version_control": (
            collect_git_metadata()
        ),
    }

    return metadata


def write_pip_freeze():
    """
    Write a pip package snapshot.

    This file supplements the metadata record. A pip snapshot alone
    may not reproduce a complete Firedrake installation.
    """

    output = command_output(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ]
    )

    if output is None:
        output = (
            "pip freeze could not be collected."
        )

    PIP_FREEZE_OUTPUT.write_text(
        output + "\n",
        encoding="utf-8",
    )


def markdown_value(
    value,
):
    """
    Format metadata values for Markdown.
    """

    if value is None:
        return "Not available"

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
        ),
    ):
        return (
            "`"
            + json.dumps(
                value,
                default=str,
            )
            + "`"
        )

    return str(
        value
    )


def append_section(
    lines,
    title,
    mapping,
):
    """
    Append a Markdown metadata section.
    """

    lines.extend(
        [
            "",
            f"## {title}",
            "",
        ]
    )

    for key, value in mapping.items():
        lines.append(
            f"- **{key}:** "
            f"{markdown_value(value)}"
        )


def write_markdown(
    metadata,
):
    """
    Write a human-readable metadata summary.
    """

    lines = [
        "# Reproducibility Metadata",
        "",
        (
            "This file records the software, platform, solver, "
            "and experiment configuration used for the finite-element "
            "verification study."
        ),
    ]

    append_section(
        lines,
        "Project",
        metadata["project"],
    )

    append_section(
        lines,
        "Platform",
        metadata["platform"],
    )

    append_section(
        lines,
        "Python",
        metadata["python"],
    )

    append_section(
        lines,
        "Software",
        metadata["software"],
    )

    append_section(
        lines,
        "PETSc",
        metadata["petsc"],
    )

    append_section(
        lines,
        "Environment",
        metadata["environment"],
    )

    append_section(
        lines,
        "Experiment",
        metadata["experiment"],
    )

    lines.extend(
        [
            "",
            "## Solver configurations",
            "",
        ]
    )

    for method, configuration in metadata[
        "solver_configurations"
    ].items():
        lines.append(
            f"- **{method}:** "
            f"`{json.dumps(configuration)}`"
        )

    append_section(
        lines,
        "Version control",
        metadata["version_control"],
    )

    lines.extend(
        [
            "",
            "## Environment snapshot",
            "",
            (
                f"A package snapshot is stored in "
                f"`{PIP_FREEZE_OUTPUT}`."
            ),
            "",
            (
                "A pip snapshot may not fully reproduce a Firedrake "
                "installation. The original Firedrake environment, "
                "container image, or installation procedure should "
                "also be retained when available."
            ),
        ]
    )

    MARKDOWN_OUTPUT.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def write_outputs(
    metadata,
):
    """
    Write JSON, Markdown, and package-snapshot outputs.
    """

    JSON_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_OUTPUT.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    write_markdown(
        metadata
    )

    write_pip_freeze()

    print(
        "Saved reproducibility outputs:"
    )

    for path in [
        JSON_OUTPUT,
        MARKDOWN_OUTPUT,
        PIP_FREEZE_OUTPUT,
    ]:
        print(
            f"  {path}"
        )


def run():
    """
    Collect and write reproducibility metadata.
    """

    metadata = collect_metadata()

    write_outputs(
        metadata
    )

    return metadata


if __name__ == "__main__":
    run()
