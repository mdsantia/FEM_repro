# Isolated Setup and Memory Diagnostics

Every repetition and level-method combination was executed in a fresh subprocess to prevent retained Python, Firedrake, PETSc, NumPy, and allocator state from contaminating later measurements.

## Summary

| Level | DoFs | Method | Repetitions | Problem setup median (s) | Setup + first solve median (s) | RSS delta median | Absolute peak RSS | Local NumPy payload | PETSc matrix memory |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 25 | Assembled AIJ | 3 | 2.014651e-02 | 1.004414e-01 | 8.70 MiB | 765.12 MiB | – | – |
| 2 | 25 | Firedrake matfree | 3 | 2.812501e-02 | 1.594406e-01 | 9.01 MiB | 765.12 MiB | – | – |
| 2 | 25 | Local precomputed | 3 | 2.003651e-02 | 9.156130e-02 | 8.61 MiB | 765.12 MiB | 6.77 KiB | – |
| 2 | 25 | Local direct | 3 | 2.020025e-02 | 9.198054e-02 | 8.21 MiB | 765.12 MiB | 4.52 KiB | – |
| 3 | 81 | Assembled AIJ | 3 | 2.184369e-02 | 1.022719e-01 | 8.74 MiB | 765.12 MiB | – | – |
| 3 | 81 | Firedrake matfree | 3 | 3.109365e-02 | 1.516235e-01 | 8.82 MiB | 765.12 MiB | – | – |
| 3 | 81 | Local precomputed | 3 | 2.362771e-02 | 1.004437e-01 | 8.41 MiB | 765.12 MiB | 26.52 KiB | – |
| 3 | 81 | Local direct | 3 | 2.182081e-02 | 9.982575e-02 | 8.59 MiB | 765.12 MiB | 17.52 KiB | – |
| 4 | 289 | Assembled AIJ | 3 | 2.815812e-02 | 1.015610e-01 | 9.46 MiB | 765.12 MiB | – | – |
| 4 | 289 | Firedrake matfree | 3 | 4.395284e-02 | 1.659342e-01 | 9.00 MiB | 765.12 MiB | – | – |
| 4 | 289 | Local precomputed | 3 | 2.904741e-02 | 1.197975e-01 | 8.94 MiB | 765.12 MiB | 105.02 KiB | – |
| 4 | 289 | Local direct | 3 | 2.809354e-02 | 1.165034e-01 | 8.73 MiB | 765.12 MiB | 69.02 KiB | – |
| 5 | 1089 | Assembled AIJ | 3 | 5.294351e-02 | 1.123284e-01 | 10.30 MiB | 765.12 MiB | – | – |
| 5 | 1089 | Firedrake matfree | 3 | 9.373427e-02 | 1.872500e-01 | 10.25 MiB | 765.12 MiB | – | – |
| 5 | 1089 | Local precomputed | 3 | 5.307984e-02 | 1.923493e-01 | 10.00 MiB | 765.12 MiB | 418.02 KiB | – |
| 5 | 1089 | Local direct | 3 | 5.351853e-02 | 2.048915e-01 | 9.73 MiB | 765.12 MiB | 274.02 KiB | – |

## Local representation payload validation

The expected difference between the precomputed and direct local payloads is nine double-precision element-matrix entries per triangle:

$$
N_T \times 9 \times 8\text{ bytes}.
$$

| Level | Cells | Direct payload | Precomputed payload | Observed difference | Expected matrix difference | Match |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 4.52 KiB | 6.77 KiB | 2.25 KiB | 2.25 KiB | True |
| 3 | 128 | 17.52 KiB | 26.52 KiB | 9.00 KiB | 9.00 KiB | True |
| 4 | 512 | 69.02 KiB | 105.02 KiB | 36.00 KiB | 36.00 KiB | True |
| 5 | 2048 | 274.02 KiB | 418.02 KiB | 144.00 KiB | 144.00 KiB | True |

## Important limitations

- `setup_and_first_solve_seconds` is the first complete method call. It may include construction, assembly, JIT compilation, preconditioner creation, and the linear solve.
- Timing medians summarize fresh-process repetitions. The summary CSV also records the first and third quartiles.
- Problem setup is mathematically identical across methods. Differences in measured problem-setup time therefore represent worker startup, caching, scheduling, and runtime variation.
- RSS measures the complete process, including loaded libraries and allocator behavior. RSS is too coarse for fine operator-storage comparisons at the tested problem sizes.
- Absolute peak RSS includes imports and runtime initialization and is not operator-only memory.
- Local NumPy payload counts retained array data only and excludes Python object overhead.
- PETSc matrix memory is reported only when PETSc provides a positive finite value. Zero and nonfinite values are treated as unavailable.
- These diagnostics are not a controlled performance benchmark.
