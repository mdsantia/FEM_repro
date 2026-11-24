# Diagnostic Solver-Performance Comparison

The table records elapsed times, iteration counts, and assembled-system residuals produced by the current implementations.

## Important limitation

This table is not a controlled assembled-versus-matrix-free performance benchmark:

- the assembled reference uses `preonly` with LU;
- the other methods use CG with Jacobi preconditioning;
- the independent local operators execute through Python and NumPy;
- code generation, setup, and one-run timing effects may be included;
- the custom scatter uses `numpy.add.at`.

The results should therefore be interpreted as implementation diagnostics.

| Level | DoFs | Method | Solver | Iterations | Time (s) | Time/DoF (s) | Time/cell (s) | Ratio to assembled | Relative residual |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | 25 | Assembled AIJ | preonly + LU | 1 | 2.923135e-02 | 1.169254e-03 | 9.134796e-04 | 1.0000 | 3.344044e-16 |
| 2 | 25 | Firedrake matfree | CG + Jacobi | 2 | 4.402800e-02 | 1.761120e-03 | 1.375875e-03 | 1.5062 | 3.629139e-16 |
| 2 | 25 | Local precomputed | CG + Jacobi | 2 | 2.328189e-02 | 9.312756e-04 | 7.275591e-04 | 0.7965 | 3.230653e-16 |
| 2 | 25 | Local direct | CG + Jacobi | 2 | 1.839977e-02 | 7.359909e-04 | 5.749929e-04 | 0.6295 | 3.230653e-16 |
| 3 | 81 | Assembled AIJ | preonly + LU | 1 | 1.803528e-02 | 2.226577e-04 | 1.409006e-04 | 1.0000 | 2.248357e-15 |
| 3 | 81 | Firedrake matfree | CG + Jacobi | 6 | 1.905082e-02 | 2.351953e-04 | 1.488345e-04 | 1.0563 | 1.778521e-15 |
| 3 | 81 | Local precomputed | CG + Jacobi | 6 | 1.603067e-02 | 1.979095e-04 | 1.252396e-04 | 0.8889 | 1.987482e-15 |
| 3 | 81 | Local direct | CG + Jacobi | 6 | 1.571557e-02 | 1.940194e-04 | 1.227779e-04 | 0.8714 | 1.812204e-15 |
| 4 | 289 | Assembled AIJ | preonly + LU | 1 | 2.052714e-02 | 7.102818e-05 | 4.009208e-05 | 1.0000 | 8.276388e-15 |
| 4 | 289 | Firedrake matfree | CG + Jacobi | 26 | 2.691505e-02 | 9.313167e-05 | 5.256846e-05 | 1.3112 | 1.129075e-14 |
| 4 | 289 | Local precomputed | CG + Jacobi | 26 | 3.585429e-02 | 1.240633e-04 | 7.002792e-05 | 1.7467 | 1.326994e-14 |
| 4 | 289 | Local direct | CG + Jacobi | 26 | 3.506549e-02 | 1.213339e-04 | 6.848728e-05 | 1.7082 | 1.298098e-14 |
| 5 | 1089 | Assembled AIJ | preonly + LU | 1 | 2.301754e-02 | 2.113640e-05 | 1.123903e-05 | 1.0000 | 3.958931e-14 |
| 5 | 1089 | Firedrake matfree | CG + Jacobi | 56 | 3.801102e-02 | 3.490452e-05 | 1.856007e-05 | 1.6514 | 8.116935e-13 |
| 5 | 1089 | Local precomputed | CG + Jacobi | 56 | 1.299114e-01 | 1.192942e-04 | 6.343331e-05 | 5.6440 | 8.094651e-13 |
| 5 | 1089 | Local direct | CG + Jacobi | 56 | 1.144405e-01 | 1.050877e-04 | 5.587917e-05 | 4.9719 | 8.087965e-13 |
