# Finite-Element Solution Convergence

The table below reports the discretization errors of the assembled reference solution. The other three operator implementations produce the same errors and rates to the displayed precision because their coefficient vectors agree with the assembled solution to solver tolerance.

For continuous $P_1$ finite elements on quasi-uniform affine triangular meshes, the expected behavior is

$$
\|u-u_h\|_{L^2(\Omega)}=O(h^2),
\qquad
|u-u_h|_{H^1(\Omega)}=O(h).
$$

## Assembled reference

| Level | $h$ | DoFs | $L^2$ error | Rate | $H^1$ seminorm error | Rate | Relative residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.25000 | 25 | 7.907545e-02 | – | 8.385483e-01 | – | 3.344044e-16 |
| 3 | 0.12500 | 81 | 2.113277e-02 | 1.9037 | 4.317983e-01 | 0.9575 | 2.248357e-15 |
| 4 | 0.06250 | 289 | 5.377435e-03 | 1.9745 | 2.175363e-01 | 0.9891 | 8.276388e-15 |
| 5 | 0.03125 | 1089 | 1.350436e-03 | 1.9935 | 1.089754e-01 | 0.9973 | 3.958931e-14 |

## All solution methods

| Method | Level | $h$ | DoFs | $L^2$ error | Rate | $H^1$ seminorm error | Rate | Relative residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Assembled AIJ | 2 | 0.25000 | 25 | 7.907545e-02 | – | 8.385483e-01 | – | 3.344044e-16 |
| Assembled AIJ | 3 | 0.12500 | 81 | 2.113277e-02 | 1.9037 | 4.317983e-01 | 0.9575 | 2.248357e-15 |
| Assembled AIJ | 4 | 0.06250 | 289 | 5.377435e-03 | 1.9745 | 2.175363e-01 | 0.9891 | 8.276388e-15 |
| Assembled AIJ | 5 | 0.03125 | 1089 | 1.350436e-03 | 1.9935 | 1.089754e-01 | 0.9973 | 3.958931e-14 |
| Firedrake matfree | 2 | 0.25000 | 25 | 7.907545e-02 | – | 8.385483e-01 | – | 3.629139e-16 |
| Firedrake matfree | 3 | 0.12500 | 81 | 2.113277e-02 | 1.9037 | 4.317983e-01 | 0.9575 | 1.778521e-15 |
| Firedrake matfree | 4 | 0.06250 | 289 | 5.377435e-03 | 1.9745 | 2.175363e-01 | 0.9891 | 1.129075e-14 |
| Firedrake matfree | 5 | 0.03125 | 1089 | 1.350436e-03 | 1.9935 | 1.089754e-01 | 0.9973 | 8.116935e-13 |
| Local precomputed | 2 | 0.25000 | 25 | 7.907545e-02 | – | 8.385483e-01 | – | 3.230653e-16 |
| Local precomputed | 3 | 0.12500 | 81 | 2.113277e-02 | 1.9037 | 4.317983e-01 | 0.9575 | 1.987482e-15 |
| Local precomputed | 4 | 0.06250 | 289 | 5.377435e-03 | 1.9745 | 2.175363e-01 | 0.9891 | 1.326994e-14 |
| Local precomputed | 5 | 0.03125 | 1089 | 1.350436e-03 | 1.9935 | 1.089754e-01 | 0.9973 | 8.094651e-13 |
| Local direct | 2 | 0.25000 | 25 | 7.907545e-02 | – | 8.385483e-01 | – | 3.230653e-16 |
| Local direct | 3 | 0.12500 | 81 | 2.113277e-02 | 1.9037 | 4.317983e-01 | 0.9575 | 1.812204e-15 |
| Local direct | 4 | 0.06250 | 289 | 5.377435e-03 | 1.9745 | 2.175363e-01 | 0.9891 | 1.298098e-14 |
| Local direct | 5 | 0.03125 | 1089 | 1.350436e-03 | 1.9935 | 1.089754e-01 | 0.9973 | 8.087965e-13 |

## Interpretation

- The $L^2$ rates approach two.
- The $H^1$-seminorm rates approach one.
- All four solution paths preserve the expected $P_1$ convergence.
- Differences among methods are algebraic-solver effects far below the finite-element discretization error.
