"""Derive and verify the unit-lapse curvature-only NEC obstruction."""

from __future__ import annotations

import numpy as np
import sympy as sp


def main() -> None:
    time, radius, theta, phi = sp.symbols("t r theta phi", real=True)
    b = sp.Function("B")(radius)
    beta = sp.Function("beta")(radius)
    coordinates = [time, radius, theta, phi]
    metric = sp.Matrix(
        [
            [-1 + b**2 * beta**2, -b**2 * beta, 0, 0],
            [-b**2 * beta, b**2, 0, 0],
            [0, 0, radius**2, 0],
            [0, 0, 0, radius**2 * sp.sin(theta) ** 2],
        ]
    )
    inverse = sp.simplify(metric.inv())
    dimension = 4

    christoffel = [
        [[sp.S.Zero] * dimension for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for a in range(dimension):
        for c in range(dimension):
            for d in range(dimension):
                christoffel[a][c][d] = sp.simplify(
                    sp.Rational(1, 2)
                    * sum(
                        inverse[a, e]
                        * (
                            sp.diff(metric[e, d], coordinates[c])
                            + sp.diff(metric[e, c], coordinates[d])
                            - sp.diff(metric[c, d], coordinates[e])
                        )
                        for e in range(dimension)
                    )
                )

    ricci = sp.MutableDenseMatrix.zeros(dimension, dimension)
    for a in range(dimension):
        for c in range(dimension):
            ricci[a, c] = sp.simplify(
                sum(
                    sp.diff(christoffel[d][a][c], coordinates[d])
                    - sp.diff(christoffel[d][a][d], coordinates[c])
                    + sum(
                        christoffel[d][d][e] * christoffel[e][a][c]
                        - christoffel[d][c][e] * christoffel[e][a][d]
                        for e in range(dimension)
                    )
                    for d in range(dimension)
                )
            )

    scalar = sp.simplify(
        sum(
            inverse[a, c] * ricci[a, c]
            for a in range(dimension)
            for c in range(dimension)
        )
    )
    einstein = sp.simplify(ricci - metric * scalar / 2)

    normal = sp.Matrix([1, beta, 0, 0])
    radial_unit = sp.Matrix([0, 1 / b, 0, 0])
    null_plus = normal + radial_unit
    null_minus = normal - radial_unit
    contraction_plus = sp.factor(
        (null_plus.T * einstein * null_plus)[0]
    )
    contraction_minus = sp.factor(
        (null_minus.T * einstein * null_minus)[0]
    )
    expected_plus = 2 * sp.diff(b, radius) * (1 + b * beta) ** 2 / (
        radius * b**3
    )
    expected_minus = 2 * sp.diff(b, radius) * (1 - b * beta) ** 2 / (
        radius * b**3
    )
    assert sp.simplify(contraction_plus - expected_plus) == 0
    assert sp.simplify(contraction_minus - expected_minus) == 0

    summed = sp.factor(contraction_plus + contraction_minus)
    expected_sum = 4 * sp.diff(b, radius) * (1 + b**2 * beta**2) / (
        radius * b**3
    )
    assert sp.simplify(summed - expected_sum) == 0

    r_grid = np.linspace(1.0 + 1.0e-6, 3.0 - 1.0e-6, 400_001)
    x = (r_grid - 1.0) / 2.0
    b_grid = 1 + 0.2 * np.sin(np.pi * x) ** 4
    b_prime = (
        0.2
        * 4
        * np.sin(np.pi * x) ** 3
        * np.cos(np.pi * x)
        * np.pi
        / 2.0
    )
    beta_grid = 0.1 * np.sin(np.pi * x) ** 4
    null_plus_grid = (
        2
        * b_prime
        * (1 + b_grid * beta_grid) ** 2
        / (r_grid * b_grid**3)
    )
    null_minus_grid = (
        2
        * b_prime
        * (1 - b_grid * beta_grid) ** 2
        / (r_grid * b_grid**3)
    )
    bump_fails = bool(
        null_plus_grid.min() < 0 or null_minus_grid.min() < 0
    )

    print(f"radial null sum = {summed}")
    print("NEC in both radial directions implies B' >= 0")
    print("B(R1)=1 and B(infinity)=1 then imply B=1")
    print(f"compact curvature bump fails NEC = {bump_fails}")
    print("VERDICT: PASS" if bump_fails else "VERDICT: FAIL")


if __name__ == "__main__":
    main()

