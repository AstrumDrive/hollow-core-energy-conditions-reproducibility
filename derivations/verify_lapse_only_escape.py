"""Verify the regular lapse-only, flat-slice hollow-shell escape."""

from __future__ import annotations

import math

import numpy as np
import sympy as sp


R1 = 1.0
R2 = 3.0
WIDTH = R2 - R1


def smoothstep9(x: np.ndarray) -> np.ndarray:
    return (
        126 * x**5
        - 420 * x**6
        + 540 * x**7
        - 315 * x**8
        + 70 * x**9
    )


def smoothstep9_prime(x: np.ndarray) -> np.ndarray:
    return 630 * x**4 * (1 - x) ** 4


def evaluate(mass: float, points: int = 400_001) -> dict[str, float | bool]:
    radius = np.linspace(R1 + 1.0e-8, R2 - 1.0e-8, points)
    x = (radius - R1) / WIDTH
    shape = smoothstep9(x)
    shape_prime = smoothstep9_prime(x) / WIDTH
    m = mass * shape
    m_prime = mass * shape_prime
    f = 1 - 2 * m / radius
    rho = m_prime / (4 * math.pi * radius**2)
    phi_prime = m / (radius * (radius - 2 * m))
    phi_outer = 0.5 * math.log(1 - 2 * mass / R2)
    phi_core = phi_outer - np.trapezoid(phi_prime, radius)

    increments = 0.5 * (phi_prime[1:] + phi_prime[:-1]) * np.diff(radius)
    phi = phi_core + np.concatenate(([0.0], np.cumsum(increments)))
    diagonal_lapse = np.exp(phi)
    diagonal_radial_factor = 1 / np.sqrt(f)
    lapse = diagonal_lapse * diagonal_radial_factor
    shift = diagonal_lapse * np.sqrt(diagonal_radial_factor**2 - 1)
    pressure_ratio = m / (2 * (radius - 2 * m))
    matter = rho > rho.max() * 1.0e-11

    reconstructed_error = np.max(
        np.abs(lapse**2 - shift**2 - diagonal_lapse**2)
    )
    passed = bool(
        f.min() > 0
        and rho.min() >= -1.0e-12
        and pressure_ratio[matter].max() <= 1 + 2.0e-8
        and reconstructed_error < 1.0e-10
    )
    return {
        "passed": passed,
        "max_compactness": float(np.max(2 * m / radius)),
        "proper_energy": float(np.trapezoid(m_prime / np.sqrt(f), radius)),
        "core_redshift": float(math.exp(-phi_core) - 1),
        "core_lapse": float(math.exp(phi_core)),
        "max_shift": float(shift.max()),
        "max_p_perp_over_rho": float(pressure_ratio[matter].max()),
        "reconstruction_error": float(reconstructed_error),
    }


def main() -> None:
    a, c = sp.symbols("A C", positive=True)
    h_prime = sp.sqrt(c**2 - 1) / a
    spatial_factor_squared = sp.simplify(c**2 - a**2 * h_prime**2)
    lapse = sp.simplify(a * c / sp.sqrt(spatial_factor_squared))
    shift = sp.simplify(a**2 * h_prime / spatial_factor_squared)
    assert spatial_factor_squared == 1
    assert sp.simplify(lapse**2 - shift**2 - a**2) == 0

    radius = np.linspace(R1, R2, 1_000_001)
    shape_ratio = smoothstep9((radius - R1) / WIDTH) / radius
    max_shape_ratio = float(shape_ratio.max())
    critical_mass = 0.4 / max_shape_ratio
    target_compactness = 0.1
    target_mass = target_compactness / (2 * max_shape_ratio)

    target = evaluate(target_mass)
    below = evaluate(0.999 * critical_mass)
    above = evaluate(1.001 * critical_mass)

    print(f"max[S9/r] = {max_shape_ratio:.12f}")
    print(f"critical mass = {critical_mass:.12f}")
    print(f"target mass at Cmax=0.1 = {target_mass:.12f}")
    for key, value in target.items():
        print(f"{key} = {value}")
    print(f"below threshold passes = {below['passed']}")
    print(f"above threshold passes = {above['passed']}")
    print(
        "VERDICT: PASS"
        if target["passed"] and below["passed"] and not above["passed"]
        else "VERDICT: FAIL"
    )


if __name__ == "__main__":
    main()

