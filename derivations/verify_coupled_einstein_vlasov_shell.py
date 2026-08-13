"""Construct a coupled static Einstein--Vlasov shell in areal coordinates.

The distribution uses the conserved particle energy and angular momentum.  The
metric variables are integrated from the Einstein equations; no lapse profile
is prescribed.  This file is intentionally self-contained so that the
constitutive-branch calculation can be reproduced on a standard Python setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid, solve_ivp


@dataclass(frozen=True)
class Model:
    amplitude: float
    central_y: float
    angular_cutoff: float = 0.5
    energy_power: int = 1
    angular_power: int = 1


@dataclass
class Solution:
    radius: np.ndarray
    mass: np.ndarray
    y: np.ndarray
    density: np.ndarray
    radial_pressure: np.ndarray
    tangential_pressure: np.ndarray
    support_function: np.ndarray


class PhaseQuadrature:
    def __init__(self, order: int) -> None:
        nodes, weights = leggauss(order)
        self.w_nodes = nodes
        self.w_weights = weights
        self.l_nodes = 0.5 * (nodes + 1.0)
        self.l_weights = 0.5 * weights

    def matter(self, radius: float, y: float, model: Model) -> tuple[float, float, float]:
        if radius <= 0.0 or y <= 0.0:
            return 0.0, 0.0, 0.0
        kinetic_cap = math.exp(2.0 * y) - 1.0
        angular_span = radius * radius * kinetic_cap - model.angular_cutoff
        if angular_span <= 0.0:
            return 0.0, 0.0, 0.0

        angular = model.angular_cutoff + angular_span * self.l_nodes
        angular_weights = angular_span * self.l_weights
        radial_cap = np.sqrt(np.maximum(kinetic_cap - angular / radius**2, 0.0))
        radial_momentum = radial_cap[:, None] * self.w_nodes[None, :]
        radial_weights = radial_cap[:, None] * self.w_weights[None, :]
        angular_grid = angular[:, None]
        epsilon = np.sqrt(1.0 + radial_momentum**2 + angular_grid / radius**2)
        energy_deficit = np.maximum(1.0 - math.exp(-y) * epsilon, 0.0)
        distribution = (
            model.amplitude
            * energy_deficit**model.energy_power
            * (angular_grid - model.angular_cutoff) ** model.angular_power
        )
        weights = angular_weights[:, None] * radial_weights

        density = 2.0 * math.pi / radius**2 * float(
            np.sum(weights * epsilon * distribution)
        )
        radial_pressure = 2.0 * math.pi / radius**2 * float(
            np.sum(weights * radial_momentum**2 / epsilon * distribution)
        )
        tangential_pressure = math.pi / radius**4 * float(
            np.sum(weights * angular_grid / epsilon * distribution)
        )
        return density, radial_pressure, tangential_pressure


def integrate_model(
    model: Model,
    quadrature_order: int,
    radial_points: int = 3001,
    outer_radius: float = 80.0,
    relative_tolerance: float = 2.0e-9,
    maximum_step: float = 0.04,
) -> Solution:
    quadrature = PhaseQuadrature(quadrature_order)

    def rhs(radius: float, state: np.ndarray) -> tuple[float, float]:
        mass, y = float(state[0]), float(state[1])
        density, radial_pressure, _ = quadrature.matter(radius, y, model)
        denominator = radius * (radius - 2.0 * mass)
        if denominator <= 0.0:
            raise RuntimeError("apparent horizon encountered")
        mass_derivative = 4.0 * math.pi * radius**2 * density
        y_derivative = -(
            mass + 4.0 * math.pi * radius**3 * radial_pressure
        ) / denominator
        return mass_derivative, y_derivative

    radius = np.linspace(1.0e-4, outer_radius, radial_points)
    integrated = solve_ivp(
        rhs,
        (float(radius[0]), float(radius[-1])),
        (0.0, model.central_y),
        t_eval=radius,
        rtol=relative_tolerance,
        atol=(1.0e-11, 1.0e-11),
        max_step=maximum_step,
    )
    if not integrated.success:
        raise RuntimeError(integrated.message)
    mass = integrated.y[0]
    y = integrated.y[1]
    density = np.empty_like(radius)
    radial_pressure = np.empty_like(radius)
    tangential_pressure = np.empty_like(radius)
    for index, (sample_radius, sample_y) in enumerate(zip(radius, y)):
        density[index], radial_pressure[index], tangential_pressure[index] = (
            quadrature.matter(float(sample_radius), float(sample_y), model)
        )
    support_function = np.exp(2.0 * y) - 1.0 - model.angular_cutoff / radius**2
    return Solution(
        radius,
        mass,
        y,
        density,
        radial_pressure,
        tangential_pressure,
        support_function,
    )


def support_interval(solution: Solution) -> tuple[float, float] | None:
    """Bracket the matter support, or return None if it has no interior edges.

    The inner edge is bracketed by ``linear_root(first - 1, first)``.  If the
    support already holds at the innermost grid point then ``first == 0`` and
    that would index ``-1``, silently wrapping to the outer end of the array
    and returning a meaningless inner radius.  That configuration is a
    center-filled (non-hollow) solution, not a shell, so it is rejected here
    rather than interpolated across the wrap.  It is unreachable while the
    angular-momentum cutoff ``L0 > 0``, but a zero-cutoff control makes it
    reachable and any other caller remains exposed.
    """
    supported = solution.support_function > 0.0
    indices = np.flatnonzero(supported)
    if indices.size == 0 or indices[-1] == solution.radius.size - 1:
        return None
    first = int(indices[0])
    last = int(indices[-1])
    # Guarded: never bracket below index 0.  A support that reaches the first
    # grid point has no resolved inner edge to report.
    if first == 0:
        return None

    def linear_root(left: int, right: int) -> float:
        if left < 0 or right >= solution.radius.size:
            raise RuntimeError(f"support edge bracket out of range: ({left}, {right})")
        radius_left = solution.radius[left]
        radius_right = solution.radius[right]
        value_left = solution.support_function[left]
        value_right = solution.support_function[right]
        return float(
            radius_left
            - value_left * (radius_right - radius_left) / (value_right - value_left)
        )

    return linear_root(first - 1, first), linear_root(last, last + 1)


def diagnostics(solution: Solution, model: Model) -> dict[str, float | int | bool]:
    interval = support_interval(solution)
    if interval is None:
        raise RuntimeError(
            "matter support is empty, reaches the first grid point (center-filled, "
            "so there is no resolved inner edge), or reaches the outer boundary"
        )
    supported = solution.support_function > 0.0
    transitions = int(np.count_nonzero(supported[1:] != supported[:-1]))
    matter = solution.density > 0.0
    mass = float(solution.mass[-1])
    compactness = float(np.max(2.0 * solution.mass / solution.radius))
    y_infinity = float(
        solution.y[-1]
        + 0.5 * math.log(1.0 - 2.0 * mass / solution.radius[-1])
    )

    margins = np.vstack(
        (
            solution.density,
            solution.density + solution.radial_pressure,
            solution.density + solution.tangential_pressure,
            solution.density
            + solution.radial_pressure
            + 2.0 * solution.tangential_pressure,
            solution.density - solution.radial_pressure,
            solution.density - solution.tangential_pressure,
        )
    )
    minimum_normalized_margin = float(
        np.min(margins[:, matter] / solution.density[matter])
    )

    mass_source = 4.0 * math.pi * solution.radius**2 * solution.density
    reconstructed_mass = np.concatenate(
        ([0.0], cumulative_trapezoid(mass_source, solution.radius))
    )
    mass_integral_residual = float(
        np.max(np.abs(solution.mass - reconstructed_mass)) / mass
    )
    y_source = -(
        solution.mass
        + 4.0 * math.pi * solution.radius**3 * solution.radial_pressure
    ) / (solution.radius * (solution.radius - 2.0 * solution.mass))
    reconstructed_y = model.central_y + np.concatenate(
        ([0.0], cumulative_trapezoid(y_source, solution.radius))
    )
    y_integral_residual = float(np.max(np.abs(solution.y - reconstructed_y)))

    exterior = solution.radius > interval[1] + 0.5
    exterior_invariant = solution.y[exterior] + 0.5 * np.log(
        1.0 - 2.0 * mass / solution.radius[exterior]
    )
    exterior_schwarzschild_residual = float(np.ptp(exterior_invariant))
    exterior_mass_residual = float(
        np.max(np.abs(solution.mass[exterior] - mass)) / mass
    )
    return {
        "inner_radius": interval[0],
        "outer_radius": interval[1],
        "support_transitions": transitions,
        "adm_mass": mass,
        "max_compactness": compactness,
        "y_infinity": y_infinity,
        "normalized_energy_cutoff": math.exp(y_infinity),
        "minimum_normalized_margin": minimum_normalized_margin,
        "mass_integral_residual": mass_integral_residual,
        "y_integral_residual": y_integral_residual,
        "exterior_schwarzschild_residual": exterior_schwarzschild_residual,
        "exterior_mass_residual": exterior_mass_residual,
    }


def main() -> None:
    model = Model(amplitude=0.03, central_y=0.20)
    configurations = (
        (12, 3001, 4.0e-8, 0.08),
        (20, 4501, 8.0e-9, 0.05),
        (32, 6001, 2.0e-9, 0.03),
    )
    analyses: list[dict[str, float | int | bool]] = []
    for order, radial_points, tolerance, maximum_step in configurations:
        solution = integrate_model(
            model,
            order,
            radial_points=radial_points,
            outer_radius=60.0,
            relative_tolerance=tolerance,
            maximum_step=maximum_step,
        )
        analysis = diagnostics(solution, model)
        analyses.append(analysis)
        print(
            f"order={order:02d} Nr={radial_points:04d} "
            f"support=[{analysis['inner_radius']:.9f},"
            f"{analysis['outer_radius']:.9f}] "
            f"M={analysis['adm_mass']:.12f} "
            f"Cmax={analysis['max_compactness']:.9f} "
            f"E0_norm={analysis['normalized_energy_cutoff']:.9f}"
        )

    coarse = analyses[0]
    fine = analyses[-1]
    mass_convergence = abs(float(fine["adm_mass"]) - float(coarse["adm_mass"])) / float(
        fine["adm_mass"]
    )
    support_convergence = max(
        abs(float(fine["inner_radius"]) - float(coarse["inner_radius"])),
        abs(float(fine["outer_radius"]) - float(coarse["outer_radius"])),
    )
    passed = (
        int(fine["support_transitions"]) == 2
        and float(fine["inner_radius"]) > 0.1
        and float(fine["outer_radius"]) < 59.0
        and float(fine["adm_mass"]) > 0.0
        and float(fine["max_compactness"]) < 0.8
        and float(fine["normalized_energy_cutoff"]) < 1.0
        and float(fine["minimum_normalized_margin"]) > 0.0
        and float(fine["mass_integral_residual"]) < 2.0e-4
        and float(fine["y_integral_residual"]) < 2.0e-5
        and float(fine["exterior_schwarzschild_residual"]) < 2.0e-9
        and float(fine["exterior_mass_residual"]) < 2.0e-9
        and mass_convergence < 1.0e-5
        and support_convergence < 2.0e-3
    )
    print(f"support transitions = {fine['support_transitions']}")
    print(f"asymptotic y = {fine['y_infinity']:.12f}")
    print(f"minimum normalized Type-I margin = {fine['minimum_normalized_margin']:.12f}")
    print(f"mass integral residual = {fine['mass_integral_residual']:.3e}")
    print(f"y integral residual = {fine['y_integral_residual']:.3e}")
    print(
        "exterior Schwarzschild/mass residuals = "
        f"{fine['exterior_schwarzschild_residual']:.3e}, "
        f"{fine['exterior_mass_residual']:.3e}"
    )
    print(f"coarse/fine mass relative change = {mass_convergence:.3e}")
    print(f"coarse/fine support absolute change = {support_convergence:.3e}")
    print("metric solved from coupled Einstein equations = True")
    print("stability or transport established = False")
    print("VERDICT: PASS" if passed else "VERDICT: FAIL")


if __name__ == "__main__":
    main()
