"""Verify a self-consistent microscopic hollow-shell branch.

The constitutive datum is the nonnegative Einstein--Vlasov distribution used by
``verify_coupled_einstein_vlasov_shell.py``.  This verifier scans the central
relative cutoff potential, checks the hollow compact support and all Type-I
energy-condition margins, and certifies the fixed-outer-radius ADM scaling.
It deliberately makes no stability, confinement, or apparatus claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from verify_coupled_einstein_vlasov_shell import (
    Model,
    Solution,
    diagnostics,
    integrate_model,
)


CENTRAL_Y = (0.20, 0.15, 0.10, 0.07, 0.05, 0.03)
AMPLITUDE = 0.03
ANGULAR_CUTOFF = 0.5

# Numerics of the zero-cutoff negative control.  Both legs of the control (with
# and without the angular-momentum cutoff) are solved at exactly these settings
# so that L0 is the only variable.  They are deliberately coarser than the
# production branch: the control establishes a qualitative transition, not a
# converged number, and an L0 = 0 solve is expensive because matter then fills
# the whole interior.
CONTROL_NUMERICS = (12, 3001, 4.0e-8, 0.08)

# Predeclared thresholds for the zero-cutoff negative control.
HOLLOW_MIN_INNER_RADIUS = 0.1
ZERO_CUTOFF_MAX_INNER_RADIUS = 1.0e-3

# Predeclared discrimination floor for "the center is materially filled".
#
# ``rho(r) > 0.0`` alone is a weak discriminator: it would also accept pure
# accumulation roundoff.  The floor is expressed relative to the solution's own
# peak density and is set from the float64 roundoff scale, NOT from the observed
# output: the density is a sum of nonnegative Gauss-Legendre contributions, so
# its roundoff is bounded by a few ulps (~2.2e-16 relative) of the largest term;
# 1e-12 is that scale times ~4500 and therefore cannot be reached by noise.  The
# L0 > 0 leg returns exactly 0.0 (the angular-momentum barrier makes the
# quadrature domain empty), so the two legs are separated by this floor by a
# mechanism, not by a margin that had to be tuned.
CENTRAL_DENSITY_NOISE_FLOOR_RELATIVE = 1.0e-12


class VerificationError(RuntimeError):
    """Raised when a scientific check cannot be evaluated."""


def support_edges(solution: Solution) -> dict[str, float | int | bool]:
    """Index-0-safe support endpoints and central density.

    ``verify_coupled_einstein_vlasov_shell.support_interval`` brackets the
    inner edge with ``linear_root(first - 1, first)``.  If the support already
    holds at the innermost grid point then ``first == 0`` and that indexes
    ``-1``, silently wrapping to the outer end of the array and returning a
    meaningless inner radius.  That is unreachable while ``L0 > 0``, but the
    zero-cutoff control below deliberately makes it reachable, so this function
    detects the case explicitly instead of interpolating across the wrap.
    """
    supported = np.asarray(solution.support_function) > 0.0
    indices = np.flatnonzero(supported)
    if indices.size == 0:
        raise VerificationError("matter support is empty")
    first = int(indices[0])
    last = int(indices[-1])

    def linear_root(left: int, right: int) -> float:
        if left < 0 or right >= solution.radius.size:
            raise VerificationError(
                f"support edge bracket out of range: ({left}, {right})"
            )
        radius_left = float(solution.radius[left])
        radius_right = float(solution.radius[right])
        value_left = float(solution.support_function[left])
        value_right = float(solution.support_function[right])
        return radius_left - value_left * (radius_right - radius_left) / (
            value_right - value_left
        )

    # Guarded: never bracket below index 0.
    center_filled = first == 0
    inner_radius = (
        float(solution.radius[0]) if center_filled else linear_root(first - 1, first)
    )
    boundary_reached = last == solution.radius.size - 1
    outer_radius = (
        float(solution.radius[-1]) if boundary_reached else linear_root(last, last + 1)
    )
    central_density = float(solution.density[0])
    peak_density = float(np.max(solution.density))
    if not peak_density > 0.0:
        raise VerificationError("solution carries no positive density anywhere")
    return {
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "support_reaches_first_grid_point": center_filled,
        "support_reaches_outer_boundary": boundary_reached,
        "support_transitions": int(np.count_nonzero(supported[1:] != supported[:-1])),
        "central_density": central_density,
        "peak_density": peak_density,
        "central_over_peak_density": central_density / peak_density,
        "innermost_grid_radius": float(solution.radius[0]),
    }


def solve_edges(central_y: float, angular_cutoff: float) -> dict[str, float | int | bool]:
    """Solve the model at CONTROL_NUMERICS and return guarded support edges."""
    order, radial_points, tolerance, maximum_step = CONTROL_NUMERICS
    solution = integrate_model(
        Model(
            amplitude=AMPLITUDE,
            central_y=central_y,
            angular_cutoff=angular_cutoff,
            energy_power=1,
            angular_power=1,
        ),
        order,
        radial_points=radial_points,
        outer_radius=80.0,
        relative_tolerance=tolerance,
        maximum_step=maximum_step,
    )
    edges = support_edges(solution)
    edges["central_y"] = central_y
    edges["angular_cutoff"] = angular_cutoff
    return edges


def zero_cutoff_control(
    central_y: float,
) -> tuple[dict[str, float | int | bool], dict[str, float | int | bool]]:
    """Actually re-solve the model with L0 = 0 and record what happens.

    Two real Einstein--Vlasov solves at identical numerics, not an inequality
    on a constant.  With L0 = 0 the centrifugal vacuum barrier is removed, so
    the compact hollow support is expected to collapse to a filled ball: the
    support lower endpoint falls to the innermost grid point, the inner
    vacuum-to-matter transition disappears, and the central density becomes
    positive.
    """
    hollow = solve_edges(central_y, ANGULAR_CUTOFF)
    filled = solve_edges(central_y, 0.0)
    return hollow, filled


def solve_point(central_y: float, order: int, radial_points: int,
                tolerance: float, maximum_step: float) -> dict[str, float | int | bool]:
    model = Model(
        amplitude=AMPLITUDE,
        central_y=central_y,
        angular_cutoff=ANGULAR_CUTOFF,
        energy_power=1,
        angular_power=1,
    )
    solution = integrate_model(
        model,
        order,
        radial_points=radial_points,
        outer_radius=80.0,
        relative_tolerance=tolerance,
        maximum_step=maximum_step,
    )
    result = diagnostics(solution, model)
    result["central_y"] = central_y
    result["mass_over_outer_radius"] = (
        float(result["adm_mass"]) / float(result["outer_radius"])
    )
    result["normalized_core_depth"] = (
        central_y - math.log(float(result["normalized_energy_cutoff"]))
    )
    return result


def main() -> None:
    branch = [solve_point(y, 20, 7501, 8.0e-9, 0.05) for y in CENTRAL_Y]

    # Independent coarse/fine comparison at the weakest-field endpoint.
    weak_coarse = solve_point(CENTRAL_Y[-1], 12, 5001, 4.0e-8, 0.08)
    weak_fine = solve_point(CENTRAL_Y[-1], 28, 10001, 2.0e-9, 0.03)
    q_coarse = float(weak_coarse["mass_over_outer_radius"])
    q_fine = float(weak_fine["mass_over_outer_radius"])
    q_convergence = abs(q_fine - q_coarse) / q_fine
    support_convergence = max(
        abs(float(weak_fine["inner_radius"]) - float(weak_coarse["inner_radius"])),
        abs(float(weak_fine["outer_radius"]) - float(weak_coarse["outer_radius"])),
    )

    compact_support = all(
        int(row["support_transitions"]) == 2
        and float(row["inner_radius"]) > 0.1
        and float(row["outer_radius"]) < 79.0
        for row in branch
    )
    causal_and_all_ec = all(
        float(row["max_compactness"]) < 2.0 / 3.0
        and float(row["minimum_normalized_margin"]) > 0.0
        and 0.0 < float(row["normalized_energy_cutoff"]) < 1.0
        for row in branch
    )
    field_equations_resolved = all(
        float(row["mass_integral_residual"]) < 1.0e-5
        and float(row["y_integral_residual"]) < 1.0e-6
        and float(row["exterior_schwarzschild_residual"]) < 2.0e-9
        and float(row["exterior_mass_residual"]) < 2.0e-9
        for row in branch
    )
    q_values = [float(row["mass_over_outer_radius"]) for row in branch]
    depth_values = [float(row["normalized_core_depth"]) for row in branch]
    compactness_values = [float(row["max_compactness"]) for row in branch]
    weak_field_monotone = (
        all(q_values[i + 1] < q_values[i] for i in range(len(q_values) - 1))
        and all(depth_values[i + 1] < depth_values[i] for i in range(len(depth_values) - 1))
        and all(
            compactness_values[i + 1] < compactness_values[i]
            for i in range(len(compactness_values) - 1)
        )
    )
    # Negative control: re-solve the same model with the angular-momentum
    # cutoff removed (L0 = 0) and check that the central vacuum barrier
    # actually disappears.  The reference leg is the same central_y at the same
    # numerics with L0 = ANGULAR_CUTOFF, so L0 is the only variable.
    hollow_reference, zero_cutoff = zero_cutoff_control(CENTRAL_Y[-1])
    hollow_inner_radius = float(hollow_reference["inner_radius"])
    filled_inner_radius = float(zero_cutoff["inner_radius"])
    zero_cutoff_fills_center = (
        hollow_inner_radius > HOLLOW_MIN_INNER_RADIUS
        and int(hollow_reference["support_transitions"]) == 2
        and not bool(hollow_reference["support_reaches_first_grid_point"])
        and float(hollow_reference["central_density"]) == 0.0
        and bool(zero_cutoff["support_reaches_first_grid_point"])
        and filled_inner_radius < ZERO_CUTOFF_MAX_INNER_RADIUS
        # Not a bare "> 0.0": the central density must clear a predeclared
        # floor relative to the same solution's peak density, so that float
        # noise cannot masquerade as a filled center.
        and float(zero_cutoff["central_over_peak_density"])
        > CENTRAL_DENSITY_NOISE_FLOOR_RELATIVE
        and int(zero_cutoff["support_transitions"]) == 1
    )
    fixed_radius_mass_reduction = q_values[0] / q_values[-1]

    passed = (
        compact_support
        and causal_and_all_ec
        and field_equations_resolved
        and weak_field_monotone
        and zero_cutoff_fills_center
        and q_convergence < 1.0e-5
        and support_convergence < 5.0e-4
    )
    payload = {
        "constitutive_family": {
            "form": "A*(E0-E)_+^k*(L-L0)_+^ell",
            "amplitude": AMPLITUDE,
            "angular_cutoff": ANGULAR_CUTOFF,
            "energy_power": 1,
            "angular_power": 1,
        },
        "branch": branch,
        "weak_endpoint_convergence": {
            "coarse": weak_coarse,
            "fine": weak_fine,
            "relative_mass_over_radius_change": q_convergence,
            "absolute_support_change": support_convergence,
        },
        "zero_cutoff_negative_control": {
            "description": (
                "two real Einstein-Vlasov solves at identical numerics, "
                "differing only in the angular-momentum cutoff, showing that "
                "L0=0 removes the central vacuum barrier and fills the center"
            ),
            "numerics": {
                "quadrature_order": CONTROL_NUMERICS[0],
                "radial_points": CONTROL_NUMERICS[1],
                "relative_tolerance": CONTROL_NUMERICS[2],
                "maximum_step": CONTROL_NUMERICS[3],
            },
            "thresholds": {
                "hollow_min_inner_radius": HOLLOW_MIN_INNER_RADIUS,
                "zero_cutoff_max_inner_radius": ZERO_CUTOFF_MAX_INNER_RADIUS,
                "central_density_noise_floor_relative": (
                    CENTRAL_DENSITY_NOISE_FLOOR_RELATIVE
                ),
            },
            "hollow_reference": hollow_reference,
            "zero_cutoff": zero_cutoff,
        },
        "certificates": {
            "compact_hollow_support": compact_support,
            "causal_and_all_energy_conditions": causal_and_all_ec,
            "field_equations_resolved": field_equations_resolved,
            "weak_field_monotone": weak_field_monotone,
            "zero_angular_cutoff_fills_center": zero_cutoff_fills_center,
            "fixed_outer_radius_mass_reduction_factor": fixed_radius_mass_reduction,
            "stability_established": False,
            "supports_or_confinement_included": False,
            "verdict": "PASS" if passed else "FAIL",
        },
    }
    # Fail-closed artifact policy (see CORRECTIONS_LEDGER.md C68).  The
    # canonical path is consumed downstream by verify_engineering_limits.py, so
    # it is written ONLY after the pass decision.  A failing run instead leaves
    # its diagnostics on a distinct, unmistakably named path that no consumer
    # reads, and exits nonzero; it never publishes a verdict:"FAIL" document to
    # the canonical name.
    results_dir = Path(__file__).resolve().parents[1] / "results"
    canonical_output = results_dir / "vlasov_constitutive_branch.json"
    failed_output = results_dir / "vlasov_constitutive_branch.FAILED.json"
    document = json.dumps(payload, indent=2) + "\n"

    for row in branch:
        print(
            f"yc={row['central_y']:.2f} "
            f"support=[{row['inner_radius']:.8f},{row['outer_radius']:.8f}] "
            f"M/Rout={row['mass_over_outer_radius']:.10f} "
            f"Cmax={row['max_compactness']:.10f} "
            f"depth={row['normalized_core_depth']:.10f} "
            f"ECmargin={row['minimum_normalized_margin']:.10f}"
        )
    print(f"weak-endpoint q relative change = {q_convergence:.3e}")
    print(f"weak-endpoint support absolute change = {support_convergence:.3e}")
    print(f"fixed-R mass reduction factor = {fixed_radius_mass_reduction:.8f}")
    print(
        f"L0=0 control: inner support edge {hollow_inner_radius:.8f} -> "
        f"{filled_inner_radius:.8f} "
        f"(transitions {int(hollow_reference['support_transitions'])} -> "
        f"{int(zero_cutoff['support_transitions'])}, "
        f"central density {zero_cutoff['central_density']:.6e}); "
        f"center filled = {zero_cutoff_fills_center}"
    )
    print("stability/supports/confinement established = False")
    print("VERDICT: PASS" if passed else "VERDICT: FAIL")

    if not passed:
        failed_output.write_text(document, encoding="utf-8")
        print(
            f"CHECK_FAIL: certificates did not all hold; diagnostics written to "
            f"{failed_output.name} and the canonical artifact "
            f"{canonical_output.name} was NOT written or updated"
        )
        raise SystemExit(1)

    canonical_output.write_text(document, encoding="utf-8")
    # A passing run supersedes any stale failed-run record.
    failed_output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
