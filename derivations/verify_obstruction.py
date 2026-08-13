"""Independent symbolic and numerical checks for the hollow-core obstruction.

The Einstein tensor is computed directly from the metric

    ds^2 = -dt^2 + (dr - beta(r) dt)^2 + r^2 dOmega^2

from the metric rather than imported identities.  The script then checks the
energy-condition relations, demonstrates the smooth-onset obstruction on a
representative profile, and generates the manuscript figure and a JSON record.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def canonical(expr: sp.Expr) -> sp.Expr:
    return sp.factor(sp.cancel(sp.together(sp.simplify(expr))))


def assert_zero(label: str, expr: sp.Expr, checks: dict[str, object]) -> None:
    reduced = canonical(expr)
    ok = reduced == 0
    checks[label] = {"passed": bool(ok), "remainder": str(reduced)}
    if not ok:
        raise AssertionError(f"{label}: {reduced}")


def einstein_tensor(metric: sp.Matrix, coords: list[sp.Symbol]) -> sp.Matrix:
    """Return G_{mu nu} from a coordinate metric."""

    dim = len(coords)
    inverse = sp.simplify(metric.inv())
    gamma = [[[
        canonical(
            sp.Rational(1, 2)
            * sum(
                inverse[a, d]
                * (
                    sp.diff(metric[d, c], coords[b])
                    + sp.diff(metric[d, b], coords[c])
                    - sp.diff(metric[b, c], coords[d])
                )
                for d in range(dim)
            )
        )
        for c in range(dim)] for b in range(dim)] for a in range(dim)]

    ricci = sp.MutableDenseMatrix.zeros(dim, dim)
    for a in range(dim):
        for b in range(dim):
            value = 0
            for c in range(dim):
                value += sp.diff(gamma[c][a][b], coords[c])
                value -= sp.diff(gamma[c][a][c], coords[b])
                for d in range(dim):
                    value += gamma[c][c][d] * gamma[d][a][b]
                    value -= gamma[c][b][d] * gamma[d][a][c]
            ricci[a, b] = canonical(value)

    scalar = canonical(sum(inverse[a, b] * ricci[a, b]
                           for a in range(dim) for b in range(dim)))
    return sp.Matrix(dim, dim, lambda a, b: canonical(
        ricci[a, b] - sp.Rational(1, 2) * metric[a, b] * scalar
    ))


def symbolic_checks() -> tuple[dict[str, object], dict[str, str]]:
    t, r, theta, phi = sp.symbols("t r theta phi", real=True, positive=True)
    beta = sp.Function("beta")(r)
    coords = [t, r, theta, phi]
    metric = sp.Matrix([
        [beta**2 - 1, -beta, 0, 0],
        [-beta, 1, 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta)**2],
    ])

    einstein = einstein_tensor(metric, coords)
    normal = sp.Matrix([1, beta, 0, 0])
    radial = sp.Matrix([0, 1, 0, 0])

    rho = canonical((normal.T * einstein * normal)[0] / (8 * sp.pi))
    p_r = canonical((radial.T * einstein * radial)[0] / (8 * sp.pi))
    p_perp = canonical(einstein[2, 2] / (8 * sp.pi * r**2))
    flux = canonical((normal.T * einstein * radial)[0] / (8 * sp.pi))
    w = r * beta**2

    checks: dict[str, object] = {}
    assert_zero("rho_from_mass_function", rho - sp.diff(w, r)/(8*sp.pi*r**2), checks)
    assert_zero("radial_equation_of_state", p_r + rho, checks)
    assert_zero("zero_energy_flux", flux, checks)
    assert_zero("transverse_pressure_density_form",
                p_perp + rho + r*sp.diff(rho, r)/2, checks)
    assert_zero("transverse_pressure_mass_form",
                p_perp + sp.diff(w, r, 2)/(16*sp.pi*r), checks)
    assert_zero("transverse_nec_margin",
                rho + p_perp + r*sp.diff(rho, r)/2, checks)

    generic_rho = sp.Function("rho")(r)
    generic_p_perp = -generic_rho - r*sp.diff(generic_rho, r)/2
    s = -r*sp.diff(generic_rho, r)/generic_rho
    assert_zero("slope_form", generic_p_perp - (s-2)*generic_rho/2, checks)

    expressions = {
        "rho": str(rho),
        "p_r": str(p_r),
        "p_perp": str(p_perp),
        "flux": str(flux),
        "w": str(w),
    }
    return checks, expressions


def smootherstep(x: np.ndarray) -> np.ndarray:
    return x**3 * (10 - 15*x + 6*x**2)


def smootherstep_derivative(x: np.ndarray) -> np.ndarray:
    return 30*x**2 * (1-x)**2


def numerical_profiles() -> dict[str, float]:
    r = np.linspace(0.02, 6.0, 30001)
    r1 = 1.0
    width = 1.0
    decay = 2.5
    x = (r-r1)/width

    rho_hollow = np.zeros_like(r)
    onset = (x > 0) & (x < 1)
    tail = x >= 1
    rho_hollow[onset] = smootherstep(x[onset]) * np.exp(-(r[onset]-r1)/decay)
    rho_hollow[tail] = np.exp(-(r[tail]-r1)/decay)

    rho_filled = np.exp(-r/decay)
    drho_hollow = np.zeros_like(r)
    onset_factor = np.exp(-(r[onset]-r1)/decay)
    drho_hollow[onset] = onset_factor * (
        smootherstep_derivative(x[onset])/width
        - smootherstep(x[onset])/decay
    )
    drho_hollow[tail] = -rho_hollow[tail]/decay
    drho_filled = -rho_filled/decay
    nec_hollow = -0.5*r*drho_hollow
    nec_filled = -0.5*r*drho_filled

    active = (r > r1 + 5e-3) & (r < r1 + width - 5e-3)
    minimum_hollow = float(np.min(nec_hollow[active]))
    minimum_filled = float(np.min(nec_filled))
    if not minimum_hollow < -1e-3:
        raise AssertionError("smooth hollow onset did not produce the expected NEC deficit")
    if not minimum_filled >= -1e-8:
        raise AssertionError("monotone filled profile unexpectedly violated NEC")

    peak_index = int(np.argmax(rho_hollow))
    start_index = int(np.searchsorted(r, r1, side="left"))
    budget_slice = slice(start_index, peak_index + 1)
    budget_r = r[budget_slice]
    budget_integrand = 2*nec_hollow[budget_slice]/budget_r
    weighted_budget = float(np.sum(
        0.5*(budget_integrand[1:] + budget_integrand[:-1]) * np.diff(budget_r)
    ))
    expected_budget = float(
        rho_hollow[start_index] - rho_hollow[peak_index]
    )
    budget_error = abs(weighted_budget - expected_budget)
    depth_bound = float(
        rho_hollow[peak_index] / (2*np.log(r[peak_index]/r1))
    )
    observed_depth = float(-np.min(nec_hollow[budget_slice]))
    if budget_error > 1e-6:
        raise AssertionError(f"onset budget error too large: {budget_error}")
    if observed_depth + 1e-10 < depth_bound:
        raise AssertionError("smooth hollow profile violated the analytic depth bound")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                             constrained_layout=True)
    # Grayscale plus redundant line/marker coding keeps the figure legible in
    # monochrome print and without relying on color discrimination.
    styles = {
        "hollow": dict(color="0.10", ls="-", marker="o", markevery=95),
        "filled": dict(color="0.48", ls="--", marker="s", markevery=110),
    }
    axes[0].plot(r, rho_hollow, lw=2.2, mfc="white", ms=4.0, **styles["hollow"],
                 label="smooth hollow onset")
    axes[0].plot(r, rho_filled, lw=2.0, mfc="white", ms=3.8, **styles["filled"],
                 label="filled monotone profile")
    axes[0].axvspan(0, r1, facecolor="0.93", edgecolor="0.72", hatch="....",
                    label="flat hollow core")
    axes[0].set_ylabel(r"normalized $\rho$")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[0].set_title("A smooth positive onset must cross a transverse NEC deficit")

    axes[1].plot(r, nec_hollow, lw=2.2, mfc="white", ms=4.0, **styles["hollow"])
    axes[1].plot(r, nec_filled, lw=2.0, mfc="white", ms=3.8, **styles["filled"])
    axes[1].axhline(0, color="black", lw=0.9)
    axes[1].fill_between(r, nec_hollow, 0, where=nec_hollow < 0,
                         facecolor="0.86", edgecolor="0.40", hatch="////",
                         label="NEC/WEC violation")
    axes[1].set_xlabel(r"radius $r/R_1$ (with $R_1=1$)")
    axes[1].set_ylabel(r"$\rho+p_\perp=-r\rho'/2$")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_ylim(-1.25, 0.7)

    fig.savefig(FIGURES / "onset_obstruction.pdf")
    fig.savefig(FIGURES / "onset_obstruction.png", dpi=220)
    plt.close(fig)

    return {
        "minimum_transverse_nec_smooth_hollow": minimum_hollow,
        "minimum_transverse_nec_filled_monotone": minimum_filled,
        "grid_points": int(r.size),
        "R1": r1,
        "onset_width": width,
        "decay_length": decay,
        "peak_radius": float(r[peak_index]),
        "peak_density": float(rho_hollow[peak_index]),
        "weighted_onset_budget": weighted_budget,
        "expected_onset_budget": expected_budget,
        "onset_budget_absolute_error": budget_error,
        "minimum_depth_bound": depth_bound,
        "observed_deficit_depth": observed_depth,
    }


def main() -> None:
    checks, expressions = symbolic_checks()
    numerical = numerical_profiles()
    payload = {
        "status": "PASS",
        "symbolic_checks": checks,
        "expressions": expressions,
        "numerical_profiles": numerical,
        "scope": "static spherical unit-lapse flat-slice radial-shift class",
    }
    output = RESULTS / "verification.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"PASS: {len(checks)} symbolic identities")
    print(f"PASS: smooth hollow min transverse NEC = {numerical['minimum_transverse_nec_smooth_hollow']:.6g}")
    print(f"PASS: filled monotone min transverse NEC = {numerical['minimum_transverse_nec_filled_monotone']:.6g}")
    print(f"Wrote {output}")
    print(f"Wrote {FIGURES / 'onset_obstruction.pdf'}")


if __name__ == "__main__":
    main()
