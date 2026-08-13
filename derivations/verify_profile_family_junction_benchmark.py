"""Certify the hollow-shell family, uniqueness, material robustness, and benchmark.

The calculation is deliberately redundant: exact SymPy identities certify the
endpoint regularity, minimum-degree uniqueness, and the S9 weak-field
coefficient.  Independent floating-point calculations map the admissible
family, the causal Einstein-cluster subdomain, a sufficient open radial-pressure
neighborhood, and the invariant core depth.  The last panel reports the
dimensionless homological exponents controlling energy, density, and signal.
Instrument-specific values remain in the machine-readable output for the
separate apparatus proposal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

G_SI = 6.67430e-11
C_SI = 299_792_458.0
HBAR_SI = 1.054_571_817e-34
ATOMIC_MASS_SI = 1.660_539_066_60e-27
CS133_MASS_SI = 132.905_451_9610 * ATOMIC_MASS_SI

# Predeclared verifier tolerances.  These are fixed budgets, not quantities
# fitted to the observed output; a wrong coefficient must break them.
#
# WEAK_FIELD_REMAINDER_BOUND bounds |(-Phi_c - kappa mu)/mu^2| uniformly over
# the whole mu ladder.  The exact core depth is analytic in mu at mu=0, so the
# ratio tends to a finite (n,eta)-dependent plateau of order 10^-1.  The bound
# is set to 2.0, a round number several times any such plateau, so that an
# error delta in the linear coefficient kappa -- which contributes delta/mu and
# therefore at least delta/3e-4 on this ladder -- is caught.  Do NOT retune this
# to whatever the current run happens to print.
WEAK_FIELD_REMAINDER_BOUND = 2.0

# Predeclared weak-field mu ladder used by both weak-field gates below.
WEAK_FIELD_MU_LADDER = (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4)

# WEAK_FIELD_KAPPA_RECOVERY_REL_TOL is the second, sharp weak-field gate.
#
# WEAK_FIELD_REMAINDER_BOUND above is a coarse sanity bound and is deliberately
# left at its ledgered value (C68, marked do-not-retune), but it is blind to
# small errors in kappa: an error delta (relative) in the linear coefficient
# shifts |remainder/mu^2| by only delta*kappa/mu, so against a bound of 2.0 and
# plateaus of order 0.1-0.4 it cannot resolve delta below ~2e-5.  kappa is
# quoted to twelve significant digits downstream, so that floor does not protect
# the published precision.
#
# The sharp gate instead RECOVERS kappa from the exact core depths alone and
# compares it with kappa_numeric.  Because the recovery uses no property of
# kappa_numeric, a relative error delta in kappa appears in the comparison as
# delta itself: the gate's sensitivity floor equals this tolerance directly.
#
# Error budget for the recovery (this is why the number is what it is; it is
# NOT fitted to the observed residual):
#   * truncation.  -Phi_c(mu)/mu = kappa + c1 mu + c2 mu^2 + O(mu^3) is analytic
#     at mu=0; evaluating the quadratic interpolant through the three smallest
#     ladder points at mu=0 leaves O(c3 mu1 mu2 mu3) = O(1)*3e-14 absolute,
#     i.e. ~6e-14 relative at kappa ~ 0.5.
#   * quadrature noise.  core_log_lapse requests epsrel=2e-13, and the
#     extrapolation weights at mu=0 for this ladder are (1.667, -0.714, 0.048),
#     an amplification of sum|w| = 2.43, giving ~5e-13 relative.
# The dominant term is the requested quadrature tolerance, ~5e-13.  The gate is
# set two decades above that conservative budget, at a round 1e-10.  Do NOT
# retune this to whatever the current run happens to print.
WEAK_FIELD_KAPPA_RECOVERY_REL_TOL = 1.0e-10

# Absolute tolerance for the numerically maximized compactness max(2 m(r)/r)
# against its analytic endpoint values 0.8 (DEC) and 2/3 (kinetic).  The
# maximizer is a smooth interior stationary point sampled on a uniform grid of
# COMPACTNESS_SAMPLES nodes, so the sampling deficit is O(dy^2) ~ 1e-9.
COMPACTNESS_TOLERANCE = 1.0e-6
COMPACTNESS_SAMPLES = 100_001

# Analytic endpoint compactnesses of the static Einstein-cluster (p_r = 0)
# shell, each the saturation point of a physical condition -- NOT free choices:
#   * DEC.  p_t/rho = m/[2(r-2m)]; rho >= |p_t| saturates at 2m/r = 4/5.
#   * Causality.  v^2 = 2 p_t/rho = m/(r-2m); v^2 <= 1 saturates at 2m/r = 2/3.
# The verifier evaluates both conditions on the actual profile at mu_DEC and
# mu_kinetic and confirms these saturation values, rather than reading them back
# out of the definitions mu_DEC = 0.4/q and mu_kinetic = 1/(3q).
DEC_COMPACTNESS = 4.0 / 5.0
KINETIC_COMPACTNESS = 2.0 / 3.0

# Tolerance on the saturated constitutive ratios (p_t/rho at mu_DEC and v^2 at
# mu_kinetic, both exactly 1).  Those ratios are functions of the sampled
# compactness z = m/r through 1/(1-2z) factors whose derivative is largest at
# the DEC endpoint, d(v^2)/dz = (1-2z)^-2 = 25, so a compactness sampling
# deficit bounded by COMPACTNESS_TOLERANCE propagates to at most
# 25*COMPACTNESS_TOLERANCE/2 = 1.25e-5.  Rounded up to 1e-4.
SATURATION_TOLERANCE = 1.0e-4

# Relative step above mu_DEC at which DEC must be found violated, establishing
# that 0.4/q is the saturation point and not merely an admissible value.  The
# probe stays horizon-free (2m/r = 0.8008 < 1) by construction.
DEC_ENDPOINT_PROBE_STEP = 1.0e-3


def require(condition: object, message: str) -> None:
    """Fail-closed assertion that survives `python -O` (bare asserts do not)."""
    if not condition:
        raise AssertionError(message)


def beta_profile(n: int, x: np.ndarray | float) -> np.ndarray | float:
    """Regularized incomplete-beta profile I_x(n+1,n+1)."""
    from scipy.special import betainc

    return betainc(n + 1, n + 1, x)


def beta_profile_derivative_y(
    n: int, eta: float, y: np.ndarray | float
) -> np.ndarray | float:
    """Derivative d F_n((y-1)/(eta-1))/dy, including endpoint limits."""
    x = (np.asarray(y) - 1.0) / (eta - 1.0)
    normalization = math.factorial(2 * n + 1) / math.factorial(n) ** 2
    value = normalization * x**n * (1.0 - x) ** n / (eta - 1.0)
    return float(value) if np.ndim(value) == 0 else value


def positive_quadratic_root(linear: float, quadratic: float, rhs: float) -> float:
    """Positive root of quadratic*x^2 + linear*x = rhs."""
    if rhs <= 0.0:
        raise ValueError("A strict positive margin is required.")
    if quadratic <= 1.0e-30:
        return rhs / linear if linear > 0.0 else math.inf
    return 2.0 * rhs / (linear + math.sqrt(linear**2 + 4.0 * quadratic * rhs))


def radial_pressure_certificate(n: int, eta: float, mu: float) -> dict[str, object]:
    """Numerically check the analytic p_r=epsilon*rho*x*(1-x) witness."""
    y = np.linspace(1.0, eta, 40_001)
    x = (y - 1.0) / (eta - 1.0)
    shape = np.asarray(beta_profile(n, x), dtype=float)
    shape_y = np.asarray(beta_profile_derivative_y(n, eta, y), dtype=float)
    s = x * (1.0 - x)
    denominator = y - 2.0 * mu * shape
    if np.min(denominator) <= 0.0:
        raise ValueError("The requested certificate is not horizon-free.")

    h0 = mu * shape / denominator
    h1 = mu * y * shape_y * s / denominator
    d_term = y * (n + 1.0) * (1.0 - 2.0 * x) / (eta - 1.0) - 2.0 * s
    a_linear = s + 0.5 * d_term + 0.5 * (s * h0 + h1)
    b_quadratic = 0.5 * s * h1

    b0_max = float(np.max(0.5 * h0))
    norm_a = float(np.max(np.abs(a_linear)))
    norm_b = float(np.max(b_quadratic))
    dec_root = positive_quadratic_root(norm_a, norm_b, 1.0 - b0_max)
    sec_root = positive_quadratic_root(2.0 * norm_a + 0.25, 2.0 * norm_b, 1.0)
    epsilon_witness = 0.99 * min(4.0, dec_root, sec_root)
    if not epsilon_witness > 0.0:
        raise AssertionError("The open radial-pressure neighborhood collapsed.")

    sampled_margins: dict[str, dict[str, float]] = {}
    for sign in (-1.0, 1.0):
        epsilon = sign * epsilon_witness
        pressure_r_ratio = epsilon * s
        pressure_t_ratio = 0.5 * h0 + epsilon * a_linear + epsilon**2 * b_quadratic
        margins = {
            "radial_nec": float(np.min(1.0 + pressure_r_ratio)),
            "tangential_nec": float(np.min(1.0 + pressure_t_ratio)),
            "sec": float(np.min(1.0 + pressure_r_ratio + 2.0 * pressure_t_ratio)),
            "radial_dec": float(np.min(1.0 - np.abs(pressure_r_ratio))),
            "tangential_dec": float(np.min(1.0 - np.abs(pressure_t_ratio))),
        }
        if min(margins.values()) <= 0.0:
            raise AssertionError(margins)
        sampled_margins[f"epsilon_{sign:+.0f}"] = margins

    return {
        "n": n,
        "eta": eta,
        "mu": mu,
        "compactness": 2.0 * mu * q_max(n, eta)[0],
        "b0_max": b0_max,
        "A_sup": norm_a,
        "B_sup": norm_b,
        "epsilon_witness": epsilon_witness,
        "sampled_endpoint_and_wall_margins": sampled_margins,
        "junction_data": "p_r and d p_r/dr vanish at x=0,1 for n>=1",
    }


def q_max(n: int, eta: float) -> tuple[float, float]:
    """Return max F_n((y-1)/(eta-1))/y and its location."""

    def objective(y: float) -> float:
        x = (y - 1.0) / (eta - 1.0)
        return -float(beta_profile(n, x)) / y

    result = minimize_scalar(
        objective,
        bounds=(1.0, eta),
        method="bounded",
        options={"xatol": 2.0e-14},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return -float(result.fun), float(result.x)


def max_compactness(n: int, eta: float, mu: float) -> float:
    """Numerically maximize 2 m(r)/r = 2 mu F_n((y-1)/(eta-1))/y over the wall.

    This is deliberately independent of ``q_max``: it resamples the profile on
    a dense uniform grid instead of reusing the bounded optimizer, so that the
    analytic endpoint compactnesses 0.8 and 2/3 are confirmed by a second
    calculation rather than reproduced by construction.
    """
    y = np.linspace(1.0, eta, COMPACTNESS_SAMPLES)
    shape = np.asarray(beta_profile(n, (y - 1.0) / (eta - 1.0)), dtype=float)
    return float(np.max(2.0 * mu * shape / y))


def profile_bounds(n: int, eta: float, mu: float) -> dict[str, float]:
    """Evaluate the shell's physical bounds directly from the profile at mu.

    Every quantity here is computed from the sampled mass function, so nothing
    is inherited from the definitions ``mu_DEC = 0.4/q`` or
    ``mu_kinetic = 1/(3q)``.  That is what makes the endpoint values 4/5 and 2/3
    certified rather than assumed: the caller supplies a mu, and this function
    reports what the energy condition, the particle speed, and the horizon
    function actually do there.

    For the static Einstein-cluster branch (p_r = 0, Florides lapse) the
    constitutive relations are p_t/rho = m/[2(r-2m)] and v^2 = 2 p_t/rho, both
    strictly increasing in the local compactness z = m/r, so their maxima sit at
    the same radius as max 2m/r.  Outside the wall 2m/r = 2 mu/y decreases and
    inside the cavity m = 0, so the wall carries the global maximum.
    """
    y = np.linspace(1.0, eta, COMPACTNESS_SAMPLES)
    shape = np.asarray(beta_profile(n, (y - 1.0) / (eta - 1.0)), dtype=float)
    mass_over_radius = mu * shape / y
    horizon_function = 1.0 - 2.0 * mass_over_radius
    if float(np.min(horizon_function)) <= 0.0:
        raise AssertionError(
            f"n={n}, eta={eta}, mu={mu}: 1-2m/r reaches "
            f"{float(np.min(horizon_function)):.6g}; the configuration is not "
            "horizon-free and its constitutive ratios are undefined."
        )
    tangential_over_density = mass_over_radius / (2.0 * horizon_function)
    return {
        "max_2m_over_r": float(np.max(2.0 * mass_over_radius)),
        "min_horizon_function": float(np.min(horizon_function)),
        "max_tangential_pressure_over_density": float(np.max(tangential_over_density)),
        "max_speed_squared": float(np.max(2.0 * tangential_over_density)),
    }


def extrapolated_kappa(mus: tuple[float, ...], depths: tuple[float, ...]) -> float:
    """Recover the linear coefficient of -Phi_c(mu) from the depths alone.

    ``-Phi_c(mu)/mu = kappa + c1 mu + c2 mu^2 + O(mu^3)`` is analytic at mu = 0.
    Evaluating the Lagrange interpolant of ``-Phi_c/mu`` through three ladder
    points at ``mu = 0`` therefore returns kappa with the c1 and c2 terms
    eliminated.  No property of ``kappa_numeric`` is used, so this is an
    independent measurement of the coefficient rather than a restatement of it.
    """
    if len(mus) != 3 or len(depths) != 3:
        raise AssertionError("the kappa recovery needs exactly three ladder points")
    recovered = 0.0
    for i, (mu_i, depth_i) in enumerate(zip(mus, depths)):
        weight = 1.0
        for j, mu_j in enumerate(mus):
            if j != i:
                weight *= -mu_j / (mu_i - mu_j)
        recovered += weight * depth_i / mu_i
    return recovered


def core_log_lapse(mu: float, n: int, eta: float) -> float:
    q, _ = q_max(n, eta)
    if 2.0 * mu * q >= 1.0:
        raise ValueError("The shell is not horizon-free.")

    def integrand(y: float) -> float:
        f = float(beta_profile(n, (y - 1.0) / (eta - 1.0)))
        return mu * f / (y * (y - 2.0 * mu * f))

    wall, error = quad(integrand, 1.0, eta, epsabs=2e-14, epsrel=2e-13)
    if error > 2.0e-11:
        raise RuntimeError(f"Unexpected quadrature error {error}")
    return 0.5 * math.log1p(-2.0 * mu / eta) - wall


def kappa_numeric(n: int, eta: float) -> float:
    value, error = quad(
        lambda y: float(beta_profile(n, (y - 1.0) / (eta - 1.0))) / y**2,
        1.0,
        eta,
        epsabs=2e-14,
        epsrel=2e-13,
    )
    if error > 2.0e-11:
        raise RuntimeError(f"Unexpected quadrature error {error}")
    return 1.0 / eta + value


def ideal_cs_phase(
    mass_kg: float,
    compact_length_m: float = 5.0e-3,
    expanded_length_m: float = 20.0e-3,
    dwell_time_s: float = 0.5,
) -> float:
    """Leading fixed-ADM core dwell phase for Cs-133."""
    kappa = kappa_numeric(4, 3.0)
    return (
        kappa
        * G_SI
        * mass_kg
        * CS133_MASS_SI
        * dwell_time_s
        / HBAR_SI
        * (1.0 / compact_length_m - 1.0 / expanded_length_m)
    )


def homology_scaling_certificate() -> dict[str, object]:
    """Derive the scale exponents and the microscopic no-free-lunch result."""
    s = sp.symbols("s", positive=True)
    p = sp.symbols("p", real=True)
    mass_0, length_0 = sp.symbols("M_0 L_0", positive=True)
    mass = s**p * mass_0
    length = s * length_0
    ratios = {
        "ADM_energy": sp.simplify(mass / mass_0),
        "mean_density": sp.simplify((mass / length**3) / (mass_0 / length_0**3)),
        "compactness": sp.simplify((mass / length) / (mass_0 / length_0)),
        "weak_core_depth_or_phase": sp.simplify(
            (mass / length) / (mass_0 / length_0)
        ),
    }
    expected = {
        "ADM_energy": s**p,
        "mean_density": s ** (p - 3),
        "compactness": s ** (p - 1),
        "weak_core_depth_or_phase": s ** (p - 1),
    }
    identities = {
        name: sp.simplify(ratios[name] - expected[name]) == 0 for name in ratios
    }
    feasible = sp.reduce_inequalities([p > 0, p >= 3, p <= 1], p)
    return {
        "definition": "L -> s L, M_ADM -> s^p M_ADM as s -> 0+",
        "exponents": {
            "ADM_energy": "p",
            "mean_density": "p-3",
            "compactness": "p-1",
            "weak_core_depth_or_fixed_probe_phase": "p-1",
        },
        "identities": identities,
        "paths": {
            "fixed_compactness_p1": {
                "energy": "s",
                "mean_density": "s^-2",
                "core_depth": "constant",
            },
            "bounded_density_p3": {
                "energy": "s^3",
                "mean_density": "constant",
                "weak_core_depth_or_phase": "s^2",
            },
            "fixed_mass_p0": {
                "energy": "constant",
                "mean_density": "s^-3",
                "compactness": "s^-1 until the admissibility bound",
            },
        },
        "requirements": {
            "decreasing_energy": "p>0",
            "bounded_mean_density": "p>=3",
            "nonvanishing_weak_signal": "p<=1",
        },
        "joint_requirements_feasible": feasible is not sp.false,
    }

def symbolic_checks() -> dict[str, object]:
    x, u = sp.symbols("x u", real=True)
    endpoint_checks: dict[str, object] = {}
    profiles_by_n: dict[int, sp.Expr] = {}
    for n in range(1, 9):
        beta = sp.factorial(n) ** 2 / sp.factorial(2 * n + 1)
        profile = sp.expand(sp.integrate(u**n * (1 - u) ** n, (u, 0, x)) / beta)
        profiles_by_n[n] = profile
        derivative_target = sp.expand(x**n * (1 - x) ** n / beta)
        require(
            sp.simplify(sp.diff(profile, x) - derivative_target) == 0,
            f"n={n}: dF_n/dx does not reproduce the incomplete-beta integrand.",
        )
        zero_derivatives = all(
            sp.diff(profile, x, order).subs(x, endpoint) == (1 if order == 0 and endpoint == 1 else 0)
            for order in range(n + 1)
            for endpoint in (0, 1)
        )
        first_unmatched = (
            sp.diff(profile, x, n + 1).subs(x, 0),
            sp.diff(profile, x, n + 1).subs(x, 1),
        )
        require(zero_derivatives, f"n={n}: endpoint jets through order n are not matched.")
        require(
            all(value != 0 for value in first_unmatched),
            f"n={n}: order n+1 endpoint derivative vanishes, so F_n would be C^(n+1).",
        )

        # The 2n+2 endpoint-jet constraints have full rank on polynomials of
        # degree <=2n+1.  Hence the displayed beta polynomial is the unique
        # minimum-degree interpolant, not just one convenient choice.
        coefficients = sp.symbols(f"c0:{2 * n + 2}")
        generic = sum(coefficients[power] * x**power for power in range(2 * n + 2))
        constraints = [generic.subs(x, 0), generic.subs(x, 1)]
        for order in range(1, n + 1):
            constraints.extend(
                [sp.diff(generic, x, order).subs(x, 0), sp.diff(generic, x, order).subs(x, 1)]
            )
        constraint_matrix, _ = sp.linear_eq_to_matrix(constraints, coefficients)
        uniqueness_rank = int(constraint_matrix.rank())
        require(
            uniqueness_rank == 2 * n + 2,
            f"n={n}: endpoint-jet constraint matrix has rank {uniqueness_rank}, "
            f"expected {2 * n + 2}; minimum-degree uniqueness would fail.",
        )
        radial_pressure_shape = x ** (n + 1) * (1 - x) ** (n + 1)
        radial_pressure_jets = all(
            sp.diff(radial_pressure_shape, x, order).subs(x, endpoint) == 0
            for order in (0, 1)
            for endpoint in (0, 1)
        )
        require(
            radial_pressure_jets,
            f"n={n}: p_r shape or its first derivative fails to vanish at an interface.",
        )
        endpoint_checks[str(n)] = {
            "degree": int(sp.degree(profile, x)),
            "C_n_endpoint_match": True,
            "not_C_n_plus_1": True,
            "minimal_degree_unique": True,
            "endpoint_constraint_rank": uniqueness_rank,
            "radial_pressure_and_first_derivative_zero_at_interfaces": radial_pressure_jets,
            "first_unmatched": [str(value) for value in first_unmatched],
        }

    # Conditional source reconstruction.  The radial Einstein equation fixes
    # Phi' uniquely once m and p_r are supplied; p_r=0 is its Florides branch.
    r, mass, pressure_r = sp.symbols("r mass pressure_r", positive=True)
    phi_prime = sp.symbols("phi_prime", real=True)
    radial_equation = -2 * mass / r**3 + 2 * (1 - 2 * mass / r) * phi_prime / r
    phi_solutions = sp.solve(sp.Eq(8 * sp.pi * pressure_r, radial_equation), phi_prime)
    # Index-0 guard: the branch is only "the" solution if the solve is unique.
    require(
        len(phi_solutions) == 1,
        f"The radial Einstein equation returned {len(phi_solutions)} branches for "
        "Phi'; conditional uniqueness of the reconstruction would not hold.",
    )
    phi_general = phi_solutions[0]
    phi_zero = sp.simplify(phi_general.subs(pressure_r, 0))
    # Substantive checks behind the reported uniqueness flag: the general branch
    # must satisfy the radial equation identically, and its p_r=0 restriction
    # must be exactly the Florides lapse.
    phi_general_solves = (
        sp.simplify(radial_equation.subs(phi_prime, phi_general) - 8 * sp.pi * pressure_r) == 0
    )
    phi_zero_is_florides = sp.simplify(phi_zero - mass / (r * (r - 2 * mass))) == 0
    conditional_source_verified = bool(phi_general_solves and phi_zero_is_florides)
    require(
        conditional_source_verified,
        "The p_r=0 branch of the radial Einstein equation is not Phi'=m/[r(r-2m)].",
    )

    rho = sp.symbols("rho", positive=True)
    pressure_t_zero = rho * mass / (2 * (r - 2 * mass))
    speed_squared = sp.simplify(2 * pressure_t_zero / rho)
    require(
        sp.simplify(speed_squared - mass / (r - 2 * mass)) == 0,
        "The Einstein-cluster tangential speed is not v^2=m/(r-2m).",
    )

    y = sp.symbols("y", positive=True)
    s9 = sp.expand(
        126 * ((y - 1) / 2) ** 5
        - 420 * ((y - 1) / 2) ** 6
        + 540 * ((y - 1) / 2) ** 7
        - 315 * ((y - 1) / 2) ** 8
        + 70 * ((y - 1) / 2) ** 9
    )
    kappa_exact = sp.simplify(sp.Rational(1, 3) + sp.integrate(s9 / y**2, (y, 1, 3)))
    kappa_target = -sp.Rational(6975, 64) + sp.Rational(25515, 256) * sp.log(3)
    require(
        sp.simplify(kappa_exact - kappa_target) == 0,
        "The exact S9 weak-field coefficient is not -6975/64+(25515/256)ln(3).",
    )

    # Positive-family interfaces.  The SHELL-side data are one-sided limits of
    # the actual incomplete-beta profile; the VACUUM-side data are the interior
    # Minkowski values (m=0, Phi'=0) and the exterior Schwarzschild values
    # (m=Mj, Phi'=Mj/[Rj(Rj-2Mj)]).  The match is therefore a consequence of the
    # endpoint jets of F_n, not an identity of construction: perturbing F_n(0)
    # or F_n(1) moves the shell side away from the vacuum side and this fails.
    Mj, Phij, mprime_left, mprime_right, jet_eps = sp.symbols(
        "Mj Phij mprime_left mprime_right jet_eps", positive=True
    )
    R1, R2 = sp.symbols("R1 R2", positive=True)
    JUNCTION_KEYS = ("h_tt", "h_theta_theta", "K_tt", "K_theta_theta")

    def junction_data(
        radius: sp.Expr, mass_value: sp.Expr, phi_prime_value: sp.Expr, lapse_squared: sp.Expr
    ) -> dict[str, sp.Expr]:
        """Induced metric and extrinsic curvature of the r=radius world tube.

        h_ab = diag(-e^(2 Phi), r^2, ...);
        K_tt = -e^(2 Phi) sqrt(f) Phi', K_theta_theta = r sqrt(f), f = 1-2m/r.
        """
        fj = 1 - 2 * mass_value / radius
        return {
            "h_tt": -lapse_squared,
            "h_theta_theta": radius**2,
            "K_tt": -lapse_squared * sp.sqrt(fj) * phi_prime_value,
            "K_theta_theta": radius * sp.sqrt(fj),
        }

    def florides_phi_prime(mass_value: sp.Expr, radius: sp.Expr) -> sp.Expr:
        """The p_r=0 lapse gradient reconstructed above, evaluated at radius."""
        return phi_zero.subs({mass: mass_value, r: radius})

    junction_reports: dict[str, object] = {}
    positive_junction_pass = True
    classical_stress_continuous = True
    for n, profile in profiles_by_n.items():
        # One-sided limits of the shell mass function m(r)=Mj F_n(x(r)).
        shape_at_inner = sp.limit(profile, x, 0, "+")
        shape_at_outer = sp.limit(profile, x, 1, "-")
        mass_shell_inner = sp.simplify(Mj * shape_at_inner)
        mass_shell_outer = sp.simplify(Mj * shape_at_outer)

        inner_shell = junction_data(
            R1,
            mass_shell_inner,
            florides_phi_prime(mass_shell_inner, R1),
            sp.exp(2 * Phij),
        )
        inner_vacuum = junction_data(R1, sp.Integer(0), sp.Integer(0), sp.exp(2 * Phij))
        outer_shell = junction_data(
            R2,
            mass_shell_outer,
            florides_phi_prime(mass_shell_outer, R2),
            1 - 2 * mass_shell_outer / R2,
        )
        outer_vacuum = junction_data(
            R2, Mj, Mj / (R2 * (R2 - 2 * Mj)), 1 - 2 * Mj / R2
        )
        no_layer = all(
            sp.simplify(side[key] - vacuum[key]) == 0
            for side, vacuum in ((inner_shell, inner_vacuum), (outer_shell, outer_vacuum))
            for key in JUNCTION_KEYS
        )

        # Classical stress continuity is a STRICTLY STRONGER statement than the
        # absence of an Israel layer: it needs the first derivative jets of F_n
        # to vanish, so that rho ~ m'(r) meets the vacuum value 0 at both walls.
        shape_prime = sp.diff(profile, x)
        stress_continuous = (
            sp.limit(shape_prime, x, 0, "+") == 0 and sp.limit(shape_prime, x, 1, "-") == 0
        )

        require(
            no_layer,
            f"n={n}: shell-side junction data do not match the vacuum side "
            "(S_ab != 0); the endpoint values of F_n are wrong.",
        )
        require(
            stress_continuous,
            f"n={n}: F_n' does not vanish at an endpoint, so the classical "
            "density is discontinuous at the wall.",
        )
        positive_junction_pass = positive_junction_pass and no_layer
        classical_stress_continuous = classical_stress_continuous and stress_continuous
        junction_reports[str(n)] = {
            "shell_side_mass_at_inner_wall": str(mass_shell_inner),
            "shell_side_mass_at_outer_wall": str(mass_shell_outer),
            "shell_side_phi_prime_at_outer_wall": str(
                sp.simplify(florides_phi_prime(mass_shell_outer, R2))
            ),
            "matches_vacuum_both_walls": bool(no_layer),
            "classical_density_continuous": bool(stress_continuous),
        }

    # C64: continuity of m alone removes the layer, EVEN IF m' jumps.  Build the
    # one-sided data from mass functions that genuinely carry distinct m' slopes
    # and show the junction quantities lose that dependence in the limit.
    mass_from_inside = Mj - mprime_left * jet_eps
    mass_from_outside = Mj + mprime_right * jet_eps
    side_from_inside = junction_data(
        R2,
        mass_from_inside,
        florides_phi_prime(mass_from_inside, R2),
        1 - 2 * mass_from_inside / R2,
    )
    side_from_outside = junction_data(
        R2,
        mass_from_outside,
        florides_phi_prime(mass_from_outside, R2),
        1 - 2 * mass_from_outside / R2,
    )
    # The slopes really are present before the limit is taken; without this the
    # free-symbol test below would be vacuous.
    slopes_present = all(
        mprime_left in side_from_inside[key].free_symbols
        and mprime_right in side_from_outside[key].free_symbols
        for key in ("h_tt", "K_tt", "K_theta_theta")
    )
    limit_from_inside = {
        key: sp.simplify(sp.limit(value, jet_eps, 0, "+"))
        for key, value in side_from_inside.items()
    }
    limit_from_outside = {
        key: sp.simplify(sp.limit(value, jet_eps, 0, "+"))
        for key, value in side_from_outside.items()
    }
    slopes_absent_after_limit = not any(
        slope in limit_from_inside[key].free_symbols
        or slope in limit_from_outside[key].free_symbols
        for key in JUNCTION_KEYS
        for slope in (mprime_left, mprime_right)
    )
    limits_agree = all(
        sp.simplify(limit_from_inside[key] - limit_from_outside[key]) == 0
        for key in JUNCTION_KEYS
    )
    derivative_independence = bool(slopes_present and slopes_absent_after_limit and limits_agree)
    require(
        positive_junction_pass,
        "At least one profile order fails the positive-family junction match.",
    )
    require(
        derivative_independence,
        "A jump in m' changes the junction data, contradicting C64 (continuity "
        "of m alone removes the Israel layer).",
    )

    # Sharp unit-lapse PG onset.  With one normal pointing from the cavity to
    # increasing r, K^tau_tau=f'/(2 sqrt(f)) and K^theta_theta=sqrt(f)/R.  The
    # two sqrt(f) values below are built from genuinely different mass
    # functions -- an empty cavity and a shell whose density switches on at
    # r=R -- so sigma=0 is a RESULT of m being continuous there.
    R, wp = sp.symbols("R wp", positive=True)
    r_probe = sp.symbols("r_probe", positive=True)
    rho_onset = wp / (8 * sp.pi * R**2)
    mass_cavity = sp.Integer(0)  # m(r)=0 for r<R
    mass_onset = 4 * sp.pi * rho_onset * R**2 * (r_probe - R)  # leading m(r) for r>R
    f_cavity = 1 - 2 * mass_cavity / r_probe
    f_onset = 1 - 2 * mass_onset / r_probe
    f_inner = sp.simplify(sp.limit(f_cavity, r_probe, R, "-"))
    f_outer = sp.simplify(sp.limit(f_onset, r_probe, R, "+"))
    onset_f_matches = bool(sp.simplify(f_inner - f_outer) == 0)
    require(
        onset_f_matches,
        "The sharp onset has a jump in f itself; the shell radius is not a "
        "regular matching surface.",
    )
    fprime_jump = sp.simplify(
        sp.limit(sp.diff(f_onset, r_probe), r_probe, R, "+")
        - sp.limit(sp.diff(f_cavity, r_probe), r_probe, R, "-")
    )
    sigma = sp.simplify(-(sp.sqrt(f_outer) - sp.sqrt(f_inner)) / (4 * sp.pi * R))
    surface_pressure = sp.simplify(fprime_jump / (16 * sp.pi * sp.sqrt(f_inner)))
    require(
        sigma == 0,
        f"Darmois surface density does not vanish at the sharp onset: {sigma}.",
    )
    require(
        sp.simplify(surface_pressure + R * rho_onset / 2) == 0,
        "The surface pressure is not P_s=-R rho_+/2.",
    )
    # Derived, not asserted: the intrinsic shell NEC/WEC combination sigma+P_s.
    sigma_plus_surface_pressure = sp.simplify(sigma + surface_pressure)
    onset_nec_violated = bool(
        sp.simplify(sigma_plus_surface_pressure + R * rho_onset / 2) == 0
        and sigma_plus_surface_pressure.is_negative is True
    )
    require(
        onset_nec_violated,
        f"sigma+P_s={sigma_plus_surface_pressure} is not negative for a positive "
        "onset density; the intrinsic shell NEC violation is not established.",
    )

    return {
        "profiles": endpoint_checks,
        "conditional_source_uniqueness": {
            "phi_prime_given_m_and_pr": str(phi_general),
            "phi_prime_pr_zero": str(phi_zero),
            "normalization_at_infinity_fixes_additive_constant": True,
            "general_branch_satisfies_radial_equation": bool(phi_general_solves),
            "pr_zero_branch_is_florides": bool(phi_zero_is_florides),
            "verified": conditional_source_verified,
        },
        "einstein_cluster": {
            "v_tan_squared": str(speed_squared),
            "massive_subluminal_compactness": "max(2m/r) < 2/3",
            "mu_domain": "0 < mu < 1/(3 q_(n,eta))",
        },
        "kappa_n4_eta3_exact": str(kappa_exact),
        "kappa_n4_eta3_decimal": float(sp.N(kappa_exact, 17)),
        "positive_family_junction": {
            "induced_metric_and_K_match": positive_junction_pass,
            "K_independent_of_mass_derivative": derivative_independence,
            "minimality": "m continuity plus bilateral lapse closure removes the layer; n>=1 additionally makes the classical stress continuous",
            "shell_side_from_one_sided_limits_of_F_n": junction_reports,
            "classical_density_continuous_all_orders": bool(classical_stress_continuous),
            "mass_slope_present_before_limit": bool(slopes_present),
            "mass_slope_absent_after_limit": bool(slopes_absent_after_limit),
        },
        "sharp_onset_junction": {
            "normal": "single normal directed from the cavity to increasing r",
            "sigma": str(sigma),
            "p_surface": str(surface_pressure),
            "rho_outer": str(rho_onset),
            "sigma_plus_p_surface_negative_for_positive_onset": onset_nec_violated,
            "sigma_plus_p_surface": str(sigma_plus_surface_pressure),
            "f_matches_across_onset": onset_f_matches,
        },
    }


def make_figure(grid: list[dict[str, float]]) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.8), constrained_layout=True)

    # Redundant gray, dash, and marker coding keeps the panels unambiguous
    # in monochrome print and without relying on color discrimination.
    family_styles = {
        2.0: dict(color="0.08", linestyle="-", marker="o"),
        3.0: dict(color="0.30", linestyle="--", marker="s"),
        4.0: dict(color="0.52", linestyle="-.", marker="^"),
        5.0: dict(color="0.70", linestyle=":", marker="D"),
    }
    for eta in (2.0, 3.0, 4.0, 5.0):
        # Sort defensively: line plots against n must not depend on the order in
        # which the grid rows happen to have been appended.
        rows = sorted((row for row in grid if row["eta"] == eta), key=lambda row: row["n"])
        ns = [row["n"] for row in rows]
        style = family_styles[eta]
        axes[0, 0].plot(ns, [row["q_max"] for row in rows], linewidth=1.6, markersize=4.2,
                        markerfacecolor="white", **style, label=fr"$\eta={eta:g}$")
        axes[0, 1].plot(ns, [row["mu_DEC"] for row in rows], linewidth=1.6, markersize=4.2,
                        markerfacecolor="white", **style, label=fr"$\eta={eta:g}$")
    axes[0, 0].set(xlabel=fr"profile order $n$", ylabel=fr"$q_{{n,\eta}}$")
    axes[0, 1].set(xlabel=fr"profile order $n$", ylabel=fr"$\mu_{{\rm DEC}}=0.4/q_{{n,\eta}}$")
    axes[0, 0].legend(ncol=2)

    for n, style in ((1, "--"), (4, "-"), (8, ":")):
        q, _ = q_max(n, 3.0)
        mu = np.linspace(1e-5, 0.98 * 0.4 / q, 160)
        depth = np.array([-core_log_lapse(value, n, 3.0) for value in mu])
        gray = {1: "0.10", 4: "0.38", 8: "0.64"}[n]
        marker = {1: "o", 4: "s", 8: "^"}[n]
        axes[1, 0].plot(mu, depth, style, color=gray, linewidth=1.7,
                        marker=marker, markevery=25, markersize=3.6,
                        markerfacecolor="white", label=fr"$n={n}$")
    kappa = kappa_numeric(4, 3.0)
    linear_mu = np.linspace(0.0, 0.35, 80)
    axes[1, 0].plot(linear_mu, kappa * linear_mu, color="0.78", linestyle="-.",
                    linewidth=1.1, label=fr"$\kappa_{{4,3}}\mu$")
    axes[1, 0].set(xlabel=fr"$\mu=M/L$", ylabel=fr"invariant core depth $-\Phi_c$")
    axes[1, 0].legend()

    p_values = np.linspace(0.0, 4.0, 240)
    axes[1, 1].plot(p_values, p_values, color="0.08", linestyle="-",
                    linewidth=1.7, label=r"energy: $p$")
    axes[1, 1].plot(
        p_values, p_values - 3.0, color="0.38", linestyle="--",
        linewidth=1.7, label=r"density: $p-3$"
    )
    axes[1, 1].plot(
        p_values,
        p_values - 1.0,
        color="0.65",
        linestyle="-.",
        linewidth=1.7,
        label=r"depth/phase: $p-1$",
    )
    axes[1, 1].axhline(0.0, color="0.55", linewidth=0.8)
    axes[1, 1].axvline(1.0, color="0.45", linestyle="--", linewidth=0.9)
    axes[1, 1].axvline(3.0, color="0.45", linestyle=":", linewidth=1.0)
    axes[1, 1].set(
        xlabel=r"mass exponent $p$ in $M\mapsto s^pM$",
        ylabel=r"scale exponent $a$ in $Q\sim s^a$",
        xlim=(0.0, 4.0),
        ylim=(-3.2, 4.2),
    )
    axes[1, 1].legend(loc="upper left")
    for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        ax.grid(alpha=0.2, linewidth=0.5)
        x_label = 0.58 if label == "(d)" else 0.02
        ax.text(
            x_label,
            0.95,
            label,
            transform=ax.transAxes,
            va="top",
            ha="center" if label == "(d)" else "left",
            fontweight="bold",
        )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "profile_family_benchmark.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "profile_family_benchmark.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    symbolic = symbolic_checks()
    homology = homology_scaling_certificate()

    grid: list[dict[str, float]] = []
    for eta in (2.0, 3.0, 4.0, 5.0):
        for n in range(1, 9):
            q, y_at_max = q_max(n, eta)
            require(
                1.0 < y_at_max < eta,
                f"n={n}, eta={eta}: q_max is attained at y={y_at_max}, on or "
                "outside the wall rather than strictly inside it.",
            )
            mu_dec = 0.4 / q
            mu_kinetic = 1.0 / (3.0 * q)
            # Physical bounds evaluated on the profile itself at the two
            # endpoints and at a probe just above the DEC endpoint.  The checks
            # below read these, never the predeclared targets, so no comparison
            # is a constant against itself.
            at_dec = profile_bounds(n, eta, mu_dec)
            at_kinetic = profile_bounds(n, eta, mu_kinetic)
            above_dec = profile_bounds(
                n, eta, (1.0 + DEC_ENDPOINT_PROBE_STEP) * mu_dec
            )
            grid.append(
                {
                    "n": n,
                    "eta": eta,
                    "q_max": q,
                    "y_at_max": y_at_max,
                    "mu_DEC": mu_dec,
                    "mu_kinetic": mu_kinetic,
                    # Analytic endpoint targets, predeclared as module constants;
                    # the *_computed entries are independent numerical
                    # maximizations of 2 m(r)/r over the actual profile and must
                    # reproduce them.
                    "compactness_at_DEC": DEC_COMPACTNESS,
                    "compactness_at_kinetic_limit": KINETIC_COMPACTNESS,
                    "compactness_at_DEC_computed": max_compactness(n, eta, mu_dec),
                    "compactness_at_kinetic_limit_computed": max_compactness(n, eta, mu_kinetic),
                    # Energy condition, causality and horizon function evaluated
                    # at the endpoints rather than inferred from them.
                    "dec_ratio_at_DEC": at_dec["max_tangential_pressure_over_density"],
                    "dec_ratio_above_DEC": above_dec[
                        "max_tangential_pressure_over_density"
                    ],
                    "horizon_function_min_at_DEC": at_dec["min_horizon_function"],
                    "speed_squared_at_DEC": at_dec["max_speed_squared"],
                    "speed_squared_at_kinetic_limit": at_kinetic["max_speed_squared"],
                    "horizon_function_min_at_kinetic_limit": at_kinetic[
                        "min_horizon_function"
                    ],
                    "kappa": kappa_numeric(n, eta),
                }
            )

    # Check one strict interior point for every profile/width pair.  At 80% of
    # the DEC endpoint the background is also inside the kinetic 2/3 domain.
    robustness = [
        radial_pressure_certificate(
            int(row["n"]), float(row["eta"]), 0.8 * float(row["mu_DEC"])
        )
        for row in grid
    ]

    weak_field: list[dict[str, float]] = []
    kappa_recovery: list[dict[str, float]] = []
    for n, eta in ((1, 2.0), (4, 3.0), (8, 5.0)):
        kappa = kappa_numeric(n, eta)
        ladder_depths: list[float] = []
        for mu in WEAK_FIELD_MU_LADDER:
            exact_depth = -core_log_lapse(mu, n, eta)
            ladder_depths.append(exact_depth)
            scaled_remainder = (exact_depth - kappa * mu) / mu**2
            # Finiteness alone is vacuous: a wrong kappa leaves a finite
            # delta/mu at every sampled mu.  Bound the ratio against the
            # predeclared budget instead, which delta/mu cannot respect.
            require(
                math.isfinite(scaled_remainder)
                and abs(scaled_remainder) <= WEAK_FIELD_REMAINDER_BOUND,
                f"n={n}, eta={eta}, mu={mu}: |remainder/mu^2|="
                f"{abs(scaled_remainder):.6g} exceeds the predeclared bound "
                f"{WEAK_FIELD_REMAINDER_BOUND}; the linear coefficient kappa "
                "does not reproduce the exact core depth to first order.",
            )
            weak_field.append(
                {
                    "n": n,
                    "eta": eta,
                    "mu": mu,
                    "exact_depth": exact_depth,
                    "linear_depth": kappa * mu,
                    "remainder_over_mu_squared": scaled_remainder,
                }
            )

        # Sharp weak-field gate.  The coarse |remainder/mu^2| <= 2 bound above
        # cannot resolve a relative error in kappa below ~2e-5.  Recover kappa
        # from the exact depths alone and compare: because the recovery is
        # independent of kappa_numeric, a relative error delta shows up as delta
        # itself, so the gate resolves down to its predeclared tolerance.
        recovered = extrapolated_kappa(
            WEAK_FIELD_MU_LADDER[:3], tuple(ladder_depths[:3])
        )
        recovery_error = abs(recovered - kappa) / abs(kappa)
        require(
            recovery_error <= WEAK_FIELD_KAPPA_RECOVERY_REL_TOL,
            f"n={n}, eta={eta}: kappa recovered from the exact core depths is "
            f"{recovered!r}, a relative {recovery_error:.6g} away from the "
            f"quadrature value {kappa!r}; the predeclared budget is "
            f"{WEAK_FIELD_KAPPA_RECOVERY_REL_TOL}.  The linear coefficient does "
            "not reproduce the weak-field core depth.",
        )
        kappa_recovery.append(
            {
                "n": n,
                "eta": eta,
                "kappa_quadrature": kappa,
                "kappa_recovered_from_depths": recovered,
                "relative_error": recovery_error,
                "extrapolation_mu": list(WEAK_FIELD_MU_LADDER[:3]),
            }
        )

    kappa = kappa_numeric(4, 3.0)
    lengths = {"compact_m": 5e-3, "expanded_m": 20e-3, "dwell_s": 0.5}
    phase_per_kg = ideal_cs_phase(1.0, lengths["compact_m"], lengths["expanded_m"], lengths["dwell_s"])
    phase_thresholds = {str(level): level / phase_per_kg for level in (1.0, 0.1, 0.01, 0.001)}
    inverse_length = 1.0 / lengths["compact_m"] - 1.0 / lengths["expanded_m"]
    clock_mass_thresholds = {
        "7.6e-21": 7.6e-21 * C_SI**2 / (kappa * G_SI * inverse_length),
        "3.2e-18": 3.2e-18 * C_SI**2 / (kappa * G_SI * inverse_length),
    }
    benchmark: dict[str, object] = {
        "observable": "leading fixed-ADM two-scale core lapse difference and ideal Cs-133 dwell phase",
        "compact_length_m": lengths["compact_m"],
        "expanded_length_m": lengths["expanded_m"],
        "dwell_time_s": lengths["dwell_s"],
        "phase_per_kg_rad": phase_per_kg,
        "phase_mass_thresholds_kg": phase_thresholds,
        "reference_mass_kg": 0.19,
        "reference_mass_phase_rad": ideal_cs_phase(0.19),
        "clock_fractional_shift_mass_thresholds_kg": clock_mass_thresholds,
        "interpretation": "ideal relational dwell term only; not a closed-fringe or apparatus claim",
    }

    checks = {
        "symbolic_profiles_pass": len(symbolic["profiles"]) == 8,
        "minimal_degree_uniqueness_pass": all(
            bool(row["minimal_degree_unique"]) for row in symbolic["profiles"].values()
        ),
        "conditional_pr_zero_uniqueness_pass": bool(
            symbolic["conditional_source_uniqueness"]["verified"]
        ),
        "kappa_exact_pass": abs(float(symbolic["kappa_n4_eta3_decimal"]) - kappa) < 2e-13,
        "all_q_maxima_in_wall": all(1.0 < row["y_at_max"] < row["eta"] for row in grid),
        # Certifies the two things the name claims, both from the profile.
        #   (i)  DEC is evaluated at mu_DEC and found exactly saturated,
        #        max p_t/rho = 1, and found violated a relative 1e-3 above it.
        #        That derives the endpoint 0.4/q from the energy condition
        #        instead of assuming it.
        #   (ii) the independently maximized max 2m(r)/r reproduces the analytic
        #        4/5, and the horizon function 1-2m/r is positive throughout.
        "dec_bound_horizon_free": all(
            abs(row["dec_ratio_at_DEC"] - 1.0) < SATURATION_TOLERANCE
            and row["dec_ratio_above_DEC"] > 1.0 + SATURATION_TOLERANCE
            and abs(row["compactness_at_DEC_computed"] - DEC_COMPACTNESS)
            < COMPACTNESS_TOLERANCE
            and row["horizon_function_min_at_DEC"] > 0.0
            and abs(
                row["horizon_function_min_at_DEC"] - (1.0 - DEC_COMPACTNESS)
            )
            < COMPACTNESS_TOLERANCE
            for row in grid
        ),
        # Likewise derived: at mu_kinetic the cluster particles are exactly
        # luminal (max v^2 = 1) and the computed compactness is the analytic
        # 2/3, while at the DEC endpoint the same particles are superluminal.
        # The subdomain is therefore strictly smaller for a physical reason, not
        # because 1/3 < 0.4 arithmetically.
        "kinetic_domain_stricter_than_dec": all(
            row["mu_kinetic"] < row["mu_DEC"]
            and abs(row["speed_squared_at_kinetic_limit"] - 1.0)
            < SATURATION_TOLERANCE
            and row["speed_squared_at_DEC"] > 1.0 + SATURATION_TOLERANCE
            and abs(
                row["compactness_at_kinetic_limit_computed"] - KINETIC_COMPACTNESS
            )
            < COMPACTNESS_TOLERANCE
            and row["compactness_at_kinetic_limit_computed"]
            < row["compactness_at_DEC_computed"]
            for row in grid
        ),
        "open_nonzero_radial_pressure_neighborhood": all(
            float(row["epsilon_witness"]) > 0.0 for row in robustness
        ),
        "weak_field_second_order_bounded": all(
            math.isfinite(row["remainder_over_mu_squared"])
            and abs(row["remainder_over_mu_squared"]) <= WEAK_FIELD_REMAINDER_BOUND
            for row in weak_field
        ),
        "weak_field_kappa_recovered_from_depths": all(
            row["relative_error"] <= WEAK_FIELD_KAPPA_RECOVERY_REL_TOL
            for row in kappa_recovery
        ),
        "homological_exponents_derived": all(homology["identities"].values()),
        "homological_no_free_lunch": not homology["joint_requirements_feasible"],
        "positive_family_junction_no_layer": symbolic["positive_family_junction"]["induced_metric_and_K_match"],
        "junction_independent_of_mass_derivative": symbolic["positive_family_junction"]["K_independent_of_mass_derivative"],
        "classical_stress_continuous_at_walls": symbolic["positive_family_junction"]["classical_density_continuous_all_orders"],
        "sharp_onset_f_continuous": symbolic["sharp_onset_junction"]["f_matches_across_onset"],
        "sharp_onset_nec_violation": symbolic["sharp_onset_junction"]["sigma_plus_p_surface_negative_for_positive_onset"],
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    payload = {
        "verdict": "PASS",
        "checks": checks,
        "regularity_note": "m and g_rr are C^n but not C^(n+1); Phi and g_tt are C^(n+1). Continuity of m alone already removes the Israel layer, even if m' jumps; n>=1 is the additional, stronger requirement that makes the classical stress continuous at the walls.",
        "energy_conditions": {
            "nontrivial_family": "0 < mu <= 0.4/q gives non-strict DEC; use < for a strict margin",
            "compactness": "C_star=2 mu q <= 0.8, hence this branch is horizon-free",
            "kinetic_subdomain": "C_star < 2/3, equivalently mu < 1/(3q), keeps Einstein-cluster particles massive and subluminal",
            "radial_pressure_robustness": "p_r=epsilon*rho*x*(1-x) preserves all EC for the certified open epsilon interval at strict DEC points",
            "logical_caution": "DEC alone is not asserted to exclude horizons outside the static horizon-free branch",
        },
        "symbolic": symbolic,
        "family_grid": grid,
        "radial_pressure_certificates": robustness,
        "weak_field_checks": weak_field,
        "weak_field_kappa_recovery": kappa_recovery,
        "homological_scale_descent": homology,
        "benchmark": benchmark,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "profile_family_junction_benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(grid)
    print(f"profiles checked: {len(symbolic['profiles'])}")
    print(f"family grid points: {len(grid)}")
    print(f"radial-pressure certificates: {len(robustness)}")
    print(
        "minimum epsilon witness: "
        f"{min(float(row['epsilon_witness']) for row in robustness):.12g}"
    )
    print(f"kappa_4_3: {kappa:.15f}")
    print(
        "worst kappa recovery relative error: "
        f"{max(row['relative_error'] for row in kappa_recovery):.3e} "
        f"(gate {WEAK_FIELD_KAPPA_RECOVERY_REL_TOL:g})"
    )
    print(f"phase per kg: {phase_per_kg:.12g} rad/kg")
    print(f"reference 0.19 kg phase: {benchmark['reference_mass_phase_rad']:.12g} rad")
    print("VERDICT: PASS")


if __name__ == "__main__":
    main()
