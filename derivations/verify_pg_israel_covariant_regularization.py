"""Verify the regularized PG--Israel sharp-onset correspondence.

The limiting Painleve--Gullstrand shift beta ~ sqrt(r-R) is only C^(0,1/2),
so this script does not assign a classical distributional Riemann tensor to the
raw limiting metric.  It verifies the narrower statement used in the paper:
smooth, horizon-free regularizations whose invariant mass variable

    w_epsilon = r beta_epsilon**2 = 2 m_MS,epsilon

converges to the same continuous piecewise-C2 function w with controlled first
derivative have an Einstein-tensor measure associated with the limit.  Its
singular part is independent of the tested mollifier and equals the tangential
Israel layer of the diagonal completion.

The calculation includes checks that reject the wrong sign in the PG
diagonalizing time transformation, a perturbed jump coefficient, and an
unnormalized mollifier.  Scientific checks use explicit exceptions so that
``python -O`` cannot disable them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "pg_israel_covariant_regularization.json"

# Fixed verifier data.  R is kept far enough from zero that every regularized
# collar lies inside the areal-radius chart.
R_VALUE = 2.0
JUMP_VALUE = 0.7
EPSILON_LADDER = (0.12, 0.06, 0.03, 0.015, 0.0075)

# Predeclared tolerances.  The weakest tested mollifier is asymmetric and has
# O(epsilon) convergence; the final collar is 1/16 of the first.
NORMALIZATION_TOL = 2.0e-11
SYMMETRY_TOL = 2.0e-11
FINAL_SURFACE_REL_TOL = 2.0e-3
CONVERGENCE_REDUCTION = 4.0
BULK_EXCESS_TOL = 2.0e-4
MUTATION_SIZE = 1.0e-2

# A separate one-sided flat-step construction regularizes beta directly while
# preserving the exact vacuum core.  Its epsilon ladder spans a factor 16, so
# W^(1,1) errors must contract by at least 10 and the C0 beta error by at least
# 3 (the analytic rates are epsilon and sqrt(epsilon), respectively).  The
# broad TV cap only detects a numerical blow-up; it is not fitted to the run.
DIRECT_BETA_EPSILON_LADDER = (0.06, 0.03, 0.015, 0.0075, 0.00375)
DIRECT_BETA_W11_REDUCTION = 10.0
DIRECT_BETA_C0_REDUCTION = 3.0
DIRECT_BETA_KERNEL_MASS_TOL = 5.0e-5
DIRECT_BETA_TV_CAP = 20.0
DIRECT_BETA_SURFACE_REL_TOL = 5.0e-3


def require(condition: object, message: str) -> None:
    """Fail-closed assertion that remains active under ``python -O``."""
    if not condition:
        raise AssertionError(message)


def canonical(expr: sp.Expr) -> sp.Expr:
    """Aggressively reduce a rational SymPy expression."""
    return sp.factor(sp.cancel(sp.together(sp.simplify(expr))))


def einstein_tensor(metric: sp.Matrix, coords: list[sp.Symbol]) -> sp.Matrix:
    """Compute the covariant Einstein tensor directly from a metric."""
    dim = len(coords)
    inverse = sp.simplify(metric.inv())
    gamma = [
        [
            [
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
                for c in range(dim)
            ]
            for b in range(dim)
        ]
        for a in range(dim)
    ]
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
    scalar = canonical(
        sum(
            inverse[a, b] * ricci[a, b]
            for a in range(dim)
            for b in range(dim)
        )
    )
    return sp.Matrix(
        dim,
        dim,
        lambda a, b: canonical(
            ricci[a, b] - sp.Rational(1, 2) * metric[a, b] * scalar
        ),
    )


@dataclass(frozen=True)
class Mollifier:
    name: str
    power: int
    tilt: float = 0.0

    def raw(self, u: float) -> float:
        if abs(u) >= 1.0:
            return 0.0
        one_minus_u2 = 1.0 - u * u
        return (1.0 + self.tilt * u) * math.exp(
            -1.0 / one_minus_u2**self.power
        )

    @cached_property
    def normalization(self) -> float:
        value, _ = quad(self.raw, -1.0, 1.0, epsabs=2.0e-14, epsrel=2.0e-14)
        return value

    def eta(self, u: float) -> float:
        return self.raw(u) / self.normalization

    def cdf(self, u: float) -> float:
        if u <= -1.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        value, _ = quad(self.eta, -1.0, u, epsabs=2.0e-13, epsrel=2.0e-13)
        return value

    def moment(self, order: int) -> float:
        value, _ = quad(
            lambda u: u**order * self.eta(u),
            -1.0,
            1.0,
            epsabs=2.0e-13,
            epsrel=2.0e-13,
        )
        return value


MOLLIFIERS = (
    Mollifier("standard_even_bump", power=1),
    Mollifier("sharper_even_bump", power=2),
    Mollifier("tilted_positive_bump", power=1, tilt=0.35),
)


def symbolic_checks() -> dict[str, object]:
    """Check the exact transformation, shell coefficient, and regularity."""
    beta = sp.symbols("beta", nonzero=True, real=True)
    f = 1 - beta**2
    g_pg = sp.Matrix([[-f, -beta], [-beta, 1]])

    # dT = dt + beta/f dr, hence dt = dT - beta/f dr.
    jacobian_correct = sp.Matrix([[1, -beta / f], [0, 1]])
    transformed = sp.simplify(jacobian_correct.T * g_pg * jacobian_correct)
    expected = sp.diag(-f, 1 / f)
    correct_diagonalization = all(
        sp.simplify(transformed[i, j] - expected[i, j]) == 0
        for i in range(2)
        for j in range(2)
    )

    # The opposite sign is a real negative control: it leaves a cross term.
    jacobian_wrong = sp.Matrix([[1, beta / f], [0, 1]])
    transformed_wrong = sp.simplify(jacobian_wrong.T * g_pg * jacobian_wrong)
    wrong_cross_term = sp.simplify(transformed_wrong[0, 1])
    wrong_sign_rejected = wrong_cross_term != 0

    # Derive the full four-dimensional mixed Einstein tensor directly in the
    # off-diagonal PG chart.  This closes the loophole left by an argument that
    # first diagonalizes and then tries to push a distribution through the
    # limiting C^(1,1/2) time map.
    t_pg, r_pg, theta, phi = sp.symbols(
        "t_pg r_pg theta phi", real=True, positive=True
    )
    beta_pg = sp.Function("beta_pg")(r_pg)
    coords_pg = [t_pg, r_pg, theta, phi]
    metric_pg = sp.Matrix(
        [
            [beta_pg**2 - 1, -beta_pg, 0, 0],
            [-beta_pg, 1, 0, 0],
            [0, 0, r_pg**2, 0],
            [0, 0, 0, r_pg**2 * sp.sin(theta) ** 2],
        ]
    )
    einstein_cov_pg = einstein_tensor(metric_pg, coords_pg)
    inverse_pg = sp.simplify(metric_pg.inv())
    einstein_mixed_pg = sp.Matrix(
        4,
        4,
        lambda a, b: canonical(
            sum(inverse_pg[a, c] * einstein_cov_pg[c, b] for c in range(4))
        ),
    )
    w_pg = r_pg * beta_pg**2
    expected_mixed_pg = sp.diag(
        -sp.diff(w_pg, r_pg) / r_pg**2,
        -sp.diff(w_pg, r_pg) / r_pg**2,
        -sp.diff(w_pg, r_pg, 2) / (2 * r_pg),
        -sp.diff(w_pg, r_pg, 2) / (2 * r_pg),
    )
    pg_tensor_exact = all(
        canonical(einstein_mixed_pg[i, j] - expected_mixed_pg[i, j]) == 0
        for i in range(4)
        for j in range(4)
    )
    pg_off_diagonal_zero = all(
        canonical(einstein_mixed_pg[i, j]) == 0
        for i in range(4)
        for j in range(4)
        if i != j
    )
    pg_volume_exact = canonical(
        -metric_pg.det() - r_pg**4 * sp.sin(theta) ** 2
    ) == 0

    r, radius, jump = sp.symbols("r R J", positive=True, finite=True)
    x = sp.symbols("x", positive=True)
    f_plus = 1 - jump * (r - radius) / r
    f_prime_at_shell = sp.simplify(sp.limit(sp.diff(f_plus, r), r, radius, dir="+"))
    rho_plus = jump / (8 * sp.pi * radius**2)
    israel_pressure = sp.simplify(f_prime_at_shell / (16 * sp.pi))
    reduced_pressure = sp.simplify(-jump / (16 * sp.pi * radius))
    coefficient_match = sp.simplify(
        israel_pressure + radius * rho_plus / 2
    ) == 0 and sp.simplify(israel_pressure - reduced_pressure) == 0

    beta_local = sp.sqrt(jump * x / (radius + x))
    beta_holder_coefficient = sp.sqrt(jump / radius)
    beta_prime_square_log_coefficient = sp.simplify(
        sp.simplify(x * sp.diff(beta_local, x) ** 2).subs(x, 0)
    )
    h_prime = sp.sqrt(jump * x * (radius + x)) / (radius + (1 - jump) * x)
    h_prime_holder_coefficient = sp.sqrt(jump / radius)
    # h'=sqrt(J/R)sqrt(x)+O(x^(3/2)), hence sqrt(x)h'' -> sqrt(J/R)/2.
    h_second_singularity = sp.sqrt(jump / radius) / 2

    regularity_match = (
        sp.simplify(beta_holder_coefficient - sp.sqrt(jump / radius)) == 0
        and sp.simplify(beta_prime_square_log_coefficient - jump / (4 * radius)) == 0
        and sp.simplify(h_prime_holder_coefficient - sp.sqrt(jump / radius)) == 0
        and sp.simplify(h_second_singularity - sp.sqrt(jump / radius) / 2) == 0
    )

    def coefficient_gate(candidate_jump: sp.Expr) -> bool:
        candidate = -candidate_jump / (16 * sp.pi * radius)
        return sp.simplify(candidate - israel_pressure) == 0

    jump_mutation_rejected = not coefficient_gate(jump * (1 + sp.Rational(1, 100)))
    zero_jump_removes_layer = sp.simplify(israel_pressure.subs(jump, 0)) == 0
    reversed_jump_reverses_pressure = sp.simplify(
        israel_pressure.subs(jump, -jump) + israel_pressure
    ) == 0

    checks = {
        "correct_pg_diagonalization_sign": bool(correct_diagonalization),
        "wrong_diagonalization_sign_rejected": bool(wrong_sign_rejected),
        "direct_pg_mixed_einstein_tensor_exact": bool(pg_tensor_exact),
        "direct_pg_off_diagonal_components_zero": bool(pg_off_diagonal_zero),
        "direct_pg_volume_density_exact": bool(pg_volume_exact),
        "israel_and_reduced_surface_coefficients_match": bool(coefficient_match),
        "square_root_regularity_and_gt_log_divergence": bool(regularity_match),
        "one_percent_jump_mutation_rejected": bool(jump_mutation_rejected),
        "zero_jump_removes_surface_layer": bool(zero_jump_removes_layer),
        "reversed_jump_reverses_surface_pressure": bool(reversed_jump_reverses_pressure),
    }
    for name, passed in checks.items():
        require(passed, f"Symbolic check failed: {name}")

    return {
        "checks": checks,
        "transformed_metric": str(transformed),
        "wrong_sign_cross_term": str(wrong_cross_term),
        "direct_pg_mixed_einstein_tensor": str(einstein_mixed_pg),
        "israel_pressure": str(israel_pressure),
        "rho_plus": str(rho_plus),
        "beta_holder_coefficient": str(beta_holder_coefficient),
        "beta_prime_square_log_coefficient": str(beta_prime_square_log_coefficient),
        "time_map_second_derivative_coefficient": str(h_second_singularity),
    }


def test_functions() -> dict[str, Callable[[float | np.ndarray], float | np.ndarray]]:
    return {
        "constant": lambda x: np.ones_like(x, dtype=float),
        "tilted_gaussian": lambda x: np.exp(-0.7 * x * x) * (1.0 + 0.2 * x),
        "oscillatory": lambda x: np.cos(0.6 * x) + 0.1 * x * x,
        "rational": lambda x: 1.0 / (1.0 + 0.3 * x * x),
    }


def surface_pairing(
    mollifier: Mollifier, epsilon: float, test: Callable[[float], float]
) -> float:
    """Pair 4 pi r^2 p_perp,epsilon with a radial scalar test.

    For w=J(r-R)_+, w_epsilon''=J eta_epsilon.  Changing variables
    r=R+epsilon*u keeps the calculation resolved at arbitrarily small epsilon.
    """
    value, _ = quad(
        lambda u: (R_VALUE + epsilon * u)
        * mollifier.eta(u)
        * test(epsilon * u),
        -1.0,
        1.0,
        epsabs=2.0e-13,
        epsrel=2.0e-13,
    )
    return -JUMP_VALUE * value / 4.0


def bulk_step_excess_pairing(
    mollifier: Mollifier, epsilon: float, test: Callable[[float], float]
) -> float:
    """Pair the regularized-minus-sharp densitized rho across the collar."""

    def integrand(u: float) -> float:
        heaviside = 1.0 if u > 0.0 else 0.0
        return (mollifier.cdf(u) - heaviside) * test(epsilon * u)

    value, _ = quad(
        integrand,
        -1.0,
        1.0,
        points=[0.0],
        epsabs=2.0e-12,
        epsrel=2.0e-12,
    )
    # 4 pi r^2 rho = w'/2, and dr=epsilon du.
    return JUMP_VALUE * epsilon * value / 2.0


def numerical_weak_limit_checks() -> dict[str, object]:
    tests = test_functions()
    expected_surface = -JUMP_VALUE * R_VALUE / 4.0
    rows: list[dict[str, object]] = []
    max_final_surface_rel_error = 0.0
    max_final_bulk_excess = 0.0
    worst_reduction = math.inf
    all_surface_errors_contract = True

    for mollifier in MOLLIFIERS:
        normalization = quad(
            mollifier.eta, -1.0, 1.0, epsabs=2.0e-13, epsrel=2.0e-13
        )[0]
        min_sample = min(mollifier.eta(float(u)) for u in np.linspace(-1.0, 1.0, 401))
        require(
            abs(normalization - 1.0) <= NORMALIZATION_TOL,
            f"Mollifier {mollifier.name} is not normalized.",
        )
        require(min_sample >= -NORMALIZATION_TOL, f"Mollifier {mollifier.name} is negative.")

        mollifier_row: dict[str, object] = {
            "mollifier": mollifier.name,
            "normalization": normalization,
            "first_moment": mollifier.moment(1),
            "second_moment": mollifier.moment(2),
            "tests": {},
        }

        # Resolve the compact C-infinity bump once.  A fixed collar coordinate
        # is preferable here to repeated nested adaptive quadratures: the weak
        # limits are then evaluated on the same 40001-node measure grid for all
        # epsilons and test functions.
        u_grid = np.linspace(-1.0, 1.0, 40_001)
        eta_grid = np.array([mollifier.eta(float(u)) for u in u_grid])
        cdf_grid = np.empty_like(eta_grid)
        cdf_grid[0] = 0.0
        increments = 0.5 * (eta_grid[1:] + eta_grid[:-1]) * np.diff(u_grid)
        cdf_grid[1:] = np.cumsum(increments)
        cdf_grid /= cdf_grid[-1]
        heaviside_grid = np.where(u_grid > 0.0, 1.0, 0.0)
        heaviside_grid[u_grid == 0.0] = 0.5

        for test_name, test in tests.items():
            surface_values = []
            bulk_excesses = []
            for epsilon in EPSILON_LADDER:
                test_values = np.asarray(test(epsilon * u_grid), dtype=float)
                surface_integrand = (
                    (R_VALUE + epsilon * u_grid) * eta_grid * test_values
                )
                surface_values.append(
                    -JUMP_VALUE * float(np.trapezoid(surface_integrand, u_grid)) / 4.0
                )
                bulk_integrand = (cdf_grid - heaviside_grid) * test_values
                bulk_excesses.append(
                    abs(
                        JUMP_VALUE
                        * epsilon
                        * float(np.trapezoid(bulk_integrand, u_grid))
                        / 2.0
                    )
                )
            surface_errors = [
                abs(value - expected_surface) / abs(expected_surface)
                for value in surface_values
            ]
            initial_error = max(surface_errors[0], 1.0e-16)
            final_error = surface_errors[-1]
            reduction = initial_error / max(final_error, 1.0e-16)

            require(
                final_error <= FINAL_SURFACE_REL_TOL,
                f"Surface weak limit failed for {mollifier.name}/{test_name}: {final_error}",
            )
            require(
                reduction >= CONVERGENCE_REDUCTION or final_error <= 5.0e-13,
                f"Surface convergence did not improve for {mollifier.name}/{test_name}.",
            )
            require(
                bulk_excesses[-1] <= BULK_EXCESS_TOL,
                f"A spurious density delta survived for {mollifier.name}/{test_name}.",
            )

            max_final_surface_rel_error = max(max_final_surface_rel_error, final_error)
            max_final_bulk_excess = max(max_final_bulk_excess, bulk_excesses[-1])
            if final_error > 5.0e-13:
                worst_reduction = min(worst_reduction, reduction)
            all_surface_errors_contract = all_surface_errors_contract and (
                reduction >= CONVERGENCE_REDUCTION or final_error <= 5.0e-13
            )
            mollifier_row["tests"][test_name] = {
                "surface_pairings": surface_values,
                "surface_relative_errors": surface_errors,
                "bulk_step_excesses": bulk_excesses,
                "error_reduction": reduction,
            }

        rows.append(mollifier_row)

    # An unnormalized kernel must change the delta coefficient by exactly its
    # normalization error.  This demonstrates that the normalization gate is
    # capable of rejecting a faulty regularization.
    mutated_normalization = 1.0 + MUTATION_SIZE
    mutated_surface = expected_surface * mutated_normalization
    unnormalized_mutation_rejected = (
        abs(mutated_surface - expected_surface) / abs(expected_surface)
        > FINAL_SURFACE_REL_TOL
    )
    require(unnormalized_mutation_rejected, "Unnormalized mollifier mutation was not rejected.")
    if math.isinf(worst_reduction):
        worst_reduction = math.inf

    checks = {
        "three_mollifiers_normalized_and_positive": len(rows) == 3,
        "surface_measure_converges_for_all_test_functions": (
            max_final_surface_rel_error <= FINAL_SURFACE_REL_TOL
        ),
        "rho_and_pr_have_no_delta_measure": max_final_bulk_excess <= BULK_EXCESS_TOL,
        "all_surface_errors_contract": all_surface_errors_contract,
        "unnormalized_mollifier_mutation_rejected": unnormalized_mutation_rejected,
    }
    for name, passed in checks.items():
        require(passed, f"Numerical check failed: {name}")

    return {
        "checks": checks,
        "R": R_VALUE,
        "jump": JUMP_VALUE,
        "rho_plus_at_R": JUMP_VALUE / (8.0 * math.pi * R_VALUE**2),
        "israel_surface_pressure": -JUMP_VALUE / (16.0 * math.pi * R_VALUE),
        "expected_densitized_surface_pairing": expected_surface,
        "epsilon_ladder": list(EPSILON_LADDER),
        "max_final_surface_relative_error": max_final_surface_rel_error,
        "max_final_bulk_step_excess": max_final_bulk_excess,
        "worst_error_reduction": worst_reduction,
        "mollifiers": rows,
    }


def direct_beta_regularization_checks() -> dict[str, object]:
    """Check the explicit flat-step beta regularization used in the proof.

    For x=r-R and target w=J*x_+, choose beta_epsilon=chi(x/epsilon)*beta,
    where chi is the standard C-infinity flat step.  On 0<u=x/epsilon<1,
    w_epsilon=J*epsilon*F(u), F(u)=u*chi(u)^2.  Hence
    w_epsilon'=J*F'(u) and w_epsilon''=(J/epsilon)*F''(u).
    """
    u = np.linspace(0.0, 1.0, 100_001)
    chi = np.zeros_like(u)
    chi_prime = np.zeros_like(u)
    interior = (u > 0.0) & (u < 1.0)
    ui = u[interior]
    z = 1.0 / ui - 1.0 / (1.0 - ui)
    chi_i = expit(-z)
    chi[interior] = chi_i
    chi[u >= 1.0] = 1.0
    chi_prime[interior] = chi_i * (1.0 - chi_i) * (
        1.0 / ui**2 + 1.0 / (1.0 - ui) ** 2
    )

    f_shape = u * chi**2
    f_prime = chi**2 + 2.0 * u * chi * chi_prime
    f_second = np.gradient(f_prime, u, edge_order=2)
    kernel_mass = float(np.trapezoid(f_second, u))
    kernel_tv = float(np.trapezoid(np.abs(f_second), u))
    require(
        abs(kernel_mass - 1.0) <= DIRECT_BETA_KERNEL_MASS_TOL,
        "The direct-beta flat-step kernel does not have unit jump mass.",
    )
    require(kernel_tv <= DIRECT_BETA_TV_CAP, "The direct-beta TV bound blew up.")
    require(
        abs(f_shape[0]) <= 1.0e-15
        and abs(f_prime[0]) <= 1.0e-12
        and abs(f_shape[-1] - 1.0) <= 1.0e-15
        and abs(f_prime[-1] - 1.0) <= 1.0e-10,
        "The flat-step endpoint data are inconsistent.",
    )

    w11_errors: list[float] = []
    beta_c0_errors: list[float] = []
    horizon_margins: list[float] = []
    surface_errors: dict[str, list[float]] = {
        name: [] for name in test_functions()
    }
    expected_surface = -JUMP_VALUE * R_VALUE / 4.0

    for epsilon in DIRECT_BETA_EPSILON_LADDER:
        w11_errors.append(
            JUMP_VALUE
            * epsilon
            * float(np.trapezoid(np.abs(f_prime - 1.0), u))
        )
        x = epsilon * u
        beta_target = np.sqrt(JUMP_VALUE * x / (R_VALUE + x))
        beta_regularized = chi * beta_target
        beta_c0_errors.append(float(np.max(np.abs(beta_regularized - beta_target))))
        w_regularized = JUMP_VALUE * epsilon * f_shape
        horizon_margins.append(
            float(np.min(1.0 - w_regularized / (R_VALUE + epsilon * u)))
        )

        for test_name, test in test_functions().items():
            test_values = np.asarray(test(epsilon * u), dtype=float)
            pairing = -JUMP_VALUE * float(
                np.trapezoid(
                    (R_VALUE + epsilon * u) * f_second * test_values,
                    u,
                )
            ) / 4.0
            surface_errors[test_name].append(
                abs(pairing - expected_surface) / abs(expected_surface)
            )

    w11_reduction = w11_errors[0] / w11_errors[-1]
    beta_c0_reduction = beta_c0_errors[0] / beta_c0_errors[-1]
    max_final_surface_error = max(errors[-1] for errors in surface_errors.values())
    require(
        w11_reduction >= DIRECT_BETA_W11_REDUCTION,
        "The direct-beta W^(1,1) error did not contract at the expected rate.",
    )
    require(
        beta_c0_reduction >= DIRECT_BETA_C0_REDUCTION,
        "The direct-beta C0 error did not contract at the expected rate.",
    )
    require(min(horizon_margins) > 0.9, "The explicit regularization lost its horizon margin.")
    require(
        max_final_surface_error <= DIRECT_BETA_SURFACE_REL_TOL,
        "The direct-beta surface pairing did not approach the Israel value.",
    )

    checks = {
        "direct_beta_flat_step_has_unit_jump_mass": (
            abs(kernel_mass - 1.0) <= DIRECT_BETA_KERNEL_MASS_TOL
        ),
        "direct_beta_flat_step_has_uniform_bv_bound": kernel_tv <= DIRECT_BETA_TV_CAP,
        "direct_beta_w11_convergence": w11_reduction >= DIRECT_BETA_W11_REDUCTION,
        "direct_beta_uniform_convergence": beta_c0_reduction >= DIRECT_BETA_C0_REDUCTION,
        "direct_beta_horizon_margin": min(horizon_margins) > 0.9,
        "direct_beta_surface_measure_converges": (
            max_final_surface_error <= DIRECT_BETA_SURFACE_REL_TOL
        ),
    }
    for name, passed in checks.items():
        require(passed, f"Direct-beta check failed: {name}")

    return {
        "checks": checks,
        "epsilon_ladder": list(DIRECT_BETA_EPSILON_LADDER),
        "kernel_mass": kernel_mass,
        "kernel_total_variation": kernel_tv,
        "w11_errors": w11_errors,
        "w11_error_reduction": w11_reduction,
        "beta_c0_errors": beta_c0_errors,
        "beta_c0_error_reduction": beta_c0_reduction,
        "minimum_horizon_margin": min(horizon_margins),
        "surface_relative_errors": surface_errors,
        "max_final_surface_relative_error": max_final_surface_error,
    }


def main() -> None:
    symbolic = symbolic_checks()
    numerical = numerical_weak_limit_checks()
    direct_beta = direct_beta_regularization_checks()
    checks = {
        **symbolic["checks"],
        **numerical["checks"],
        **direct_beta["checks"],
    }
    require(all(checks.values()), "At least one regularization check failed.")

    payload = {
        "status": "PASS",
        "claim": (
            "For the stated admissible smooth regularizations, the densitized "
            "Einstein tensor has a mollifier-independent associated measure; "
            "its singular part is the tangential Israel layer.  This is not a "
            "claim that the raw C^(0,1/2) PG metric has a classical "
            "distributional Riemann tensor."
        ),
        "assumptions": [
            "static spherical unit-lapse PG form",
            "continuous piecewise-C2 w=2 m_MS with w(R)=0 and finite jump [w']",
            "smooth horizon-free approximants with w_epsilon -> w in W^(1,1)_loc",
            "a fixed sign branch with beta_epsilon -> beta locally uniformly",
            "uniform local total variation of w_epsilon'",
            "unit-mass smooth mollifiers in the invariant areal-radius collar",
        ],
        "symbolic": symbolic,
        "weak_limit": numerical,
        "direct_beta_regularization": direct_beta,
        "checks": checks,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for name, passed in checks.items():
        print(f"CHECK {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Checks passed: {payload['checks_passed']}/{payload['checks_total']}")
    print(f"Max final surface relative error: {numerical['max_final_surface_relative_error']:.6e}")
    print(f"Max final bulk step excess: {numerical['max_final_bulk_step_excess']:.6e}")
    print(f"Worst error reduction: {numerical['worst_error_reduction']:.6e}")
    print(
        "Direct-beta max final surface relative error: "
        f"{direct_beta['max_final_surface_relative_error']:.6e}"
    )
    print(f"Wrote {OUTPUT}")
    print("VERDICT: PASS")


if __name__ == "__main__":
    main()
