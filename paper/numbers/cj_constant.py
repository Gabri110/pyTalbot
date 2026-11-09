'''The constant C_J of Theorem 1 and Appendix B, and the constants built from it (Section 4 and Appendix B).

Usage, from the root of the repository:
    python paper/numbers/cj_constant.py
It runs in about a second and only needs numpy and scipy.

We check, from the J_1 of scipy only:
* C_J := sup_{x>0} sqrt(x) |J_1(x)| ≈ 0.825 (Theorem 1 and Appendix B);
* kappa = sqrt(31/10) solves 2 kappa^{1/2} = (31/5) kappa^{-3/2} (proof of Theorem 1);
* the constant 4 sqrt(31/10) C_J^2 ≈ 4.79 of the mean-square bound (4.3);
* C(omega, d) ≈ 29 omega d/λ for d >> λ (remark after Theorem 1), from its definition in the proof of Theorem 1,
      C = C_J^2 [ 4 sum_{0<|k_n|<=omega} |k_n| + (31/5)^2 omega^4 sum_{|k_n|>omega} |k_n|^{-3} ],   k_n = 2 π n/d,  omega = 2 π/λ.
'''
import os, sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import j1, hankel1, zeta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check, check_rounded


def c_j():
    '''Computes C_J by maximising sqrt(x) J_1(x) on its first lobe (0, j_{1,1}), and checks that no later lobe is higher.

    Returns
    -------
    c_j : float
        The constant C_J.
    x_max : float
        The point where sqrt(x) J_1(x) attains C_J.
    envelope : float
        The bound sqrt(x) |H_1^{(1)}(x)| on the later lobes, at x = j_{1,1}.
    scan : float
        The maximum of sqrt(x) |J_1(x)| on a fine grid of (0, 10^4].
    '''
    result = minimize_scalar(lambda x: -np.sqrt(x) * j1(x), bounds=(0.5, 3.8), method="bounded",
                             options={"xatol": 1e-12})
    c_j, x_max = -result.fun, result.x

    # Beyond the first lobe sqrt(x)|J_1(x)| <= sqrt(x) M_1(x), with M_1 = |H_1^{(1)}|, and x M_1(x)^2 decreases for
    # nu = 1 > 1/2 (DLMF §10.18, from Nicholson's formula). So it suffices that sqrt(x) M_1(x) < C_J at x = j_{1,1}
    envelope = np.sqrt(3.8317) * abs(hankel1(1, 3.8317))

    # We also scan a fine grid up to x = 10^4, as a plain numerical cross-check
    x = np.linspace(1e-6, 1e4, 20_000_001)
    scan = np.max(np.sqrt(x) * np.abs(j1(x)))
    return c_j, x_max, envelope, scan


def c_omega_d(d_lambda, c_j):
    '''Computes C(omega, d)/(omega d/λ) from its definition in the proof of Theorem 1, with d = 1.

    Parameters
    ----------
    d_lambda : float
        The ratio d/λ.
    c_j : float
        The constant C_J.

    Returns
    -------
    ratio : float
        C(omega, d)/(omega d/λ), where omega d/λ = 2 π (d/λ)^2.
    '''
    omega = 2 * np.pi * d_lambda
    n_propagating = np.arange(1, int(np.floor(d_lambda)) + 1)
    sum_propagating = 2 * np.sum(2 * np.pi * n_propagating) # Sum over 0 < |k_n| <= omega, both signs
    n_0 = int(np.floor(d_lambda)) + 1
    sum_evanescent = 2 * (2 * np.pi) ** -3 * zeta(3, n_0) # Sum over |k_n| > omega of |k_n|^{-3}, exact tail
    C = c_j ** 2 * (4 * sum_propagating + (31 / 5) ** 2 * omega ** 4 * sum_evanescent)
    return C / (omega * d_lambda)


if __name__ == "__main__":
    c_j_value, x_max, envelope, scan = c_j()
    print("C_J = sup sqrt(x)|J_1(x)| (Theorem 1, Appendix B):")
    check_rounded("C_J ('approx 0.825')", "0.825", c_j_value, 3, fmt="{:.6f}")
    print(f"    attained at x = {x_max:.6f}; later lobes: sqrt(x) M_1(x) at j_11 = {envelope:.4f} < C_J; "
          f"scan of (0, 1e4]: max = {scan:.6f}")

    kappa = np.sqrt(31 / 10)
    print("Constants of the proof of Theorem 1:")
    print(f"    kappa = sqrt(31/10): 2 kappa^(1/2) = {2 * kappa ** 0.5:.6f}, (31/5) kappa^(-3/2) = {31 / 5 * kappa ** -1.5:.6f}")
    check_rounded("4 sqrt(31/10) C_J^2 (Theorem 1: 'approx 4.79')", "4.79", 4 * kappa * c_j_value ** 2, 2)

    print("C(omega, d)/(omega d/lambda) (remark after Theorem 1: 'approx 29 omega d/lambda for d >> lambda'):")
    for d_lambda in (100, 100.5, 1000, 1e5):
        check(f"d/lambda = {d_lambda:g}", "29", c_omega_d(d_lambda, c_j_value), 28.5, 29.5, "{:.3f}")
    print(f"    limit d/lambda -> oo: C_J^2 (4 + (31/5)^2) = {c_j_value ** 2 * (4 + (31 / 5) ** 2):.3f}")
