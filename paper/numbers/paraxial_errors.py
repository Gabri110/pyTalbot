'''Relative L^2 error of the paraxial approximation for the binary grating (remark "Sharp-edged slits", Section 6.2).

Usage, from the root of the repository:
    python paper/numbers/paraxial_errors.py
It runs in a few seconds and only needs numpy.

We sum the series of Parseval's identity in the proof of Theorem 2 exactly, without truncating it at 5 d/λ:
    ||U_eps - U_Par||^2 / ||Psi||^2 = sum_n |g_n|^2 a_n(eps) / sum_n |g_n|^2,     sum_n |g_n|^2 = d/l,
    a_n = |e^{i phi_n} - 1|^2 = 4 sin^2(phi_n/2)                                       for |n| <= floor(1/eps) (propagating),
    a_n = |e^{-2 π zeta sqrt((n eps)^2 - 1)/eps^2} - e^{-i omega z + i π zeta n^2}|^2      otherwise (evanescent),
    phi_n = (2 π zeta/eps^2) (1 - sqrt(1 - x) - x/2) = (2 π zeta/eps^2) x^2 / (2 (1 + sqrt(1 - x))^2),  x = (n eps)^2,
with eps = λ/d, zeta = 2 z/z_T and omega z = 2 π zeta/eps^2. We use the second form of phi_n, which is free of
cancellation. Beyond |n| = 4/eps the evanescent factor is below e^{-10^3}, so a_n = 1 to machine precision and we add
that tail in closed form, d/l minus the partial sum of |g_n|^2.

We check the numbers of Section 6.2: the errors 28% (l/d = 0.2, zeta = 0.3, λ/d = 10^-2) and 16% (λ/d = 10^-3), the
fraction λ/(π^2 l) of sum |g_n|^2 carried by the modes with |n| > d/λ, and the floor (λ/l)^{1/2}/π = 14% for l/λ = 5.
'''
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import g_n_two_sided, check, check_rounded


def relative_error(eps, eta, zeta):
    '''Computes the exact relative error ||U_eps - U_Par|| / ||Psi|| for the binary grating.

    Parameters
    ----------
    eps : float
        The ratio λ/d.
    eta : float
        The ratio l/d.
    zeta : float
        The depth in units of z_T/2.

    Returns
    -------
    error : float
        The relative error.
    error_evanescent : float
        The part of the relative error that comes from the modes with |n| > d/λ.
    fraction_evanescent : float
        The fraction of sum_n |g_n|^2 carried by the modes with |n| > d/λ.
    '''
    n_c = int(np.floor(1 / eps + 1e-9))
    n_max = int(np.ceil(4 / eps))
    n_values = np.arange(1, n_max + 1) # n > 0; we double for -n, since everything is even in n
    g2_values = g_n_two_sided(n_values, eta) ** 2
    x = (n_values * eps) ** 2
    propagating = n_values <= n_c
    a_values = np.empty(n_values.size)
    phi = 2 * np.pi * zeta / eps ** 2 * x[propagating] ** 2 / (2 * (1 + np.sqrt(1 - x[propagating])) ** 2)
    a_values[propagating] = 4 * np.sin(phi / 2) ** 2
    evanescent = ~propagating
    decay = np.exp(-2 * np.pi * zeta * np.sqrt(x[evanescent] - 1) / eps ** 2)
    theta = -2 * np.pi * zeta / eps ** 2 + np.pi * zeta * n_values[evanescent].astype(float) ** 2
    theta = np.mod(theta, 2 * np.pi) # We reduce the large arguments before taking cos and sin
    a_values[evanescent] = np.abs(decay - np.exp(1j * theta)) ** 2
    total = 1 / eta # Sum over all n of |g_n|^2 = d/l
    tail = total - (1.0 + 2 * g2_values.sum()) # |n| > n_max, where a_n = 1
    error2 = 2 * np.sum(g2_values * a_values) + tail # a_0 = 0
    error2_evanescent = 2 * np.sum(g2_values[evanescent] * a_values[evanescent]) + tail
    fraction_evanescent = (2 * g2_values[evanescent].sum() + tail) / total
    return np.sqrt(error2 / total), np.sqrt(error2_evanescent / total), fraction_evanescent


if __name__ == "__main__":
    zeta = 0.3
    print("Exact relative L^2 error, l/d = 0.2, zeta = 0.3 (Section 6.2):")
    error_2, _, _ = relative_error(1e-2, 0.2, zeta)
    error_3, _, _ = relative_error(1e-3, 0.2, zeta)
    check_rounded("lambda/d = 1e-2 ('about 28%')", "28", error_2, 0, scale=100, fmt="{:.2f}%")
    check_rounded("lambda/d = 1e-3 ('16%')", "16", error_3, 0, scale=100, fmt="{:.2f}%")

    print("Modes beyond cut-off at fixed l/lambda (Section 6.2):")
    for l_lambda in (5, 2):
        for d_lambda in (100, 1000, 10000):
            eps = 1 / d_lambda
            _, error_evanescent, fraction = relative_error(eps, l_lambda * eps, zeta)
            print(f"    l/lambda = {l_lambda}, d/lambda = {d_lambda:5d}: fraction of sum|g_n|^2 at |n| > d/lambda = "
                  f"{fraction:.5f} (lambda/(pi^2 l) = {1 / (np.pi ** 2 * l_lambda):.5f}); evanescent part of the error "
                  f"{100 * error_evanescent:.2f}%")
    _, error_evanescent, fraction = relative_error(1e-4, 5e-4, zeta)
    check("fraction at |n| > d/lambda, l/lambda = 5, d/lambda = 1e4", "~0.0203", fraction,
          0.95 / (5 * np.pi ** 2), 1.05 / (5 * np.pi ** 2), "{:.5f}")
    check_rounded("floor (lambda/l)^{1/2}/pi, l/lambda = 5 ('14%')", "14", np.sqrt(1 / 5) / np.pi, 0, scale=100,
                  fmt="{:.2f}%")
    check("evanescent part of the exact error, l/lambda = 5, d/lambda = 1e4", ">= ~14%", 100 * error_evanescent, 13.5,
          100, "{:.2f}%")
