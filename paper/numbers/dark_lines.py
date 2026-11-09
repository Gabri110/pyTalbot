'''Mean intensity on the dark lines of Sections 5.2 and 7.2, relative to the mean intensity of the carpet.

Usage, from the root of the repository:
    python paper/numbers/dark_lines.py
It runs in about a minute and only needs numpy.

The lines (7.3) are xi = 1/2 + kappa sigma, zeta = 2 sigma, with kappa odd, that is, from (x,z) = (d/2, 0) to
((kappa + 1/2) d, z_T). Along each line we evaluate the envelope of the stationary field built from the propagating
orders,
    e^{i omega z} U(xi,z) = sum_{|n| <= d/λ} g_n e^{2 π i n xi} e^{i omega z (1 - sqrt(1 - (n λ/d)^2))},
on 2^20 equally spaced values of sigma in (0, 1), and we divide the mean of |U|^2 by its mean over the carpet,
sum_{|n| <= d/λ} |g_n|^2 (Parseval). The same ratio with the paraxial phase π zeta n^2 isolates the effect of the slit
width alone, that is, of the paired weights g_n and g_{n+kappa}.

Section 7.2 says that the lines are only approximately dark because these weights differ and because of the
non-paraxial phases, and that "In Figures 3(e) and 3(f) the latter effect dominates". We check this statement.
'''
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import g_n_two_sided, check


def line_ratio(d_lambda, eta, kappa, paraxial=False, samples=2 ** 20, chunk=2 ** 14):
    '''Computes the mean of |U|^2 along the line kappa (0 < sigma < 1) over its mean on the carpet (d = 1).

    Parameters
    ----------
    d_lambda : float
        The ratio d/λ.
    eta : float
        The ratio l/d.
    kappa : int
        The slope of the line, odd.
    paraxial : bool, optional
        Whether we use the paraxial phase π zeta n^2 instead of the exact one.
    samples : int, optional
        The number of values of sigma.
    chunk : int, optional
        The number of values of sigma that we evaluate at once.

    Returns
    -------
    ratio : float
        The mean of |U|^2 on the line over its mean on the carpet.
    '''
    n_values = np.arange(-int(np.floor(d_lambda)), int(np.floor(d_lambda)) + 1)
    g_n_values = g_n_two_sided(n_values, eta)
    x = (n_values / d_lambda) ** 2

    # The phase of the envelope per unit of zeta: omega z (1 - sqrt(1 - x)) with omega z = 2 π zeta (d/λ)^2
    if paraxial:
        phase_per_zeta = np.pi * n_values.astype(float) ** 2
    else:
        phase_per_zeta = 2 * np.pi * d_lambda ** 2 * x / (1 + np.sqrt(1 - x))

    sigma = (np.arange(samples) + 0.5) / samples
    total = 0.0
    for j in range(0, samples, chunk):
        s = sigma[j:j + chunk, None]
        phase = 2 * np.pi * n_values * (0.5 + kappa * s) + phase_per_zeta * (2 * s)
        total += np.sum(np.abs(np.exp(1j * np.mod(phase, 2 * np.pi)) @ g_n_values) ** 2)
    return total / samples / np.sum(g_n_values ** 2)


if __name__ == "__main__":
    print("Gratings of Figures 3(e), 3(f), d/lambda = 100 (Section 7.2: 'the latter effect dominates'):")
    for l_lambda in (5, 2):
        for kappa in (1, -1, 3):
            ratio = line_ratio(100, l_lambda / 100, kappa)
            ratio_paraxial = line_ratio(100, l_lambda / 100, kappa, paraxial=True)
            print(f"    l/lambda = {l_lambda}, kappa = {kappa:+d}: exact {ratio:.3f}, paraxial phases only "
                  f"{ratio_paraxial:.4f}")
    for l_lambda in (5, 2):
        ratio = line_ratio(100, l_lambda / 100, 1)
        ratio_paraxial = line_ratio(100, l_lambda / 100, 1, paraxial=True)
        check(f"l/lambda = {l_lambda}, kappa = 1: non-paraxial part / slit-width part", "> 1 (dominates)",
              (ratio - ratio_paraxial) / ratio_paraxial, 1.0, np.inf, "{:.0f}")
