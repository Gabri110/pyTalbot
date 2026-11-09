'''Peak intensities on the subimage planes of the stationary carpets of Figure 3 (Section 5.2).

Usage, from the root of the repository:
    python paper/numbers/peak_values.py
It runs in about ten seconds and only needs numpy.

We evaluate the stationary field (4.1) of the binary grating on a plane z = zeta z_T/2 by one FFT in x,
    e^{i omega z} U(x,z) = sum_n g_n e^{2 π i n x/d} e^{i omega z (1 - sqrt(1 - (n λ/d)^2))},
truncated at |n| < N = 5 d/λ as in Figure 3. For d/λ = 10^5 we keep only the propagating orders, since the evanescent
ones are below e^{-10^4} on these planes. We compute the phase as omega z x/(1 + sqrt(1 - x)), x = (n λ/d)^2, which is
free of cancellation. The intensities are point values of |U|^2 in units of (A d/l)^2, on a grid of 2^16 to 2^22 points
per period, that is, at least 600 points per wavelength.

The "band-limited value" of the paper is the same sum with the paraxial phase π zeta n^2 and |n| <= d/λ only.

We check the numbers of Section 5.2:
* the Gibbs overshoot of the band-limited paraxial field, "to about 1.2 for q = 1, or up to 1.26 when l/λ = 2" (Figure 3);
* panels (g) and (h) (l/d = 0.2, d/λ = 100 and 400): the peaks of |U|^2 exceed 1.2/q by 9 to 35% on the planes
  z = z_T, z_T/2, z_T/4 and z_T/6, and by up to 73% on the other subimage planes (we take those with q <= 4);
* on z = z_T/6 they are 0.54 and 0.50, against about 0.40 (band-limited);
* for l/d = 0.2 and d/λ = 10^5 the peaks are still about 1.3/q.
'''
import os, sys
from math import gcd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import g_n_two_sided, check, check_rounded

#                d/λ, l/λ
panels = {"a": (5, 2),
          "b": (10, 5),
          "c": (20, 5),
          "d": (20, 2),
          "e": (100, 5),
          "f": (100, 2),
          "g": (100, 20),
          "h": (400, 80)}


def grid_size(d_lambda):
    '''Computes the size of the FFT: a power of two with at least 600 points per wavelength and room for |n| < 5 d/λ.'''
    return int(2 ** np.ceil(np.log2(max(600 * d_lambda, 4 * 5 * d_lambda, 2 ** 16))))


def plane_intensity(d_lambda, l_lambda, zeta, paraxial=False, n_cut=None):
    '''Computes |U|^2/(A d/l)^2 on the plane z = zeta z_T/2, on equally spaced points of one period (d = 1).

    Parameters
    ----------
    d_lambda : int
        The ratio d/λ.
    l_lambda : int
        The ratio l/λ.
    zeta : float
        The depth in units of z_T/2.
    paraxial : bool, optional
        Whether we use the band-limited paraxial field instead of U.
    n_cut : int, optional
        The largest |n| in the sum. By default 5 d/λ - 1 (d/λ only if d/λ > 1000 or paraxial is True).

    Returns
    -------
    intensity : np.ndarray of floats
        The values of |U|^2/(A d/l)^2 on grid_size(d_lambda) points.
    '''
    eta = l_lambda / d_lambda
    if n_cut is None:
        n_cut = d_lambda if paraxial else (5 * d_lambda - 1 if d_lambda <= 1000 else d_lambda)
    n_values = np.arange(-int(n_cut), int(n_cut) + 1)
    if paraxial:
        n_values = n_values[np.abs(n_values) <= d_lambda]
        phase = np.pi * zeta * n_values.astype(float) ** 2
        amplitude = np.ones(n_values.size)
    else:
        x = (n_values / d_lambda) ** 2
        omega_z = 2 * np.pi * zeta * d_lambda ** 2 # omega z with omega = 2 π d/λ and z = zeta d/λ (d = 1)
        propagating = x <= 1
        phase = np.where(propagating, omega_z * x / (1 + np.sqrt(np.where(propagating, 1 - x, 0.0))), omega_z)
        amplitude = np.where(propagating, 1.0, np.exp(-omega_z * np.sqrt(np.where(propagating, 0.0, x - 1))))
    M = grid_size(d_lambda)
    coeffs = np.zeros(M, dtype=complex)
    np.add.at(coeffs, n_values % M, g_n_two_sided(n_values, eta) * amplitude * np.exp(1j * np.mod(phase, 2 * np.pi)))
    U = np.fft.ifft(coeffs) * M
    return np.abs(U) ** 2 * eta ** 2


def peak(d_lambda, l_lambda, zeta, **kwargs):
    '''Computes the maximum over x of plane_intensity(d_lambda, l_lambda, zeta, **kwargs).'''
    return float(plane_intensity(d_lambda, l_lambda, zeta, **kwargs).max())


def subimage_planes(q_max):
    '''Computes the planes zeta = p/q in (0, 2] with gcd(p, q) = 1 and q <= q_max, as a list of pairs (p, q).'''
    return [(p, q) for q in range(1, q_max + 1) for p in range(1, 2 * q + 1) if gcd(p, q) == 1]


if __name__ == "__main__":
    print("Gibbs overshoot of the band-limited paraxial field on z = z_T (q = 1), Section 5.2:")
    narrow = []
    for key, (d_lambda, l_lambda) in panels.items():
        value = peak(d_lambda, l_lambda, 2.0, paraxial=True)
        if l_lambda == 2:
            narrow.append(value)
            check(f"panel ({key}), d/lambda = {d_lambda}, l/lambda = {l_lambda}", "up to 1.26", value, 1.15, 1.265,
                  "{:.3f}") # At most 1.26 once rounded
        else:
            check(f"panel ({key}), d/lambda = {d_lambda}, l/lambda = {l_lambda}", "about 1.2", value, 1.15, 1.25,
                  "{:.3f}") # Rounds to 1.2
    check_rounded("largest of the panels with l/lambda = 2", "1.26", max(narrow), 2, fmt="{:.3f}")

    # On the four planes that stationary_carpets.py --check uses, z = z_T, z_T/2 (q = 1), z_T/4 (q = 2) and z_T/6
    # (q = 3), and on every other subimage plane p/q with q <= 4 in 0 < z <= z_T (l/d = 0.2 < 1/q, so the replicas do
    # not overlap)
    four_planes = [(2, 1), (1, 1), (1, 2), (1, 3)]
    print("Panels (g), (h): excess of the peaks of |U|^2 over 1.2/q on the subimage planes:")
    four_all, other_all = [], []
    for key in ("g", "h"):
        d_lambda, l_lambda = panels[key]
        excess = {(p, q): peak(d_lambda, l_lambda, p / q) / (1.2 / q) - 1 for p, q in subimage_planes(4)}
        band_limited = {(p, q): peak(d_lambda, l_lambda, p / q, paraxial=True) for p, q in four_planes}
        four_all += [excess[pq] for pq in four_planes]
        other_all += [e for pq, e in excess.items() if pq not in four_planes]
        print(f"    panel ({key}), planes z_T, z_T/2, z_T/4, z_T/6: "
              + ", ".join(f"{100 * excess[pq]:+.1f}%" for pq in four_planes)
              + "  (band-limited peaks: " + ", ".join(f"{band_limited[pq]:.3f}" for pq in four_planes) + ")")
        print(f"    panel ({key}), the other planes with q <= 4: "
              + ", ".join(f"{p}/{q}: {100 * e:+.0f}%" for (p, q), e in excess.items() if (p, q) not in four_planes))
    check_rounded("four planes, both panels: smallest excess", "9", min(four_all), 0, scale=100, fmt="{:.1f}%")
    check_rounded("four planes, both panels: largest excess", "35", max(four_all), 0, scale=100, fmt="{:.1f}%")
    check_rounded("other planes with q <= 4, both panels: largest excess", "73", max(other_all), 0, scale=100,
                  fmt="{:.1f}%")

    print("Panels (g), (h) on z = z_T/6 (zeta = 1/3):")
    check_rounded("panel (g), peak of |U|^2", "0.54", peak(100, 20, 1 / 3), 2)
    check_rounded("panel (h), peak of |U|^2", "0.50", peak(400, 80, 1 / 3), 2)
    for key in ("g", "h"):
        d_lambda, l_lambda = panels[key]
        check(f"panel ({key}), band-limited paraxial peak", "about 0.40", peak(d_lambda, l_lambda, 1 / 3, paraxial=True),
              0.38, 0.42, "{:.3f}")

    print("l/d = 0.2, d/lambda = 10^5 ('the peaks are still about 1.3/q'): q x peak on the planes zeta = p/q:")
    values = []
    for p, q in subimage_planes(3):
        value = q * peak(100_000, 20_000, p / q)
        values.append(value)
        print(f"    zeta = {p}/{q}: q x peak = {value:.3f}")
    # We accept what rounds to 1.3 with one decimal
    check("smallest q x peak, d/lambda = 1e5", "about 1.3", min(values), 1.25, 1.35, "{:.3f}")
    check("largest q x peak, d/lambda = 1e5", "about 1.3", max(values), 1.25, 1.35, "{:.3f}")
