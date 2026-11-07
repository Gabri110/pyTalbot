'''Table 1 of the paper: direct quadrature on the real axis against contour deformation (Appendix C).

Usage, from the root of the repository:
    python paper/convergence.py [OUT.json]
It prints, for the four cases of Table 1, the number of integrand evaluations that each method needs to reach an error
below 1e-10, and the error against the number of evaluations. With OUT.json it also saves all these data.

We compare the two forms of the coefficients (R = sqrt(t^2 - z^2), rho = sqrt(r^2 + z^2)):
    c_n(t,z) = sin(omega (t-z)) - k_n z int_0^R J_1(k_n r) sin(omega (t - rho))/rho dr     (direct form, (C.1))
             = S_n(t,z) + k_n z Im[e^{i omega t} (F_1 + F_2)/2]                         (contour form, (C.3))
in units with d = c = 1 and d/λ = 100, at t = 2 z_T = 400. Unlike pyTalbot/transient_fast.py, this script evaluates the
Bessel and Hankel functions with scipy.
'''
import json, sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.laguerre import laggauss
from scipy.special import j1, hankel1e, hankel2e


##################################
# Configuration of the comparison.
##################################

d_lambda = 100
omega = 2 * np.pi * d_lambda # Angular frequency of the source
z_T = 2.0 * d_lambda # Talbot distance = 2 d^2/λ
t = 2 * z_T # Time at which we compare
LEGENDRE_X, LEGENDRE_W = leggauss(16)

# The four cases of Table 1: label, n and z
cases = [("(a) $n=50$, $z=z_T$: propagating", 50, z_T),
         ("(b) $n=95$, $z=z_T/2$: near grazing, arrived", 95, z_T / 2),
         ("(c) $n=95$, $z=z_T$: front not arrived", 95, z_T),
         ("(d) $n=101$, $z=z_T$: evanescent, near cut-off", 101, z_T)]


def stationary_mode(k_n, z):
    '''Computes the stationary coefficient S_n(t,z) without the g_n factor.'''
    return np.sin(omega * t - z * np.sqrt(omega * omega - k_n * k_n)) if k_n <= omega else np.sin(omega * t) * np.exp(-z * np.sqrt(k_n * k_n - omega * omega))


def direct_mode(k_n, z, panels):
    '''Computes the coefficient $c_n(t,z)$ without the g_n factor by composite 16-point Gauss-Legendre quadrature on [0, R].

    Parameters
    ----------
    k_n : float
        The wave number of the mode.
    z : float
        The z at which we compute the field.
    panels : int
        The number of panels.

    Returns
    -------
    coeff : float
        The value of the coefficient.
    evaluations : int
        The number of evaluations of the integrand.
    '''
    R = np.sqrt(t * t - z * z)
    total = 0.0
    edges = np.linspace(0.0, R, panels + 1)

    # We sum the panels in chunks to avoid memory problems
    chunk = 20000
    for a in range(0, panels, chunk):
        b = min(a + chunk, panels)
        e_0, e_1 = edges[a:b], edges[a + 1:b + 1]
        mid, half = 0.5 * (e_0 + e_1), 0.5 * (e_1 - e_0)
        r = (mid[:, None] + half[:, None] * LEGENDRE_X).ravel()
        weights = (half[:, None] * LEGENDRE_W).ravel()
        rho = np.sqrt(r * r + z * z)
        total += np.sum(weights * j1(k_n * r) * np.sin(omega * (t - rho)) / rho)
    return np.sin(omega * (t - z)) - k_n * z * total, 16 * panels


def hankel1_integrand(r, k_n, z):
    '''Computes f_1(r) = H_1^(1)(k_n r) e^{-i omega rho}/rho as amplitude * exp(exponent), using the scaled Hankel function.'''
    rho = np.sqrt(r * r + z * z)
    return hankel1e(1, k_n * r) / rho, 1j * k_n * r - 1j * omega * rho


def hankel2_integrand(r, k_n, z):
    '''Computes f_2(r) = H_1^(2)(k_n r) e^{-i omega rho}/rho as amplitude * exp(exponent), using the scaled Hankel function.'''
    rho = np.sqrt(r * r + z * z)
    return hankel2e(1, k_n * r) / rho, -1j * k_n * r - 1j * omega * rho


def ray_integral(f, r_0, direction, decay, k_n, z, N):
    '''Computes the integral of f along the ray r = r_0 + direction i s, s >= 0, by N-point Gauss-Laguerre quadrature.
    Returns the integral and the number of evaluations of the integrand.'''
    x, w_l = laggauss(N)
    amplitude, exponent = f(r_0 + direction * 1j * x / decay, k_n, z)
    return (direction * 1j / decay) * np.sum(w_l * amplitude * np.exp(exponent + x)), N


def real_segment_integral(f, r_a, r_b, k_n, z, panels):
    '''Computes the integral of f on the real segment [r_a, r_b] with 16-point Gauss-Legendre panels uniform in v = 1/r.
    Returns the integral and the number of evaluations of the integrand.'''
    v_a, v_b = 1.0 / r_b, 1.0 / r_a
    edges = np.linspace(v_a, v_b, panels + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    v = (mid[:, None] + half[:, None] * LEGENDRE_X).ravel()
    weights = (half[:, None] * LEGENDRE_W).ravel()
    amplitude, exponent = f(1.0 / v, k_n, z)
    return np.sum(weights * amplitude * np.exp(exponent) / (v * v)), 16 * panels


def contour_mode(k_n, z, N, segment_panels=None):
    '''Computes the coefficient $c_n(t,z)$ without the g_n factor by contour deformation.

    Parameters
    ----------
    k_n : float
        The wave number of the mode.
    z : float
        The z at which we compute the field.
    N : int
        The number of Gauss-Laguerre nodes per ray.
    segment_panels : int, optional
        The number of panels on the real segment, for the modes that need it.

    Returns
    -------
    coeff : float
        The value of the coefficient.
    evaluations : int
        The number of evaluations of the integrand.
    segment_evaluations : int
        The number of those on the real segment.
    '''
    R = np.sqrt(t * t - z * z)
    rho = lambda r: np.sqrt(r * r + z * z)
    F_2, n_2 = ray_integral(hankel2_integrand, R, -1.0, k_n + omega * R / t, k_n, z, N)
    n_segment = 0
    if k_n > omega:
        F_1, n_1 = ray_integral(hankel1_integrand, R, +1.0, k_n - omega * R / t, k_n, z, N)
    else:
        r_star = z * k_n / np.sqrt(omega * omega - k_n * k_n)
        if r_star < 0.8 * R:
            F_1, n_1 = ray_integral(hankel1_integrand, R, -1.0, omega * R / t - k_n, k_n, z, N)
        else:
            r_0 = 1.25 * r_star
            S, n_segment = real_segment_integral(hankel1_integrand, R, r_0, k_n, z, segment_panels)
            F_1, n_1 = ray_integral(hankel1_integrand, r_0, -1.0, omega * r_0 / rho(r_0) - k_n, k_n, z, N)
            F_1 = F_1 + S
    E = k_n * z * np.imag(np.exp(1j * omega * t) * 0.5 * (F_1 + F_2))
    return stationary_mode(k_n, z) + E, n_1 + n_2 + n_segment, n_segment


def segment_panels_needed(k_n, z):
    '''Computes the number of panels of the real segment from the variation of the phase (about one panel per half
    oscillation), and that variation.'''
    R = np.sqrt(t * t - z * z)
    r_star = z * k_n / np.sqrt(omega * omega - k_n * k_n)
    r_values = np.linspace(R, 1.25 * r_star, 2001)
    phase = k_n * r_values - omega * np.sqrt(r_values * r_values + z * z)
    return int(np.ceil((phase.max() - phase.min()) / np.pi)) + 8, (phase.max() - phase.min())


if __name__ == "__main__":
    results = {}
    for label, n, z in cases:
        k_n = 2 * np.pi * n
        R = np.sqrt(t * t - z * z)
        oscillations = (k_n * R + omega * (t - z)) / (2 * np.pi) # The phase k_n r + omega rho grows by k_n R + omega (t - z) on [0, R]
        segment_panels = None
        phase_variation = 0.0
        if k_n < omega and z * k_n / np.sqrt(omega * omega - k_n * k_n) >= 0.8 * R:
            segment_panels, phase_variation = segment_panels_needed(k_n, z)

        # Reference values, computed with the contour method at a much higher resolution, and a check of them
        reference, _, _ = contour_mode(k_n, z, 150, None if segment_panels is None else 4 * segment_panels)
        reference_2, _, _ = contour_mode(k_n, z, 120, None if segment_panels is None else 3 * segment_panels)
        E_reference = reference - stationary_mode(k_n, z)

        # Error of the contour method against the number of evaluations
        contour_errors = []
        if segment_panels is None:
            for N in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100):
                c, evaluations, n_segment = contour_mode(k_n, z, N)
                contour_errors.append((N, evaluations, abs(c - reference)))
        else: # The rays have converged with N = 32, so we vary the resolution of the real segment
            for fraction in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0):
                c, evaluations, n_segment = contour_mode(k_n, z, 32, max(1, int(fraction * segment_panels)))
                contour_errors.append((fraction, evaluations, abs(c - reference)))

        # Error of the direct method against the number of evaluations
        direct_errors = []
        for P in [2 ** j for j in range(4, 20)]:
            c, evaluations = direct_mode(k_n, z, P)
            direct_errors.append((P, evaluations, abs(c - reference)))

        # Evaluations needed for an error below 1e-10 (the error stays below it at all the finer resolutions tested)
        Ps = np.unique(np.round(np.geomspace(oscillations / 32, oscillations / 2, 200)).astype(int))
        errors = np.array([abs(direct_mode(k_n, z, int(P))[0] - reference) for P in Ps])
        threshold_direct = int(16 * Ps[np.where(errors >= 1e-10)[0].max() + 1])
        threshold_contour = None
        if segment_panels is None:
            threshold_contour = min(evaluations for N, evaluations, e in contour_errors if e < 1e-10)
        else:
            Qs = np.unique(np.round(np.geomspace(0.05 * segment_panels, 0.5 * segment_panels, 200)).astype(int))
            errors = np.array([abs(contour_mode(k_n, z, 32, int(Q))[0] - reference) for Q in Qs])
            Q = int(Qs[np.where(errors >= 1e-10)[0].max() + 1])
            threshold_contour = contour_mode(k_n, z, 32, Q)[1]

        r_star = z * k_n / np.sqrt(omega * omega - k_n * k_n) if k_n < omega else None
        decay_F_1 = (k_n - omega * R / t) if k_n > omega else (omega * R / t - k_n if segment_panels is None else None)
        results[label] = dict(n=n, z=z, R=R, k=k_n, oscillations=oscillations, seg_panels=segment_panels,
                              seg_phase=phase_variation, rstar=r_star, rate_F1=decay_F_1, ref=reference,
                              E_ref=E_reference, ref_check=abs(reference - reference_2), contour=contour_errors,
                              direct=direct_errors, threshold_direct=threshold_direct, threshold_contour=threshold_contour)
        print(f"{label}: c_n={reference:.10f} E_n={E_reference:.3e} refcheck={abs(reference-reference_2):.1e} osc={oscillations:,.0f} r*={r_star} R={R:.1f} segp={segment_panels} seg_osc={phase_variation/(2*np.pi):.0f} rateF1={decay_F_1}", flush=True)
        print(f"   thresholds for 1e-10: direct {threshold_direct} ({threshold_direct/oscillations:.2f} per oscillation), contour {threshold_contour}", flush=True)
        for a, evaluations, e in contour_errors: print(f"   contour {a}: evals={evaluations:7d} err={e:.2e}")
        for P, evaluations, e in direct_errors: print(f"   direct  P={P:6d}: evals={evaluations:8d} err={e:.2e}")

    if len(sys.argv) > 1:
        json.dump(results, open(sys.argv[1], "w"), indent=1, default=float)
