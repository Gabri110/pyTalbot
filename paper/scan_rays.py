'''Quadrature error of the rays (Gauss-Laguerre, N nodes) over every mode except the grazing one and every depth
0 < z <= z_T, at t = 2 z_T and t = 2 z_T - T/4, for d/λ = 5, 10, 20, 50 and 100. The paper quotes the worst cases in Appendix C.

Usage, from the root of the repository:
    python paper/scan_rays.py

We measure the error in c_n/g_n due to the rays alone, k_n z |Im[e^{i omega t} (dF_1 + dF_2)/2]|, where dF is the
difference between the rays with N and with N_ref nodes; the real segment, where there is one, is the same for both and
cancels. The Hankel functions are evaluated through the asymptotic series of pyTalbot/transient_fast.py.
We use units with d = c = 1, so that λ = 1/(d/λ).
'''
import os, sys
import numpy as np
from multiprocessing import Pool
from numpy.polynomial.laguerre import laggauss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from pyTalbot.transient_fast import hankel_integrand

node_numbers = (3, 4, 5, 6, 8, 12, 16, 24, 32, 48) # Numbers of nodes per ray that we test
N_ref, N_ref_2 = 150, 120 # Numbers of nodes of the reference, and of a check of it
laguerre = {N: laggauss(N) for N in node_numbers + (N_ref, N_ref_2)}


def ray_integral(r_0, k_n, z, omega, sign, direction, decay, N):
    '''Computes the integral along the ray r = r_0 + direction i s, s >= 0, by N-point Gauss-Laguerre quadrature.'''
    x, w = laguerre[N]
    s = x[:, None] / decay
    amplitude, exponent = hankel_integrand(r_0 + direction * 1j * s, k_n, z, omega, sign)
    return (direction * 1j / decay) * np.sum(w[:, None] * amplitude * np.exp(exponent + x[:, None]), axis=0)


def mode_errors(args):
    '''Computes the largest error over z of the mode n at time t for each number of nodes.

    Parameters
    ----------
    args : tuple
        (d/λ, n, t).

    Returns
    -------
    d_lambda, n, t : int, int, float
        The arguments.
    errors : dict
        For each number of nodes N, the largest error, the z where it occurs and the decay rate of the H_1^(1) ray there.
    '''
    d_lambda, n, t = args
    omega = 2 * np.pi * d_lambda
    k_n = 2 * np.pi * n
    z_T = 2.0 * d_lambda
    z = np.linspace(0, z_T, 2880)[1:]
    R = np.sqrt(t * t - z * z)

    # The H_1^(1) ray goes up for the evanescent modes, and down (after the real segment, if needed) for the propagating ones
    if k_n > omega:
        r_0 = R
        direction_1 = +1.0
        decay_1 = k_n - omega * R / t
    else:
        r_star = z * k_n / np.sqrt(omega * omega - k_n * k_n)
        r_0 = np.maximum(R, 1.25 * r_star)
        direction_1 = -1.0
        decay_1 = omega * r_0 / np.sqrt(r_0 * r_0 + z * z) - k_n
    decay_2 = k_n + omega * R / t

    F = {}
    for N in node_numbers + (N_ref, N_ref_2):
        F[N] = ray_integral(R, k_n, z, omega, -1, -1.0, decay_2, N) + ray_integral(r_0, k_n, z, omega, +1, direction_1, decay_1, N)
    errors = {}
    for N in node_numbers + (N_ref_2,):
        e = k_n * z * np.abs(np.imag(np.exp(1j * omega * t) * 0.5 * (F[N] - F[N_ref])))
        j = int(np.argmax(e))
        errors[N] = (float(e[j]), float(z[j]), float(decay_1[j]) if np.ndim(decay_1) else float(decay_1))
    return d_lambda, n, t, errors


if __name__ == "__main__":
    # Every mode except the grazing one, at t = 2 z_T and at t - T/4
    jobs = []
    for d_lambda in (5, 10, 20, 50, 100):
        z_T = 2.0 * d_lambda
        for t in (2 * z_T, 2 * z_T - 0.25 / d_lambda):
            for n in range(1, 5 * d_lambda):
                if n != d_lambda:
                    jobs.append((d_lambda, n, t))

    # We keep the worst case for each d/λ and number of nodes
    worst = {}
    with Pool(6) as pool:
        for d_lambda, n, t, errors in pool.imap_unordered(mode_errors, jobs, chunksize=4):
            for N, (e, z, decay_1) in errors.items():
                key = (d_lambda, N)
                if key not in worst or e > worst[key][0]:
                    worst[key] = (e, n, z, t, decay_1)
    for d_lambda in (5, 10, 20, 50, 100):
        for N in node_numbers + (N_ref_2,):
            e, n, z, t, decay_1 = worst[(d_lambda, N)]
            print(f"d/l={d_lambda:3d} N={N:3d}: max err {e:.2e}  at n={n} z={z:.3f} (z/zT={z/(2*d_lambda):.3f}) t={t} a1={decay_1:.3f}", flush=True)
