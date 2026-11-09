'''How fast the carpet forms at a fixed depth: the mean-square error of Theorem 1 against the bound (4.3).

Usage, from the root of the repository:
    python paper/numbers/formation_times.py [--procs P]
It runs in a few seconds and needs numpy and scipy.

We take the grating d/λ = 100 with l/λ = 5 (Figure 2; g_{d/λ} = 0, so no order is at cut-off) and with l/λ = 4.5
(g_{d/λ} =/= 0, so the order n = d/λ at cut-off is excited). At the depths z = z_T/10, z_T/4 and z_T/2 and the times
t = 2z, 5z, 7.5z, 10z, 10^2 z, 10^3 z and 10^4 z, we compute E_n(t,z) = c_n(t,z)/g_n - S_n(t,z) for every mode
|n| < 5 d/λ with snapshot_mode of transient_snapshots.py, as for Figure 2. By Parseval's identity (proof of Theorem 1),
the relative root-mean-square error of the field over one period is
    e(t,z) = ( sum_n |g_n|^2 E_n(t,z)^2 / sum_n |g_n|^2 )^{1/2},
which is the square root of the left-hand side of (4.3) divided by (1/d) int |g|^2. The bound (4.3) says
e <= (4 sqrt(31/10) C_J^2 omega z^2/R)^{1/2}, with R = sqrt(t^2 - z^2). The coefficients c_n/g_n do not depend on the slit
width, so we compute them once and weight them for each grating. At a fixed t, e(t,z) depends on the phase of the
carrier (the order at cut-off, for instance, oscillates at frequency omega), so we also give its average over one period,
    e_T = ( sum_n |g_n|^2 [E_n(t)^2 + E_n(t - T/4)^2]/2 / sum_n |g_n|^2 )^{1/2},
which is exact for errors that oscillate harmonically at frequency omega.

The remark after Theorem 1 says that omega z^2 = π z_T z^2/d^2, that no bound uniform in the grating can do better than
t >> omega z^2 because of an order at cut-off, and that otherwise the carpet forms much earlier. We print e and e_T
for both gratings and check that, without an order at cut-off, the error is far below the bound.
'''
import os, sys, time
import numpy as np
from multiprocessing import Pool

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, os.pardir))
sys.path.insert(0, here)
from common import g_n_two_sided, check
from cj_constant import c_j
from transient_snapshots import snapshot_mode
from pyTalbot.stationary_utils import stationary_coeff


##################################
# Configuration of the computation.
##################################

d_lambda = 100
omega = 2 * np.pi * d_lambda # Angular frequency of the source (d = c = 1)
z_T = 2.0 * d_lambda # Talbot distance = 2 d^2/λ
T = 1.0 / d_lambda # Period of the source
N_max = 5 * d_lambda # We keep the modes |n| < N_max
depths = (0.1 * z_T, 0.25 * z_T, 0.5 * z_T)
factors = (2.0, 5.0, 7.5, 10.0, 1e2, 1e3, 1e4) # The times t/z


def mode_errors(n):
    '''Computes E_n(t,z), E_n(t - T/4,z) and the stationary coefficients at every depth and time.

    Parameters
    ----------
    n : int
        The index of the mode, n >= 0.

    Returns
    -------
    n : int
        The index of the mode.
    errors : np.ndarray of floats
        The array of shape (len(depths), len(factors), 4) with E_n(t,z), E_n(t - T/4,z), S_n(t,z) and S_n(t - T/4,z).
    '''
    k_n = 2 * np.pi * n
    errors = np.empty((len(depths), len(factors), 4))
    for i, z in enumerate(depths):
        for j, factor in enumerate(factors):
            for s, t in enumerate((factor * z, factor * z - T / 4)):
                coeff = np.sin(omega * (t - z)) if n == 0 else snapshot_mode(k_n, t, np.array([z]), omega)[0]
                stationary = stationary_coeff(k_n, t, z, omega)
                errors[i, j, s] = coeff - stationary
                errors[i, j, 2 + s] = stationary
    return n, errors


if __name__ == "__main__":
    args = sys.argv[1:]
    procs = int(args[args.index("--procs") + 1]) if "--procs" in args else 4

    # We compute the errors of every mode, distributed among the processes
    start = time.time()
    errors = np.empty((N_max, len(depths), len(factors), 4))
    with Pool(procs) as pool:
        for n, mode_error in pool.imap_unordered(mode_errors, range(N_max), chunksize=4):
            errors[n] = mode_error
    print(f"modes 0 .. {N_max - 1} at {len(depths)} depths x {len(factors)} times: {time.time() - start:.0f} s")

    c_j_value = c_j()[0]
    constant = 4 * np.sqrt(31 / 10) * c_j_value ** 2
    n_values = np.arange(N_max)
    multiplicity = np.where(n_values == 0, 1.0, 2.0) # We count n and -n

    # We weight the errors of the modes for each grating
    results = {}
    for l_lambda in (5, 4.5):
        weights = multiplicity * g_n_two_sided(n_values, l_lambda / d_lambda) ** 2
        total = weights.sum()
        e = np.sqrt(np.einsum("n,nij->ij", weights, errors[..., 0] ** 2) / total)
        e_average = np.sqrt(np.einsum("n,nij->ij", weights, errors[..., 0] ** 2 + errors[..., 1] ** 2) / (2 * total))
        results[l_lambda] = (e, e_average)
        print(f"l/lambda = {l_lambda}: relative rms error e(t, z) [period average e_T]")
        for i, z in enumerate(depths):
            print(f"  z = {z / z_T:.2f} z_T: " + "  ".join(
                f"t = {factor:g}z: {100 * e[i, j]:.3g}% [{100 * e_average[i, j]:.3g}%]" for j, factor in enumerate(factors)))

    bound = np.array([[np.sqrt(constant * omega * z * z / np.sqrt((factor * z) ** 2 - z * z)) for factor in factors]
                      for z in depths])
    print("Square root of the right-hand side of (4.3), constant 4.79:")
    for i, z in enumerate(depths):
        print(f"  z = {z / z_T:.2f} z_T: " + "  ".join(f"t = {factor:g}z: {bound[i, j]:.0f}"
                                                       for j, factor in enumerate(factors[:4])))

    print("Remark after Theorem 1:")
    for z in depths:
        print(f"    w z^2 = {omega * z * z / z:.3g} z = pi z_T z^2/d^2 = {np.pi * z_T * z * z / z:.3g} z at z = {z / z_T:.2f} z_T")
    e_5 = results[5][0]
    for i, z in enumerate(depths):
        check(f"z = {z / z_T:.2f} z_T: error at t = 10 z far below the bound (ratio)", "<< 1",
              e_5[i, 3] / bound[i, 3], 0, 1e-3, "{:.1e}")
    for i, z in enumerate(depths):
        print(f"    l/lambda = 4.5, z = {z / z_T:.2f} z_T: e_T from t = 10 z to 10^4 z: "
              f"{100 * results[4.5][1][i, 3:].min():.2f}-{100 * results[4.5][1][i, 3:].max():.2f}% (t = 10^4 z is still "
              f"below w z^2 = {omega * z:.2g} z, so the order at cut-off has not settled)")
