'''Difference between the transient carpets of Figure 2 and the stationary carpet, by range of z/t (Section 5.1).

Usage, from the root of the repository:
    python paper/numbers/front_bands.py [--stride S] [--procs P] [--coeffs COEFFS.npy]
e.g. python paper/numbers/front_bands.py --stride 4      (about five minutes on four cores)
     python paper/numbers/front_bands.py --coeffs results/coeffs_d100_Nz1440x16_t0.25_0.5_0.75.npy      (seconds)
It needs numpy and scipy.

The grating has d/λ = 100 and l/λ = 5, and the times are t = z_T/4, z_T/2 and 3z_T/4, as in Figure 2. Section 5.1 says
that the transient carpets "differ from the stationary one ... in pointwise relative root-mean-square of the squared
envelope by less than 0.5% for z < t/4, by about 3% for z < t/2, and by about 37% within 0.05 t of the front", and that
on each of these ranges "the difference depends on z/t alone, to within one percentage point". The remark after
Theorem 1 repeats the 3% for z < t/2.

We compute three relative root-mean-square differences, over the points (x,z) of each range, with x on the N_x = 405
cell centres of [0, d/2] and z on the sub-samples of Figures 2 and 3:
* intensity: sqrt( mean (env - |U|^2)^2 / mean |U|^4 ), with env = u(t)^2 + u(t - T/4)^2 the squared envelope that
  Figure 2 shows. This is the quantity of Section 5.1, which transient_snapshots.py prints as "pointwise";
* complex field: sqrt( mean |V - U e^{i omega t}|^2 / mean |U|^2 ), with V = -u(t - T/4) + i u(t), which equals
  U e^{i omega t} once u = Im[U e^{i omega t}];
* instantaneous field: sqrt( mean (u(t) - Im[U e^{i omega t}])^2 / mean Im[U e^{i omega t}]^2 ).
We check the numbers of the paper against the first one, and print the other two for comparison.

We compute the coefficients c_n(t,z) with snapshot_mode of transient_snapshots.py, as for Figure 2 (contour deformation
of Appendix C, and direct quadrature of (C.1) in the thin layer behind the front). The full grid of Figure 2 has
1440 x 16 depths over [0, z_T], which is the expensive computation of transient_snapshots.py. By default we take every
S-th sub-sample (S = 4, a spacing of 3.5 λ, incommensurate with λ/2 so that it samples the fringes evenly). S = 1 is
the grid of the paper. With --coeffs we instead read, without modifying it, the file of coefficients that
    python paper/transient_snapshots.py OUT_DIR 100 5 0.25,0.5,0.75
writes, OUT_DIR/coeffs_d100_Nz1440x16_t0.25_0.5_0.75.npy. The coarser grid of S = 4 is enough for the ranges well behind
the front. Within 0.05 t of the front, where the carpet changes fastest, it can move the result by about one percentage
point, so for the 37% we recommend the grid of the paper.
'''
import os, sys, time
import numpy as np
from multiprocessing import Pool

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, os.pardir))
sys.path.insert(0, here)
from common import check
from transient_snapshots import SnapshotConfig, z_subsamples, mode_worker, field_from_coeffs
from stationary_carpets import field


##################################
# Configuration of the comparison.
##################################

d_lambda, l_lambda = 100, 5
fractions = (0.25, 0.5, 0.75) # The times t/z_T of Figure 2
bands = ((0.0, 0.25), (0.25, 0.5), (0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)) # Ranges of z/t, as in transient_snapshots.py


def snapshot_coeffs(config, times, z_values, procs):
    '''Computes the coefficients $c_n(t,z)$ without the g_n factor for n = 0, ..., N_max-1 at each of the given times,
    on any depths. Unlike generate_snapshot_coeffs, it does not use all the sub-samples.

    Parameters
    ----------
    config : SnapshotConfig
        The class containing all the parameters of the snapshots.
    times : list of floats
        The times at which we compute the field.
    z_values : np.ndarray of floats
        The zs at which we compute the field.
    procs : int
        The number of processes.

    Returns
    -------
    coeffs : np.ndarray of floats
        The values of the coefficients. The array is of shape (len(times), N_max, len(z_values)).
    '''
    coeffs = np.zeros((len(times), config.N_max, z_values.size))

    # The mode n = 0 has no tail
    for i, t in enumerate(times):
        coeffs[i, 0] = np.where(z_values < t, np.sin(config.omega * (t - z_values)), 0.0)

    # We distribute the other modes among the processes
    jobs = [(n, config.omega, times, z_values) for n in range(1, config.N_max)]
    with Pool(procs) as pool:
        for n, mode_coeffs in pool.imap_unordered(mode_worker, jobs, chunksize=2):
            for i, c in enumerate(mode_coeffs):
                coeffs[i, n] = c
    return coeffs


def band_differences(coeffs, config, z_values, U):
    '''Computes the relative rms differences from the stationary carpet by range of z/t, for each time of Figure 2.

    Parameters
    ----------
    coeffs : np.ndarray of floats
        The coefficients at t and t - T/4 for each time, of shape (2 len(fractions), N_max, len(z_values)).
    config : SnapshotConfig
        The class containing all the parameters of the snapshots.
    z_values : np.ndarray of floats
        The zs of the coefficients.
    U : np.ndarray of complex
        The stationary field at the cell centres x and at z_values.

    Returns
    -------
    differences : dict
        For each fraction t/z_T, a list with one tuple per range of z/t: the differences of the intensity, of the
        complex field and of the instantaneous field, and the number of points.
    '''
    differences = {}
    for i, f in enumerate(fractions):
        t = f * config.z_T
        u, u_quarter = field_from_coeffs(coeffs[2 * i], config), field_from_coeffs(coeffs[2 * i + 1], config) # u(t), u(t - T/4)
        V = -u_quarter + 1j * u
        S = U * np.exp(1j * config.omega * t)
        result = []
        for lo, hi in bands:
            s = (z_values >= lo * t) & (z_values < hi * t)
            intensity = np.abs(U[:, s]) ** 2
            envelope = u[:, s] ** 2 + u_quarter[:, s] ** 2
            r_intensity = np.sqrt(np.mean((envelope - intensity) ** 2) / np.mean(intensity ** 2))
            r_field = np.sqrt(np.mean(np.abs(V[:, s] - S[:, s]) ** 2) / np.mean(intensity))
            r_instantaneous = np.sqrt(np.mean((u[:, s] - S[:, s].imag) ** 2) / np.mean(S[:, s].imag ** 2))
            result.append((r_intensity, r_field, r_instantaneous, int(s.sum())))
        differences[f] = result
    return differences


if __name__ == "__main__":
    args = sys.argv[1:]
    stride = int(args[args.index("--stride") + 1]) if "--stride" in args else 4
    procs = int(args[args.index("--procs") + 1]) if "--procs" in args else 4
    config = SnapshotConfig(d_lambda, l_lambda)

    # We compute the field at each time t and at t - T/4, for the squared envelope
    T = config._lambda / config.c
    times = []
    for f in fractions:
        times += [f * config.z_T, f * config.z_T - T / 4]

    z_all = z_subsamples(config)
    if "--coeffs" in args:
        coeffs_file = args[args.index("--coeffs") + 1]
        coeffs = np.load(coeffs_file, mmap_mode="r")
        z_values = z_all
        print(f"read {coeffs_file} (grid of the paper, {z_values.size} depths)")
    else:
        z_values = z_all[stride // 2::stride]
        start = time.time()
        coeffs = snapshot_coeffs(config, times, z_values, procs)
        print(f"d/lambda = {d_lambda}: {z_values.size} depths (every {stride}th sub-sample), {len(times)} times, "
              f"{time.time() - start:.0f} s on {procs} processes")

    # The stationary field, with the same truncation |n| < 5 d/λ
    x_values = (np.arange(config.N_x) + 0.5) * config.delta_x
    U = field(d_lambda, l_lambda, x_values, z_values)
    differences = band_differences(np.asarray(coeffs), config, z_values, U)

    print("relative rms difference from the stationary carpet: intensity / complex field / instantaneous field")
    for f in fractions:
        for (lo, hi), (r_i, r_f, r_s, points) in zip(bands, differences[f]):
            print(f"  t = {f:.2f} z_T, z/t in [{lo:.2f}, {hi:.2f}): {100 * r_i:6.2f}% / {100 * r_f:6.2f}% / "
                  f"{100 * r_s:6.2f}%   ({points} points)")

    # "about N%" accepts N +- 1 percentage point, a plain "less than N%" what is below N
    print("Section 5.1 (and the remark after Theorem 1), squared envelope:")
    claims = [(0, "less than 0.5%", 0.0, 0.5), (2, "about 3%", 2.0, 4.0), (5, "about 37%", 36.0, 38.0)]
    for b, paper, lo, hi in claims:
        values = [100 * differences[f][b][0] for f in fractions]
        check(f"z/t in [{bands[b][0]:.2f}, {bands[b][1]:.2f}), smallest of the 3 times", paper, min(values), lo, hi,
              "{:.2f}%")
        check(f"z/t in [{bands[b][0]:.2f}, {bands[b][1]:.2f}), largest of the 3 times", paper, max(values), lo, hi,
              "{:.2f}%")
        check(f"   spread over the 3 times ({min(values):.2f}-{max(values):.2f}%)", "<= 1 point",
              max(values) - min(values), 0, 1.0, "{:.2f} pt")
