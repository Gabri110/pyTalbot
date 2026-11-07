'''Snapshots of the transient Talbot carpet while it forms, t < z_T (the data of Figure 2 of the paper).

Usage, from the root of the repository:
    python paper/transient_snapshots.py OUT_DIR D_LAMBDA L_LAMBDA T1/z_T,T2/z_T,...
e.g. python paper/transient_snapshots.py results 100 5 0.25,0.5,0.75     (Figure 2; about eight minutes on six cores)
It writes OUT_DIR/snapshots_d<D_LAMBDA>_l<L_LAMBDA>.npz, which plot_snapshots.py plots, and prints the root-mean-square
differences from the stationary carpet quoted in Section 5.1. The coefficients c_n(t,z) do not depend on the slit width,
so they are saved in OUT_DIR and reused by later runs with other values of L_LAMBDA.

For t < z_T the field vanishes beyond the front z = t (c = 1). Behind it, we compute the coefficients c_n(t,z) with
pyTalbot/transient_fast.py (contour deformation, Appendix C), except in a thin layer behind the front where
k_n sqrt(t^2 - z^2) < 100 is too small for the asymptotic series of the Hankel functions used there. In that layer we
integrate equation (C.1) of the paper directly,
    c_n(t,z) = sin(omega (t-z)) - k_n z int_0^R J_1(k_n r) sin(omega (t - rho))/rho dr,   R = sqrt(t^2 - z^2), rho = sqrt(r^2 + z^2),
by composite Gauss-Legendre quadrature, which is cheap there because t - z is small.

We plot the squared envelope u(t)^2 + u(t - T/4)^2 in units of (A d/l)^2. The grid is the one of stationary_carpets.py
(Figure 3): N_x = 405 cell centres per half period and N_z = 1440 pixels over 0 < z < z_T. For d/λ = 100 a pixel is
about 14 wavelengths wide in z, so point samples would alias the structure on the scale of the wavelength. Therefore
every pixel holds the AVERAGE of the envelope over its width in z (S = 16 equally spaced sub-samples), as in Figure 3.
'''
import os, sys, time
import numpy as np
from multiprocessing import Pool
from numpy.polynomial.legendre import leggauss
from scipy.special import j1

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from pyTalbot.talbotconfig import TalbotConfig
from pyTalbot.stationary_utils import g_n_rect_delta
from pyTalbot.transient_fast import transient_mode

kR_min = 100.0 # Below this value of k_n sqrt(t^2 - z^2) we integrate (C.1) directly
LEGENDRE_X, LEGENDRE_W = leggauss(16)


class SnapshotConfig(TalbotConfig):
    def __init__(self, d_lambda, l_lambda, N_x=405, N_z=1440, S_sub=16):
        '''The parameters of the snapshots. They are those of TalbotConfig, with the following changes.

        z_T : Talbot distance = 2 d^2/λ, as in the paper.
        N_x : Number of cell centres in 0 < x < d/2.
        N_z : Number of pixels in 0 < z < z_T.
        S_sub : Number of sub-samples in z per pixel.
        N_max : Value of the series' cutoff, N = 5 d/λ as in the paper.
        '''
        super().__init__()

        # Parameters of the problem
        self._lambda = self.d / d_lambda # Wavelength
        self.l = l_lambda * self._lambda # Width of the gratings

        # Related relevant magnitudes
        self.omega = 2 * np.pi * self.c / self._lambda # Angular frequency of the source
        self.z_T = 2 * self.d ** 2 / self._lambda # Talbot distance = 2 d^2/λ

        # Grid parameters
        self.N_x, self.N_z, self.S_sub = N_x, N_z, S_sub

        # Simulation parameters
        self.N_max = int(round(5 * d_lambda)) # Number of terms in the series

        # Other relevant magnitudes
        self.delta_t = self.z_T/self.c/(self.N_t-1) * (self.final_t_zT - self.initial_t_zT) # Time between photos
        self.delta_x = self.d / 2 / self.N_x # X-Distance between points
        self.delta_z = self.z_T / self.N_z # Width of a pixel in z


def z_subsamples(config):
    '''Computes the depths of the S_sub equally spaced sub-samples inside each of the N_z pixels of 0 < z < z_T.

    Parameters
    ----------
    config : SnapshotConfig
        The class containing all the parameters of the snapshots.

    Returns
    -------
    z_values : np.ndarray of floats
        The depths, pixel by pixel. The array is of length N_z * S_sub.
    '''
    j = np.arange(config.N_z)[:, None] + (np.arange(config.S_sub)[None, :] + 0.5) / config.S_sub
    return (j * config.delta_z).ravel()


def pixel_average(values, config):
    '''Averages values at the sub-samples over each pixel.

    Parameters
    ----------
    values : np.ndarray of floats
        The values at the sub-samples, with shape (..., N_z * S_sub).
    config : SnapshotConfig
        The class containing all the parameters of the snapshots.

    Returns
    -------
    averages : np.ndarray of floats
        The averages over each pixel, with shape (..., N_z).
    '''
    return values.reshape(values.shape[:-1] + (config.N_z, config.S_sub)).mean(axis=-1)


def direct_mode(k_n, t, z, omega):
    '''Computes the coefficient $c_n(t,z)$ without the g_n factor by composite 16-point Gauss-Legendre quadrature of (C.1).

    Parameters
    ----------
    k_n : float
        The wave number of the mode.
    t : float
        The time at which we compute the field.
    z : float
        The z at which we compute the field. Must satisfy 0 < z < t.
    omega : float
        The angular frequency of the source.

    Returns
    -------
    coeff : float
        The value of the coefficient.
    '''
    R = np.sqrt(t * t - z * z)
    panels = int(np.ceil((omega * (t - z) + k_n * R) / np.pi)) + 4
    edges = np.linspace(0.0, R, panels + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    r = (mid[:, None] + half[:, None] * LEGENDRE_X[None, :]).ravel()
    weights = (half[:, None] * LEGENDRE_W[None, :]).ravel()
    rho = np.sqrt(r * r + z * z)
    integral = np.sum(weights * j1(k_n * r) * np.sin(omega * (t - rho)) / rho)
    return np.sin(omega * (t - z)) - k_n * z * integral


def snapshot_mode(k_n, t, z, omega):
    '''Computes the coefficients $c_n(t,z)$ without the g_n factor for any z >= 0: zero beyond the front, (C.1) in the
    thin layer behind it and transient_mode everywhere else.

    Parameters
    ----------
    k_n : float
        The wave number of the mode.
    t : float
        The time at which we compute the field.
    z : np.ndarray of floats
        The zs at which we compute the field.
    omega : float
        The angular frequency of the source.

    Returns
    -------
    coeff : np.ndarray of floats
        The values of the coefficients at each z.
    '''
    coeff = np.zeros_like(z)
    behind = z < t
    near = behind & (k_n * np.sqrt(np.maximum(t * t - z * z, 0.0)) < kR_min)
    far = behind & ~near
    if np.any(far):
        coeff[far] = transient_mode(k_n, t, z[far], omega)
    for j in np.where(near)[0]:
        coeff[j] = direct_mode(k_n, t, z[j], omega) if z[j] > 0 else np.sin(omega * t)
    return coeff


def mode_worker(args):
    '''Computes the coefficients of the mode n at each of the given times. Used by generate_snapshot_coeffs.'''
    n, omega, times, z = args
    k_n = 2 * np.pi * n
    return n, [snapshot_mode(k_n, t, z, omega) for t in times]


def generate_snapshot_coeffs(config, times, procs=os.cpu_count()):
    '''Computes the coefficients $c_n(t,z)$ without the g_n factor for n = 0, ..., N_max-1 at each of the given times.

    Parameters
    ----------
    config : SnapshotConfig
        The class containing all the parameters of the snapshots.
    times : list of floats
        The times at which we compute the field.
    procs : int, optional
        The number of processes.

    Returns
    -------
    coeffs : np.ndarray of floats
        The values of the coefficients. The array is of shape (len(times), N_max, N_z * S_sub).
    '''
    z = z_subsamples(config)
    coeffs = np.zeros((len(times), config.N_max, z.size))

    # The mode n = 0 has no tail
    for i, t in enumerate(times):
        coeffs[i, 0] = np.where(z < t, np.sin(config.omega * (t - z)), 0.0)

    # We distribute the other modes among the processes
    jobs = [(n, config.omega, times, z) for n in range(1, config.N_max)]
    with Pool(procs) as pool:
        for n, mode_coeffs in pool.imap_unordered(mode_worker, jobs, chunksize=1):
            for i, c in enumerate(mode_coeffs):
                coeffs[i, n] = c
    return coeffs


def field_from_coeffs(coeffs, config):
    '''Computes the value of the solution $u(t,x,z)$ at the cell centres x and the sub-samples z from its coefficients.

    Parameters
    ----------
    coeffs : np.ndarray of floats
        The coefficients at one time. The array is of shape (N_max, N_z * S_sub).
    config : SnapshotConfig
        The class containing all the parameters of the snapshots.

    Returns
    -------
    field : np.ndarray of floats
        The values of the field. The array is of shape (N_x, N_z * S_sub).
    '''
    n_values = np.arange(config.N_max)
    g_n_values = g_n_rect_delta(n_values, config) # g_0 and then 2 g_n for n >= 1
    x_values = (np.arange(config.N_x) + 0.5) * config.delta_x # Cell centres, as in stationary_carpets.py
    return np.cos(np.outer(2 * np.pi * n_values, x_values)).T @ (g_n_values[:, None] * coeffs)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    out_dir, d_lambda, l_lambda = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fractions = [float(a) for a in sys.argv[4].split(",")]
    config = SnapshotConfig(d_lambda, l_lambda)

    # We compute the field at each time t and at t - T/4, for the squared envelope
    T = config._lambda / config.c
    times = []
    for f in fractions:
        times += [f * config.z_T, f * config.z_T - T / 4]

    # The coefficients do not depend on the slit width: we compute them once per d/λ and set of times, and reuse them
    coeffs_file = os.path.join(out_dir, f"coeffs_d{d_lambda}_Nz{config.N_z}x{config.S_sub}_t{sys.argv[4].replace(',', '_')}.npy")
    if os.path.exists(coeffs_file):
        coeffs = np.load(coeffs_file)
    else:
        start = time.time()
        coeffs = generate_snapshot_coeffs(config, times)
        print(f"d/lambda={d_lambda}: {len(times)} times, {time.time() - start:.1f} s", flush=True)
        np.save(coeffs_file, coeffs)

    # We compute the stationary field |U|^2 at the sub-samples and its pixel averages, as in Figure 3
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stationary_carpets import field
    z_values = z_subsamples(config)
    x_values = (np.arange(config.N_x) + 0.5) * config.delta_x
    stationary_points = np.abs(field(d_lambda, l_lambda, x_values, z_values)) ** 2
    stationary = pixel_average(stationary_points, config)

    # We compute the snapshots and the pointwise rms differences from |U|^2 by range of z/t
    snapshots, rms = [], []
    bands = ((0.0, 0.25), (0.25, 0.5), (0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0))
    for i, f in enumerate(fractions):
        envelope = field_from_coeffs(coeffs[2 * i], config) ** 2 + field_from_coeffs(coeffs[2 * i + 1], config) ** 2
        snapshots.append(pixel_average(envelope, config))
        selections = [(z_values >= lo * f * config.z_T) & (z_values < hi * f * config.z_T) for lo, hi in bands]
        rms.append([np.sqrt(np.mean((envelope[:, s] - stationary_points[:, s]) ** 2) / np.mean(stationary_points[:, s] ** 2)) for s in selections])
    np.savez_compressed(os.path.join(out_dir, f"snapshots_d{d_lambda}_l{l_lambda}.npz"), fractions=np.array(fractions),
                        snapshots=np.array(snapshots), stationary=stationary, z_T=config.z_T, delta_z=config.delta_z,
                        d_lambda=d_lambda, l_lambda=l_lambda)

    # We print the rms relative differences from the stationary |U|^2 by range of z/t: pointwise (at the sub-samples;
    # the numbers quoted in Section 5.1) and between pixel averages (what Figure 2 shows, against Figure 3)
    z_pixels = (np.arange(config.N_z) + 0.5) * config.delta_z # Pixel centres
    vmax = (config.d / config.l) ** 2
    for f, envelope, pointwise in zip(fractions, snapshots, rms):
        for (lo, hi), r in zip(bands, pointwise):
            selection = (z_pixels >= lo * f * config.z_T) & (z_pixels < hi * f * config.z_T)
            relative = np.sqrt(np.mean((envelope[:, selection] - stationary[:, selection]) ** 2) / np.mean(stationary[:, selection] ** 2))
            print(f"  t = {f:.3f} z_T, z/t in [{lo:.2f},{hi:.2f}): rms rel. diff from |U|^2: pointwise {100 * r:5.2f}%, "
                  f"pixel averages {100 * relative:5.2f}%", flush=True)
        print(f"  t = {f:.3f} z_T: max {envelope.max() / vmax:.3f} at z = {z_pixels[np.unravel_index(envelope.argmax(), envelope.shape)[1]] / config._lambda:.1f} lambda, "
              f"pixels above 1: {int((envelope > vmax).sum())}, beyond front max {envelope[:, z_pixels >= f * config.z_T].max():.2e}")
