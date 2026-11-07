'''Figure 3 of the paper: the stationary Talbot carpets |U|^2, one image per panel.

Usage, from the root of the repository:
    python paper/stationary_carpets.py OUT_PDF_DIR OUT_PNG_DIR    (writes stationary_d_lambda=..._l_lambda=....pdf and ...-1.png)
    python paper/stationary_carpets.py --check                    (prints the numbers quoted in Section 5.2)

This script only needs numpy and matplotlib. U is the stationary field (4.1) of the paper for the binary grating
g = A d/l on |x| < l/2, truncated at |n| < N = 5 d/lambda as in Section 5:
    U(x,z) = sum_{|n| < N} g_n cos(k_n x) exp(-i z sqrt(omega^2 - k_n^2)),   sqrt(omega^2 - k^2) := -i sqrt(k^2 - omega^2) for k > omega,
    g_0 = A,  g_n = A (d/l) sin(n π l/d)/(n π),  k_n = 2 π n/d,  omega = 2 π/λ (c = 1),  z_T = 2 d^2/λ.

Panels (a)-(f) have narrow slits, l/λ = 2 or 5. Panels (g) and (h) keep l/d = 0.2 fixed, so that l >> λ and
λ d/l^2 = 25 λ/d decreases as d/λ grows (0.25 and 0.0625): they approach the regime of Sections 6 and 7.

Grid: N_x = 405 points per half period (cell centres) and N_z = 1440 pixels over 0 < z < z_T. For d/λ >= 100 a pixel
is several wavelengths wide in z, so point samples of |U|^2 would alias the structure on the scale of the wavelength.
Therefore every pixel shows the AVERAGE of |U|^2 over its width in z (S = 16 equally spaced sub-samples) and over
3 samples in x. The colour scale is in units of (A d/l)^2 and saturates at 1 (the overflow is marked on the colour bar).
The axes and the colour bar follow the conventions of Figure 2 (plot_snapshots.py).
'''
import os, sys
import numpy as np


##################################
# Configuration of the figure.
##################################

A = 1. # Amplitude of the source
N_x, N_z, S_sub = 405, 1440, 16 # Samples in x, pixels in z and sub-samples per pixel

#         d/λ, l/λ, label of the slit width
panels = [(5, 2, r"$l/\lambda = 2$"),
          (10, 5, r"$l/\lambda = 5$"),
          (20, 5, r"$l/\lambda = 5$"),
          (20, 2, r"$l/\lambda = 2$"),
          (100, 5, r"$l/\lambda = 5$"),
          (100, 2, r"$l/\lambda = 2$"),
          (100, 20, r"$l/d = 0.2$"),
          (400, 80, r"$l/d = 0.2$")]


def coefficients(d_lambda, l_lambda):
    '''Computes the one-sided coefficients of the grating and the propagation constants of the modes (d = 1).

    Parameters
    ----------
    d_lambda : int
        The ratio d/λ.
    l_lambda : int
        The ratio l/λ.

    Returns
    -------
    n_values : np.ndarray of ints
        The ns, 0 <= n < N = 5 d/λ.
    g_n_values : np.ndarray of floats
        g_0 and then g_n + g_{-n} = 2 g_n for n >= 1.
    k_n_values : np.ndarray of floats
        The values of the k_n.
    kz_values : np.ndarray of floats
        sqrt(omega^2 - k_n^2) for the propagating modes and 0 for the evanescent ones.
    kappa_values : np.ndarray of floats
        sqrt(k_n^2 - omega^2) for the evanescent modes and 0 for the propagating ones.
    z_T : float
        The Talbot distance 2 d^2/λ.
    eta : float
        The ratio l/d.
    '''
    _lambda = 1.0 / d_lambda
    omega = 2 * np.pi / _lambda
    eta = l_lambda / d_lambda
    n_values = np.arange(int(round(5 * d_lambda)))

    # We multiply by 2 to account at the same time for g_n and g_(-n)
    g_n_values = np.empty(n_values.size)
    g_n_values[0] = A
    g_n_values[1:] = 2 * A * np.sin(n_values[1:] * np.pi * eta) / (n_values[1:] * np.pi * eta)

    k_n_values = 2 * np.pi * n_values
    kz_values = np.where(k_n_values <= omega, np.sqrt(np.maximum(omega ** 2 - k_n_values ** 2, 0.0)), 0.0) # Propagating modes
    kappa_values = np.where(k_n_values > omega, np.sqrt(np.maximum(k_n_values ** 2 - omega ** 2, 0.0)), 0.0) # Evanescent modes
    return n_values, g_n_values, k_n_values, kz_values, kappa_values, 2.0 / _lambda, eta


def field(d_lambda, l_lambda, x, z):
    '''Computes the stationary field U(x,z) on the grid x by z, in units of A.

    Parameters
    ----------
    d_lambda : int
        The ratio d/λ.
    l_lambda : int
        The ratio l/λ.
    x : np.ndarray of floats
        The xs at which we compute the field.
    z : np.ndarray of floats
        The zs at which we compute the field.

    Returns
    -------
    field : np.ndarray of complex
        The values of U at each x,z. The array is of shape (len(x), len(z)).
    '''
    n_values, g_n_values, k_n_values, kz_values, kappa_values, z_T, eta = coefficients(d_lambda, l_lambda)
    cos_values = np.cos(np.outer(x, k_n_values)) * g_n_values # Shape (N_x, N)
    propagation = np.exp(-1j * np.outer(kz_values, z) - np.outer(kappa_values, z)) # Shape (N, N_z)
    return (cos_values @ propagation.real) + 1j * (cos_values @ propagation.imag)


def carpet(d_lambda, l_lambda, n_x=N_x, n_z=N_z, s_sub=S_sub):
    '''Computes |U|^2/(A d/l)^2 on [0, d/2] x [0, z_T], averaged over the width of each pixel in z.

    Parameters
    ----------
    d_lambda : int
        The ratio d/λ.
    l_lambda : int
        The ratio l/λ.
    n_x, n_z, s_sub : int, optional
        The number of samples in x, of pixels in z and of sub-samples per pixel.

    Returns
    -------
    carpet : np.ndarray of floats
        The values of the carpet. The array is of shape (n_x, n_z).
    '''
    z_T = 2.0 * d_lambda
    eta = l_lambda / d_lambda
    x = (np.arange(n_x) + 0.5) * (0.5 / n_x)
    result = np.empty((n_x, n_z))

    # We compute the pixels in blocks to avoid memory problems
    block = 180
    for j_0 in range(0, n_z, block):
        j = np.arange(j_0, min(j_0 + block, n_z))
        z = ((j[:, None] + (np.arange(s_sub)[None, :] + 0.5) / s_sub) * (z_T / n_z)).ravel()
        intensity = np.abs(field(d_lambda, l_lambda, x, z)) ** 2
        result[:, j] = intensity.reshape(n_x, j.size, s_sub).mean(axis=2)
    return result * eta ** 2 / A ** 2


def full_period(half):
    '''Extends a carpet from 0 <= x <= d/2 to -d <= x <= d, using that the field is even about x = 0 and about x = d/2.

    Parameters
    ----------
    half : np.ndarray of floats
        The carpet on [0, d/2]. The array is of shape (N_x, N_z).

    Returns
    -------
    full : np.ndarray of floats
        The carpet on [-d, d]. The array is of shape (4 N_x, N_z).
    '''
    one_period = np.vstack([half, half[::-1]]) # [0, d]
    return np.vstack([one_period[::-1], one_period]) # [-d, d]


def plane_peak(d_lambda, l_lambda, zeta, n_x=4001):
    '''Computes the maximum over x of |U|^2/(A d/l)^2 on the plane z = zeta z_T/2, from point values.

    Parameters
    ----------
    d_lambda : int
        The ratio d/λ.
    l_lambda : int
        The ratio l/λ.
    zeta : float
        The depth in units of z_T/2.
    n_x : int, optional
        The number of points in [0, d/2].

    Returns
    -------
    peak : float
        The maximum.
    '''
    x = np.linspace(0, 0.5, n_x)
    U = field(d_lambda, l_lambda, x, np.array([zeta * d_lambda]))
    return float((np.abs(U) ** 2).max() * (l_lambda / d_lambda) ** 2)


def check():
    '''Prints the numbers quoted in Section 5.2 of the paper and checks field() against the two-sided series.'''
    print("q = 1 peak of |U|^2/(Ad/l)^2 on z = z_T:")
    for d_lambda, l_lambda in ((100, 2), (10, 5), (20, 5), (100, 5), (100, 20), (400, 80)):
        print(f"  d/lambda = {d_lambda:3d}, l/lambda = {l_lambda:2d}: {plane_peak(d_lambda, l_lambda, 2.0):.3f}")
    print("panels (g), (h): peaks on z = z_T, z_T/2 (q = 1), z_T/4 (q = 2), z_T/6 (q = 3); paraxial value 1/q:")
    for d_lambda, l_lambda in ((100, 20), (400, 80)):
        print(f"  d/lambda = {d_lambda:3d}, l/d = 0.2: " + ", ".join(f"{plane_peak(d_lambda, l_lambda, zt):.2f}" for zt in (2.0, 1.0, 0.5, 1 / 3)))

    # We evaluate U at random points independently, directly from the two-sided series
    rng = np.random.default_rng(1)
    for d_lambda, l_lambda in ((20, 2), (400, 80)):
        _lambda = 1.0 / d_lambda
        omega = 2 * np.pi / _lambda
        eta = l_lambda / d_lambda
        x = rng.uniform(0, 0.5, 3)
        z = rng.uniform(0, 2 * d_lambda, 3)
        n = np.arange(-(5 * d_lambda) + 1, 5 * d_lambda)
        g = np.where(n == 0, A, A * np.sin(np.pi * np.where(n == 0, 1, n) * eta) / (np.pi * np.where(n == 0, 1, n) * eta))
        k = 2 * np.pi * n
        root = np.where(np.abs(k) <= omega, np.sqrt(np.abs(omega ** 2 - k ** 2)) + 0j, -1j * np.sqrt(np.abs(k ** 2 - omega ** 2)))
        direct = np.array([(g * np.cos(k * x_i) * np.exp(-1j * z_i * root)).sum() for x_i, z_i in zip(x, z)])
        ours = np.array([field(d_lambda, l_lambda, np.array([x_i]), np.array([z_i]))[0, 0] for x_i, z_i in zip(x, z)])
        print(f"  two-sided series vs field(): d/lambda = {d_lambda}: max |difference| = {np.abs(direct - ours).max():.1e}")


def plot_carpets(out_pdf_dir, out_png_dir):
    '''Plots one image per panel and saves it as stationary_d_lambda=<d/λ>_l_lambda=<l/λ>.pdf and ...-1.png.

    Parameters
    ----------
    out_pdf_dir : string
        Folder where the PDF images are stored. The folder MUST exist.
    out_png_dir : string
        Folder where the PNG images are stored. The folder MUST exist.

    Returns
    -------
    None
    '''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = 1 / 2.54
    for i, (d_lambda, l_lambda, slit) in enumerate(panels):
        intensity = carpet(d_lambda, l_lambda)
        print(f"({'abcdefgh'[i]}) d/lambda = {d_lambda:3d}, l/lambda = {l_lambda:2d}: max = {intensity.max():.2f}, "
              f"fraction of pixels above 1: {np.mean(intensity > 1):.4f}", flush=True)
        image = full_period(intensity.reshape(N_x // 3, 3, N_z).mean(axis=1)) # 3 samples in x per row of the image

        fig, ax = plt.subplots(figsize=(8.0 * cm, 4.3 * cm))
        im = ax.imshow(image, cmap="turbo", vmin=0, vmax=1, interpolation="none", origin="lower",
                       extent=[0, 1, -1, 1], aspect="auto")

        # Label the axes and set the ticks
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["$-d$", "$-d/2$", "$0$", "$d/2$", "$d$"], fontsize=7)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["$0$", "$z_T/4$", "$z_T/2$", "$3z_T/4$", "$z_T$"], fontsize=7)
        ax.set_xlabel("$z$", fontsize=8, labelpad=0.5)
        ax.tick_params(length=2.5, width=0.6)

        # Colour bar
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025, aspect=22, extend="max",
                            ticks=[0, 0.25, 0.5, 0.75, 1])
        cbar.ax.set_yticklabels(["$0$", "$0.25$", "$0.5$", "$0.75$", "$1$"], fontsize=7)
        cbar.ax.tick_params(length=2, width=0.6)
        cbar.set_label(r"$|U|^2\,/\,(A d/l)^2$", fontsize=8)

        base = f"stationary_d_lambda={d_lambda}_l_lambda={l_lambda}"
        fig.savefig(os.path.join(out_pdf_dir, base + ".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=400)
        im.set_interpolation("antialiased") # The PNG is averaged over the area of each pixel when rasterised
        im.set_interpolation_stage("data")
        fig.savefig(os.path.join(out_png_dir, base + "-1.png"), bbox_inches="tight", pad_inches=0.02, dpi=400)
        plt.close(fig)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif len(sys.argv) < 3:
        sys.exit(__doc__)
    else:
        plot_carpets(sys.argv[1], sys.argv[2])
