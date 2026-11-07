'''Checks of pyTalbot/transient_fast.py against direct quadrature on the real axis. They only need numpy.

Run them with `python tests/test_transient_fast.py`, or with `pytest`.
We use units with d = c = 1 and d/lambda = 5, at t = 2 z_T = 20.
'''
import os, sys
import warnings
import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from pyTalbot.transient_fast import transient_mode, transient_field


# Parameters of the checks
d_lambda = 5
omega = 2 * np.pi * d_lambda # Angular frequency of the source
z_T = 2. * d_lambda # Talbot distance = 2 d^2/λ
t = 2 * z_T # Time at which we compare
tolerance = 1e-9

# The (n, z) we check: a propagating mode, a propagating mode whose front has not arrived (so that the real segment
# is used), the grazing mode k_n = omega and an evanescent mode just above the cut-off
cases = [(1, 5.0), (4, 10.0), (4, 15.0), (5, 8.0), (6, 12.0)]


def bessel_j1(x):
    '''Computes J_1(x) = (1/π) int_0^π cos(θ - x sin θ) dθ by the trapezoidal rule, which is spectrally accurate here.

    Parameters
    ----------
    x : np.ndarray of floats
        The points at which we compute J_1.

    Returns
    -------
    j1 : np.ndarray of floats
        The values of J_1(x), flattened.
    '''
    x = np.asarray(x, dtype=float)
    m = int(np.max(np.abs(x))) + 60
    theta = np.linspace(0.0, np.pi, m + 1)
    weights = np.full(m + 1, np.pi / m)
    weights[[0, -1]] *= 0.5
    return np.cos(theta[None, :] - x.ravel()[:, None] * np.sin(theta[None, :])) @ weights / np.pi


def direct_mode(k_n, t, z, omega, per_oscillation=8):
    '''Computes the coefficient $c_n(t,z)$ without the g_n factor by composite 16-point Gauss-Legendre quadrature of
    sin(omega (t-z)) - k_n z int_0^R J_1(k_n r) sin(omega (t - rho))/rho dr, with R = sqrt(t^2 - z^2) and rho = sqrt(r^2 + z^2).

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
    per_oscillation : int, optional
        Number of panels per oscillation of the integrand.

    Returns
    -------
    coeff : float
        The value of the coefficient.
    '''
    R = np.sqrt(t * t - z * z)
    panels = int(per_oscillation * (k_n * R + omega * (t - z)) / (2 * np.pi)) + 16
    x_g, w_g = leggauss(16)
    edges = np.linspace(0.0, R, panels + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    r = (mid[:, None] + half[:, None] * x_g).ravel()
    weights = (half[:, None] * w_g).ravel()
    rho = np.sqrt(r * r + z * z)
    bessel = np.concatenate([bessel_j1(k_n * r[i:i + 2000]) for i in range(0, r.size, 2000)]) # In chunks to save memory
    return np.sin(omega * (t - z)) - k_n * z * np.sum(weights * bessel * np.sin(omega * (t - rho)) / rho)


def test_against_direct_quadrature():
    for n, z in cases:
        k_n = 2 * np.pi * n
        with warnings.catch_warnings():
            warnings.simplefilter("error") # All the cases have k_n sqrt(t^2 - z^2) >= 100, so there must be no warning
            fast = transient_mode(k_n, t, z, omega)
        reference = direct_mode(k_n, t, z, omega)
        assert abs(fast - reference) < tolerance, (n, z, fast, reference)


def test_array_input_matches_scalar_input():
    k_n = 2 * np.pi * 3
    z = np.array([2.0, 7.0, 13.0])
    values = transient_mode(k_n, t, z, omega)
    assert values.shape == z.shape
    for z_j, value_j in zip(z, values):
        assert abs(transient_mode(k_n, t, z_j, omega) - value_j) < 1e-14


def test_boundary_and_wavefront():
    k_n = 2 * np.pi * 2
    z = np.array([0.0, t, t + 1.0])
    values = transient_mode(k_n, t, z, omega)
    assert abs(values[0] - np.sin(omega * t)) < 1e-14 # The boundary condition at the grating
    assert np.all(values[1:] == 0.0) # The wave has not reached z >= t yet


def test_warns_close_to_the_wavefront():
    k_n = 2 * np.pi
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transient_mode(k_n, t, t - 0.01, omega) # Here k_n sqrt(t^2 - z^2) is about 4
    assert any(issubclass(c.category, RuntimeWarning) for c in caught)


def test_field_is_the_mode_sum():
    n = np.arange(4)
    g_n = np.array([1.0, 0.5, -0.25, 0.1])
    k_n = 2 * np.pi * n
    x = np.array([0.0, 0.1, 0.3])
    z = np.array([1.0, 4.0])
    field = transient_field(g_n, k_n, t, x, z, omega)
    expected = g_n[0] * np.sin(omega * (t - z))[None, :]
    for g, k in zip(g_n[1:], k_n[1:]):
        expected = expected + g * np.outer(np.cos(k * x), transient_mode(k, t, z, omega))
    assert np.allclose(field, expected, atol=1e-14)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"{name}: ok")
