'''Fast evaluation of the transient solution $u(t,x,z)$ at a fixed time t.

The integrator in perform_integrals.py computes the coefficients
    c_n(t,z) = sin(omega (t-z)) - k_n z int_z^t J_1(k_n sqrt(tau^2-z^2))/sqrt(tau^2-z^2) sin(omega (t-tau)) dtau
by adaptive quadrature along the real axis. The integrand oscillates about (k_n R + omega (t-z))/(2 pi) times,
with R = sqrt(t^2 - z^2), so the cost of a whole carpet grows like (d/lambda)^3. For d/lambda = 100 this needs a cluster.

Here we write instead
    c_n(t,z) = S_n(t,z) + E_n(t,z),   E_n = k_n z Im[e^{i omega t} F],   F = int_R^inf J_1(k_n r) e^{-i omega rho}/rho dr,
where S_n is the stationary coefficient and rho = sqrt(r^2 + z^2). Writing J_1 = (H_1^(1) + H_1^(2))/2, each half of F
is an oscillatory integral that decays exponentially along a vertical ray of the complex r-plane, where a few
Gauss-Laguerre nodes suffice. For the propagating modes whose stationary point r* = z k_n/sqrt(omega^2 - k_n^2) lies
beyond R (their front has not arrived yet), we first integrate the H_1^(1) half along the real axis past r*.
The method is explained in Appendix C of the paper, "From an oscillatory to an exponentially decaying integral".

Domain of validity: we evaluate the Hankel functions through the first four terms of their asymptotic series, whose
relative error is below 1.5e-9 for |k_n r| >= 100. Therefore the results are accurate to about 1e-9 whenever
k_n sqrt(t^2 - z^2) >= 100. In the thin layer behind the wavefront where this fails, transient_mode warns; there we
integrate the defining formula directly instead, as paper/transient_snapshots.py does for Figure 2 of the paper.
At t = 2 z_T the results agree with those of perform_integrals.py to within 3e-9 for d/lambda = 5 and 10.
'''
import warnings
import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss

# Coefficients a_k(nu = 1) of the asymptotic series of the Hankel functions
HANKEL_COEFFS = (1.0, 3.0 / 8.0, -15.0 / 128.0, 315.0 / 3072.0)

# Nodes and weights of the Gauss-Laguerre (rays) and Gauss-Legendre (real segment) quadratures
LAGUERRE_X, LAGUERRE_W = laggauss(48)
LEGENDRE_X, LEGENDRE_W = leggauss(12)


def hankel_series(x, sign):
    '''Computes the asymptotic series sum_k (sign i)^k a_k / x^k of the Hankel functions of order 1.

    Parameters
    ----------
    x : np.ndarray of complex
        The (complex) arguments k_n r.
    sign : int
        +1 for H_1^(1) and -1 for H_1^(2).

    Returns
    -------
    series : np.ndarray of complex
        The value of the series at each x.
    '''
    inverse_x = 1.0 / x
    series = np.zeros_like(x, dtype=complex)
    power = np.ones_like(x, dtype=complex)
    for a_k in HANKEL_COEFFS:
        series = series + a_k * power
        power = power * (sign * 1j) * inverse_x
    return series


def hankel_integrand(r, k_n, z, omega, sign):
    '''Computes the integrand H_1(k_n r) e^{-i omega rho}/rho of F, written as amplitude * exp(exponent).

    Parameters
    ----------
    r : np.ndarray of complex
        The points of the complex r-plane at which we evaluate the integrand.
    k_n : float
        The wave number of the mode.
    z : np.ndarray of floats
        The zs at which we compute the field.
    omega : float
        The angular frequency of the source.
    sign : int
        +1 for H_1^(1) and -1 for H_1^(2).

    Returns
    -------
    amplitude : np.ndarray of complex
        The slowly varying factor of the integrand.
    exponent : np.ndarray of complex
        The exponent of its oscillating (or decaying) factor.
    '''
    x = k_n * r
    rho = np.sqrt(r * r + z * z)
    amplitude = np.sqrt(2.0 / (np.pi * x)) * hankel_series(x, sign) / rho
    if sign > 0:
        # We write k_n r - omega rho = (k_n - omega) r - omega z^2/(r + rho) to avoid cancellations when k_n is close to omega and |r| is large
        exponent = 1j * ((k_n - omega) * r - omega * z * z / (r + rho) - 0.75 * np.pi)
    else:
        exponent = -1j * (x - 0.75 * np.pi) - 1j * omega * rho
    return amplitude, exponent


def ray_integral(r_0, k_n, z, omega, sign, direction, decay):
    '''Computes the integral of the integrand along the vertical ray r = r_0 + direction i s, s >= 0, by Gauss-Laguerre quadrature.

    Parameters
    ----------
    r_0 : np.ndarray of floats
        The starting points of the rays on the real axis.
    k_n : float
        The wave number of the mode.
    z : np.ndarray of floats
        The zs at which we compute the field.
    omega : float
        The angular frequency of the source.
    sign : int
        +1 for H_1^(1) and -1 for H_1^(2).
    direction : float
        +1. for a ray going up and -1. for a ray going down.
    decay : np.ndarray of floats
        A lower bound of the decay rate of the integrand along the ray, used to scale the nodes.

    Returns
    -------
    integral : np.ndarray of complex
        The value of the integral for each z.
    '''
    # The nodes have shape (nodes, len(z)), or (nodes,) for a scalar z
    s = LAGUERRE_X[:, None] / decay
    r = r_0 + direction * 1j * s
    amplitude, exponent = hankel_integrand(r, k_n, z, omega, sign)
    values = amplitude * np.exp(exponent + LAGUERRE_X[:, None])
    return (direction * 1j / decay) * np.sum(LAGUERRE_W[:, None] * values, axis=0)


def real_segment_integral(r_a, r_b, k_n, z, omega, sign):
    '''Computes the integral of the integrand on the real segment [r_a, r_b] by composite Gauss-Legendre quadrature.
    The panels are uniform in v = 1/r, where the phase varies almost uniformly.

    Parameters
    ----------
    r_a, r_b : float
        The ends of the segment.
    k_n : float
        The wave number of the mode.
    z : float
        The z at which we compute the field.
    omega : float
        The angular frequency of the source.
    sign : int
        +1 for H_1^(1) and -1 for H_1^(2).

    Returns
    -------
    integral : complex
        The value of the integral.
    '''
    # We estimate the total variation of the phase to choose the number of panels
    r_values = np.linspace(r_a, r_b, 257)
    rho = np.sqrt(r_values * r_values + z * z)
    dphase_dr = np.abs(k_n - omega * r_values / rho)
    dphase_dv = np.max(r_values * r_values * dphase_dr)
    v_a, v_b = 1.0 / r_b, 1.0 / r_a
    panels = int(np.ceil(dphase_dv * (v_b - v_a) / np.pi)) + 8

    # We build the nodes and weights of the composite quadrature
    edges = np.linspace(v_a, v_b, panels + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    half = 0.5 * (edges[1:] - edges[:-1])
    v = (mid[:, None] + half[:, None] * LEGENDRE_X[None, :]).ravel()
    weights = (half[:, None] * LEGENDRE_W[None, :]).ravel()

    amplitude, exponent = hankel_integrand(1.0 / v, k_n, z, omega, sign)
    return np.sum(weights * amplitude * np.exp(exponent) / (v * v))


def transient_mode(k_n, t, z, omega):
    '''Computes the coefficients $c_n(t,z)$ without the g_n factor for one wave number k_n > 0 at a fixed time t.
    Behind the wavefront (0 < z < t) we add the tail E_n to the stationary coefficient, at z = 0 we get sin(omega t)
    and ahead of the wavefront (z >= t) the coefficient vanishes.

    Parameters
    ----------
    k_n : float
        The wave number of the mode. Must be positive.
    t : float
        The time at which we compute the field.
    z : np.ndarray of floats or float
        The zs at which we compute the field.
    omega : float
        The angular frequency of the source.

    Returns
    -------
    coeff : np.ndarray of floats or float
        The values of the coefficients $c_n(t,z)$ without the g_n factor, with the same shape as z.
    '''
    scalar = np.ndim(z) == 0
    z = np.atleast_1d(np.asarray(z, dtype=float))
    coeff = np.zeros_like(z)
    behind = z < t

    # We compute the stationary coefficient
    if k_n <= omega:
        coeff[behind] = np.sin(omega * t - z[behind] * np.sqrt(omega * omega - k_n * k_n))
    else:
        coeff[behind] = np.sin(omega * t) * np.exp(-z[behind] * np.sqrt(k_n * k_n - omega * omega))

    # Only the points with 0 < z < t have a tail
    inside = behind & (z > 0)
    if not np.any(inside):
        return coeff[0] if scalar else coeff
    z_in = z[inside]
    R = np.sqrt(t * t - z_in * z_in)

    kR = k_n * R.min()
    if kR < 100:
        warnings.warn(f"transient_mode: k sqrt(t^2 - z^2) = {kR:.3g} < 100, where the asymptotic Hankel series "
                      f"has a relative error of about {0.15 / kR ** 4:.1e}; use perform_integrals near the wavefront",
                      RuntimeWarning, stacklevel=2)

    # The H_1^(2) half has phase -(k_n r + omega rho) and decays in the lower half plane at a rate >= k_n + omega R/t
    F_2 = ray_integral(R, k_n, z_in, omega, -1, -1.0, k_n + omega * R / t)

    # The H_1^(1) half depends on the kind of mode
    F_1 = np.empty_like(z_in, dtype=complex)
    if abs(k_n - omega) <= 1e-9 * omega:
        # Grazing mode: the integrand only decays like s^{-3/2}. We map s = R y^2/(1-y)^2, which keeps the integrand
        # smooth and bounded at y = 1 (s = R y/(1-y) would leave a (1-y)^{-1/2} singularity), and use Gauss-Legendre
        x_l, w_l = leggauss(200)
        y = 0.5 * (x_l + 1.0)
        w_y = 0.5 * w_l
        for j in range(z_in.size):
            s = R[j] * y ** 2 / (1.0 - y) ** 2
            ds = 2.0 * R[j] * y / (1.0 - y) ** 3
            amplitude, exponent = hankel_integrand(R[j] + 1j * s, k_n, z_in[j], omega, +1)
            F_1[j] = 1j * np.sum(w_y * ds * amplitude * np.exp(exponent))
    elif k_n > omega:
        # Evanescent mode: the decay rate along the ray falls from k_n - omega R/t (at s = 0) to k_n - omega (as s -> inf),
        # but only on the scale s ~ t^2/z, so it is much more efficient to scale the nodes with the local rate
        F_1[:] = ray_integral(R, k_n, z_in, omega, +1, +1.0, k_n - omega * R / t)
    else:
        # Propagating mode: if the stationary point r* lies beyond R, we first integrate along the real axis up to 1.25 r*
        r_star = z_in * k_n / np.sqrt(omega * omega - k_n * k_n)
        r_c = np.maximum(R, 1.25 * r_star)
        direct = r_c == R
        if np.any(direct):
            idx = np.where(direct)[0]
            decay = omega * R[idx] / t - k_n
            F_1[idx] = ray_integral(R[idx], k_n, z_in[idx], omega, +1, -1.0, decay)
        for j in np.where(~direct)[0]:
            segment = real_segment_integral(R[j], r_c[j], k_n, z_in[j], omega, +1)
            decay = omega * r_c[j] / np.sqrt(r_c[j] ** 2 + z_in[j] ** 2) - k_n
            F_1[j] = segment + ray_integral(np.array([r_c[j]]), k_n, np.array([z_in[j]]), omega, +1, -1.0, np.array([decay]))[0]

    # We add the tail E_n to the stationary coefficient
    F = 0.5 * (F_1 + F_2)
    coeff[inside] += k_n * z_in * np.imag(np.exp(1j * omega * t) * F)
    return coeff[0] if scalar else coeff


def transient_field(g_n, k_n, t, x, z, omega):
    '''Computes the value of the solution $u(t,x,z)$ at a fixed time t for each value of x and z.
    We follow the convention of stationary_utils.py: n >= 0 and g_n already doubled for n >= 1.

    Parameters
    ----------
    g_n : np.ndarray of floats
        The values of the g_n.
    k_n : np.ndarray of floats
        The values of the k_n.
    t : float
        The time at which we compute the field.
    x : np.ndarray of floats
        The xs at which we compute the field.
    z : np.ndarray of floats
        The zs at which we compute the field.
    omega : float
        The angular frequency of the source.

    Returns
    -------
    field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each x,z. The array is of shape (N_x, N_z).
    '''
    field = np.zeros((x.size, z.size))
    for g, k in zip(g_n, k_n):
        if g == 0.0:
            continue
        if k == 0.0:
            coeff = np.where(z <= t, np.sin(omega * (t - z)), 0.0) # The mode n = 0 has no tail
        else:
            coeff = transient_mode(k, t, z, omega)
        field += g * np.outer(np.cos(k * x), coeff)
    return field
