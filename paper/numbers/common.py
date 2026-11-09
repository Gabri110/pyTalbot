'''Helpers shared by the scripts of paper/numbers: the coefficients of the binary grating and the comparison of each
recomputed number with the value that the paper quotes.

Every script of this folder prints one line per number, of the form
    <label>   paper: <value quoted in the paper>   recomputed: <value>   -> agrees / DIFFERS
A number agrees when the recomputed value, rounded as the paper rounds it, is the value of the paper. For a number that
the paper gives as "about x" we accept the interval [lo, hi] stated in each call, since "about" has no fixed precision.
'''
import numpy as np


def g_n_two_sided(n, eta):
    '''Computes the two-sided coefficients g_n of the binary grating g = A d/l on |x| < l/2, with A = 1.

    Parameters
    ----------
    n : np.ndarray of ints
        The ns for which we compute g_n. Unlike g_n_rect_delta, we do not double g_n for n =/= 0.
    eta : float
        The ratio l/d.

    Returns
    -------
    g_n_values : np.ndarray of floats
        g_0 = 1 and g_n = sin(n π eta)/(n π eta) for n =/= 0.
    '''
    n = np.asarray(n, dtype=float)
    g_n_values = np.ones_like(n)
    nonzero = n != 0
    g_n_values[nonzero] = np.sin(n[nonzero] * np.pi * eta) / (n[nonzero] * np.pi * eta)
    return g_n_values


def check(label, paper, value, lo, hi, fmt="{:.4g}"):
    '''Prints the comparison of one number with the paper. The number agrees when lo <= value <= hi.

    Parameters
    ----------
    label : string
        What the number is.
    paper : string
        The value quoted in the paper.
    value : float
        The recomputed value.
    lo, hi : float
        The interval of values that agree with the paper.
    fmt : string, optional
        The format of the recomputed value.

    Returns
    -------
    agrees : bool
        Whether the number agrees with the paper.
    '''
    agrees = lo <= value <= hi
    print(f"  {label:<62s} paper: {paper:<12s} recomputed: {fmt.format(value):<12s} -> "
          f"{'agrees' if agrees else 'DIFFERS'}", flush=True)
    return agrees


def check_rounded(label, paper, value, decimals, scale=1.0, fmt=None):
    '''Prints the comparison of one number with the paper, to the precision of the paper. The number agrees when
    round(value * scale, decimals) is the value of the paper.

    Parameters
    ----------
    label : string
        What the number is.
    paper : string
        The value quoted in the paper, without units or %.
    value : float
        The recomputed value.
    decimals : int
        The number of decimals of the value of the paper.
    scale : float, optional
        The factor between value and the value of the paper, e.g. 100 for a percentage.
    fmt : string, optional
        The format of the recomputed value. By default we print two more decimals than the paper.

    Returns
    -------
    agrees : bool
        Whether the number agrees with the paper.
    '''
    shown = round(value * scale, decimals)
    agrees = abs(shown - float(paper)) < 1e-9
    fmt = fmt or "{:." + str(decimals + 2) + "f}"
    print(f"  {label:<62s} paper: {paper:<12s} recomputed: {fmt.format(value * scale):<12s} -> "
          f"{'agrees' if agrees else 'DIFFERS'}", flush=True)
    return agrees
