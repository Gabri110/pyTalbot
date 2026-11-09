# Scripts for the figures of the paper

These scripts reproduce the figures and the table of the paper *The Talbot Effect as a Causal Initial–Boundary-Value Problem: Formation, Stationary Limit and Paraxial Limit*, whose first version is the arXiv preprint [2507.00820](https://arxiv.org/abs/2507.00820). The section, equation, figure and table numbers refer to the current version of the paper. Run them from the root of the repository. We use units with $d = c = 1$ throughout, and $z_T = 2d^2/\lambda$ as in the paper.

| Script | Reproduces | Needs |
|---|---|---|
| `transient_snapshots.py` | The data of Figure 2: the transient carpet of the grating with $d/\lambda = 100$ and $l/\lambda = 5$ while it forms, at $t = z_T/4$, $z_T/2$ and $3z_T/4$ (squared envelope $u(t)^2 + u(t - T/4)^2$), and the root-mean-square differences from the stationary carpet quoted in Section 5.1 | numpy, scipy |
| `plot_snapshots.py` | Figure 2, from the output of `transient_snapshots.py` | numpy, matplotlib, Pillow |
| `validate_snapshots.py` | The check of the method of Figure 2 against direct quadrature of equation (C.1), quoted in Appendix C | numpy, scipy |
| `stationary_carpets.py` | Figure 3: the stationary carpets $\vert U\vert^2$, one image per panel. With `--check`, the numbers quoted in Section 5.2 | numpy, matplotlib |
| `convergence.py` | Table 1: direct quadrature on the real axis against contour deformation, for $d/\lambda = 100$ at $t = 2 z_T$ | numpy, scipy |
| `scan_rays.py` | The worst-case quadrature errors of the Gauss–Laguerre rays quoted in Appendix C | numpy |

scipy and Pillow are not in `requirements.txt`. You may install them through `pip install scipy pillow`.

```
mkdir -p results
python paper/transient_snapshots.py results 100 5 0.25,0.5,0.75
python paper/plot_snapshots.py results/snapshots_d100_l5.npz results/transient_forming_d_lambda=100_w_lambda=5.pdf results/transient_forming_d_lambda=100_w_lambda=5-1 0.25,0.5,0.75
python paper/validate_snapshots.py
python paper/stationary_carpets.py results results
python paper/stationary_carpets.py --check
python paper/convergence.py
python paper/scan_rays.py
```

## Numbers quoted in the paper

The scripts in `paper/numbers/` recompute the numbers that the paper quotes in its text. Each line of their output gives the value of the paper, the recomputed value and whether they agree, that is, whether the recomputed value, rounded as the paper rounds it, is the value of the paper. For "about $N\%$" we accept $N \pm 1$ percentage points. The run times are on a laptop; `front_bands.py` and `formation_times.py` use four processes.

| Script | Recomputes | Run time | Needs |
|---|---|---|---|
| `cj_constant.py` | The constant $C_J \approx 0.825$, $4\sqrt{31/10}\,C_J^2 \approx 4.79$ in (4.3) and $C(\omega,d) \approx 29\,\omega d/\lambda$ (Theorem 1, Appendix B) | 1 s | numpy, scipy |
| `paraxial_errors.py` | The exact $L^2$ error of the paraxial approximation for sharp-edged slits, and its floor at fixed $l/\lambda$ (Section 6.2) | 1 s | numpy |
| `peak_values.py` | The peaks of $\vert U\vert^2$ on the subimage planes and the Gibbs overshoot of the band-limited field (Section 5.2) | 10 s | numpy |
| `front_bands.py` | The differences between the transient carpets of Figure 2 and the stationary one, by range of $z/t$ (Section 5.1) | 5 min, or seconds with `--coeffs` | numpy, scipy |
| `dark_lines.py` | The mean intensity on the dark lines (7.3) of Figures 3(e) and 3(f) (Section 7.2) | 1 min | numpy |
| `formation_times.py` | The mean-square error of Theorem 1 at a fixed depth against the bound (4.3), and $\omega z^2 = \pi z_T z^2/d^2$ (remark after Theorem 1) | 2 s | numpy, scipy |

By default `front_bands.py` computes the coefficients on every 4th depth of the grid of Figure 2. With `--coeffs` it instead reads the file that `transient_snapshots.py` saves, which is on the full grid.

```
python paper/numbers/cj_constant.py
python paper/numbers/paraxial_errors.py
python paper/numbers/peak_values.py
python paper/numbers/front_bands.py --stride 4
python paper/numbers/front_bands.py --coeffs results/coeffs_d100_Nz1440x16_t0.25_0.5_0.75.npy
python paper/numbers/dark_lines.py
python paper/numbers/formation_times.py
```

## Method

The transient carpets use `pyTalbot/transient_fast.py`. It evaluates each mode $c_n(t,z)$ at a fixed time with a different method from `perform_integrals.py`, explained in Appendix C of the paper.
* **Splitting.** It writes $c_n$ as the stationary term plus a tail integral.
* **Contour deformation.** It moves the path of integration of the tail into the complex plane, where the integrand decays exponentially and a few Gauss–Laguerre nodes suffice.

Therefore the cost no longer grows with the number of oscillations of the integrand, which is between $2 \times 10^4$ and $2.4 \times 10^5$ for every mode and depth of the carpets with $d/\lambda = 100$ at $t = 2 z_T$.

**Domain of validity.** The method is accurate to about $10^{-9}$ wherever $k_n\sqrt{t^2-z^2} \geq 100$. While the carpet forms there is a thin layer behind the front $z = t$ where this fails. There `transient_snapshots.py` integrates equation (C.1) directly by Gauss–Legendre quadrature, which is cheap because $t - z$ is small.

**Agreement with other methods.** At $t = 2 z_T$ the method agrees with `perform_integrals.py` to within $3\times10^{-9}$ for $d/\lambda = 5$ and $10$. At the times of Figure 2 it agrees with direct quadrature of (C.1) to within $3\times10^{-10}$ for $d/\lambda = 10$ and $100$ (`validate_snapshots.py`).

**Tests.** `tests/test_transient_fast.py` checks the module against direct quadrature on the real axis. It needs numpy only.
