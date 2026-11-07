'''Checks the method of transient_snapshots.py against direct Gauss-Legendre quadrature of equation (C.1) of the paper
at the times of Figure 2 (t = z_T/4, z_T/2 and 3z_T/4), for d/λ = 10 and 100 and selected modes and depths.

Usage, from the root of the repository:
    python paper/validate_snapshots.py

The modes include those near the cut-off, and the depths lie on both sides of the switch at k_n sqrt(t^2 - z^2) = 100
and of the front z = v_n t of each mode. The paper quotes the largest difference in Appendix C.
'''
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transient_snapshots import SnapshotConfig, snapshot_mode, direct_mode

rng = np.random.default_rng(7)
for d_lambda in (10, 100):
    config = SnapshotConfig(d_lambda, 5)
    omega = config.omega
    worst = (0, None)
    n_values = sorted(set([1, 2, d_lambda // 4, d_lambda // 2, int(0.8 * d_lambda), int(0.9 * d_lambda), d_lambda - 2,
                           d_lambda - 1, d_lambda + 1, d_lambda + 3, int(1.5 * d_lambda)]))
    for fraction in (0.25, 0.5, 0.75):
        t = fraction * config.z_T
        for n in n_values:
            k_n = 2 * np.pi * n
            v_n = np.sqrt(max(0.0, 1 - (k_n / omega) ** 2)) # Group velocity of the mode

            # Random depths, and depths just behind and ahead of the front of the mode
            z_values = list(rng.uniform(0.02, 0.98, 4) * t) + [0.97 * v_n * t, 1.03 * v_n * t] if v_n > 0 else list(rng.uniform(0.02, 0.98, 6) * t)

            # Depths just outside and inside the layer where we use (C.1)
            R_100 = 100.0 / k_n
            for R in (0.8 * R_100, 1.25 * R_100):
                if R < 0.9 * t:
                    z_values.append(np.sqrt(t * t - R * R))

            z_values = np.array([z for z in z_values if 0 < z < t])
            ours = snapshot_mode(k_n, t, z_values, omega)
            for z, value in zip(z_values, ours):
                reference = direct_mode(k_n, t, z, omega)
                if abs(value - reference) > worst[0]:
                    worst = (abs(value - reference), (n, fraction, z / t))
    print(d_lambda, 'worst |figure method - direct quadrature| =', worst, flush=True)
