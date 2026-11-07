'''Figure 2 of the paper: plots the snapshots computed by transient_snapshots.py.

Usage, from the root of the repository:
    python paper/plot_snapshots.py SNAPSHOTS.npz[,SNAPSHOTS2.npz,...] OUT.pdf OUT_PNG_STEM T1/z_T,T2/z_T,...
e.g. python paper/plot_snapshots.py results/snapshots_d100_l5.npz results/transient_forming_d_lambda=100_w_lambda=5.pdf results/transient_forming_d_lambda=100_w_lambda=5-1 0.25,0.5,0.75

There is one row per time and one column per grating, with x/d vertical and z/z_T horizontal. Each grating is shown
in its own units, (A d/l)^2. The pixels hold averages over their width in z (see transient_snapshots.py). The colour
scale saturates at 1 and the overflow is marked on the colour bar, as in Figure 3 (stationary_carpets.py).
'''
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from pyTalbot.plotter import resize_field

fraction_labels = {0.25: "z_T/4", 0.5: "z_T/2", 0.75: "3z_T/4", 1.0: "z_T"} # Labels of the times


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    datasets = [np.load(f) for f in sys.argv[1].split(",")]
    out_pdf, out_png = sys.argv[2], sys.argv[3]
    wanted = [float(a) for a in sys.argv[4].split(",")]
    rows, cols = len(wanted), len(datasets)

    cm = 1 / 2.54
    fig, axes = plt.subplots(rows, cols, figsize=(16 * cm, (5.2 if cols == 1 else 4.4) * cm * rows), sharex=True,
                             sharey=True, squeeze=False, gridspec_kw={"hspace": 0.12, "wspace": 0.05})

    # We plot each grating in its own column
    for j, data in enumerate(datasets):
        fractions = list(data["fractions"])
        d_lambda, l_lambda = int(data["d_lambda"]), int(data["l_lambda"])
        vmax = (d_lambda / l_lambda) ** 2 # (A d/l)^2 with A = 1
        for i, f in enumerate(wanted):
            ax = axes[i, j]
            envelope = data["snapshots"][fractions.index(f)] / vmax
            im = ax.imshow(resize_field(envelope), cmap="turbo", vmin=0, vmax=1, interpolation="none", origin="lower",
                           extent=[0, 1, -1, 1], aspect="auto")
            ax.text(0.985, 0.93, f"({'abcdefghijkl'[i * cols + j]}) $t = {fraction_labels[f]}$", transform=ax.transAxes,
                    ha="right", va="top", fontsize=10 if cols == 1 else 9, color="white")
        if cols > 1:
            axes[0, j].set_title(rf"$d/\lambda = {d_lambda}$, $l/\lambda = {l_lambda}$", fontsize=10)

        # Label the X axis and set the ticks
        axes[-1, j].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axes[-1, j].set_xticklabels(["$0$", "$z_T/4$", "$z_T/2$", "$3z_T/4$", "$z_T$"], fontsize=9)
        axes[-1, j].set_xlabel("$z$", fontsize=10)

    # Label the Y axis and set the ticks
    for i in range(rows):
        axes[i, 0].set_yticks([-1, -0.5, 0, 0.5, 1])
        axes[i, 0].set_yticklabels(["$-d$", "$-d/2$", "$0$", "$d/2$", "$d$"], fontsize=9)

    # We keep the last tick label of each column off the next one
    if cols > 1:
        for j in range(cols - 1):
            axes[-1, j].get_xticklabels()[-1].set_ha("right")
        axes[-1, -1].get_xticklabels()[0].set_ha("left")

    # Colour bar
    cbar = fig.colorbar(im, ax=list(axes.ravel()), fraction=0.025, pad=0.02, aspect=35, extend="max",
                        ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(["$0$", "$0.25$", "$0.5$", "$0.75$", "$1$"], fontsize=9)
    cbar.set_label(r"Squared envelope $/\,(A d/l)^2$", fontsize=10)
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)

    # The PNG is averaged over the area of each pixel when rasterised, as in stationary_carpets.py
    for ax in axes.ravel():
        for img in ax.get_images():
            img.set_interpolation("antialiased")
            img.set_interpolation_stage("data")
    plt.savefig(out_png + ".png", bbox_inches="tight", dpi=250)
    plt.close(fig)

    from PIL import Image
    Image.open(out_png + ".png").convert("RGB").save(out_png + ".png", optimize=True)
