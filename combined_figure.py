# combined_figure_2x2.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional LaTeX-like fonts for publication polish
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})

def make_combined_figure(
    rssi_csv="rssi_log.csv",
    pf_csv="pf_estimates.csv",
    out="fig_combined_2x2.png",
):
    # Load data
    rssi_df = pd.read_csv(rssi_csv)
    pf_df = pd.read_csv(pf_csv)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # ================================================================
    # (a) RSSI Time Series
    # ================================================================
    ax = axes[0, 0]
    t0 = rssi_df["time"].min()

    for minor in sorted(rssi_df["minor"].unique()):
        d = rssi_df[rssi_df["minor"] == minor]
        ax.plot(d["time"] - t0, d["rssi"], label=f"Beacon {minor}", lw=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RSSI (dBm)")
    ax.set_title("(a) RSSI Instability Over Time")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    # ================================================================
    # (b) RSSI Variance
    # ================================================================
    ax = axes[0, 1]
    vars = rssi_df.groupby("minor")["rssi"].var()
    ax.bar(vars.index, vars.values)

    ax.set_xlabel("Beacon ID")
    ax.set_ylabel("Variance (dBm²)")
    ax.set_title("(b) Per-Beacon RSSI Variance")
    ax.grid(axis="y")

    # ================================================================
    # (c) PF Trajectory
    # ================================================================
    ax = axes[1, 0]
    ax.plot(pf_df["x"], pf_df["y"], "-o", markersize=3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("(c) PF Estimated Trajectory (Static Test)")
    ax.grid(True)

    # ================================================================
    # (d) PF Scatter Density Map (optional)
    # ================================================================
    ax = axes[1, 1]
    heatmap, xedges, yedges = np.histogram2d(
        pf_df["x"], pf_df["y"], bins=30
    )
    ax.imshow(
        heatmap.T,
        origin="lower",
        cmap="viridis",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("(d) PF Position Density")
    ax.grid(False)

    # Save combined figure
    fig.savefig(out, dpi=300)
    print(f"Saved 2x2 figure: {out}")

make_combined_figure()