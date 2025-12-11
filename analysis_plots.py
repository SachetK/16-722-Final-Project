# analysis_plots.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_rssi_timeseries(csv_path="rssi_log.csv"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12,5))
    start_time = df["time"].min()
    for m in df["minor"].unique():
        d = df[df["minor"] == m]
        plt.plot(d["time"] - start_time, d["rssi"], label=f"Beacon {m}")

    plt.xlabel("Time (s)")
    plt.ylabel("RSSI (dBm)")
    plt.title("RSSI Instability Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_rssi_timeseries.png", dpi=200)
    plt.show()


def plot_rssi_variance(csv_path="rssi_log.csv"):
    df = pd.read_csv(csv_path)
    variances = df.groupby("minor")["rssi"].var()

    plt.figure(figsize=(6,4))
    plt.bar(variances.index, variances.values)
    plt.xlabel("Beacon ID")
    plt.ylabel("RSSI Variance (dBm²)")
    plt.title("Per-Beacon RSSI Instability")
    plt.tight_layout()
    plt.savefig("fig_rssi_variance.png", dpi=200)
    plt.show()


def plot_pf_trajectory(csv_path="pf_estimates.csv"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6,4))
    plt.plot(df["x"], df["y"], "-o", markersize=2)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("PF Estimated Trajectory (Should Be Static)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_pf_trajectory.png", dpi=200)
    plt.show()