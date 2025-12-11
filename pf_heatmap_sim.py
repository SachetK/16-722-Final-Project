import numpy as np
import matplotlib.pyplot as plt

from filter import ParticleFilter2D   # YOUR EXACT PF FROM filter.py
from trilateration import rssi_to_distance


# ============================================================
# Room + Beacon Layout
# ============================================================
ROOM_W, ROOM_H = 10.0, 16.0

BEACONS = {
    1: (0.0, 0.0),
    2: (0.0, ROOM_H),
    3: (ROOM_W, ROOM_H),
    4: (ROOM_W, 0.0),
}


# ============================================================
# REALISTIC RSSI SIMULATION (BREAK PERFECT INVERSION)
# ============================================================

RSSI0 = -59
N = 2.0

# These two parameters MAKE the PF TRACK:
DIST_NOISE = 1.5     # noisy distance before converting to RSSI
DRIFT_STD  = 0.20    # slowly varying multipath drift per step
RSSI_STD   = 2.0     # instantaneous RSSI noise


def noisy_distance(x, y, bx, by):
    """Simulate BLE-like distance with noise (NOT invertible)."""
    d_true = np.sqrt((x - bx)**2 + (y - by)**2)
    d_noisy = d_true + np.random.normal(0, DIST_NOISE)
    return max(d_noisy, 0.2)


def distance_to_rssi_for_pf(d_noisy, drift):
    """Convert noisy distance into RSSI with multipath drift."""
    ideal = RSSI0 - 10 * N * np.log10(d_noisy)
    drift += np.random.normal(0, DRIFT_STD)  # multipath drift
    rssi = ideal + drift + np.random.normal(0, RSSI_STD)
    return rssi, drift


# ============================================================
# Trajectory Generator
# ============================================================
def generate_trajectory(T=600):
    t = np.linspace(0, 2*np.pi, T)
    cx, cy = ROOM_W/2, ROOM_H/2
    a, b = ROOM_W*0.35, ROOM_H*0.35
    x = cx + a*np.sin(t)
    y = cy + b*np.sin(t)*np.cos(t)
    return np.stack([x, y], axis=1)


# ============================================================
# Heatmap helper
# ============================================================
def accumulate_heatmap(Hmap, Nx, Ny, x, y, grid=1.0):
    i = int(x // grid)
    j = int(y // grid)
    if 0 <= i < Nx and 0 <= j < Ny:
        Hmap[j, i] += 1


# ============================================================
# MAIN SIMULATION (USING YOUR PF EXACTLY)
# ============================================================
def run_simulation():
    GRID = 1.0
    NX = int(ROOM_W / GRID)
    NY = int(ROOM_H / GRID)

    # Heatmaps
    heat_gt = np.zeros((NY, NX))
    heat_pf = np.zeros((NY, NX))
    heat_err = np.zeros((NY, NX))

    # SAME PF YOU USE IRL
    pf = ParticleFilter2D(
        beacons=BEACONS,
        x_range=(0.0, ROOM_W),
        y_range=(0.0, ROOM_H),
        num_particles=2000,
        motion_std=0.15,    # increased so PF can follow target
        meas_std=1.5,       # matches DIST_NOISE
    )

    # Drift state per beacon
    drift = {bid: 0.0 for bid in BEACONS}

    trajectory = generate_trajectory(T=600)

    for (x_gt, y_gt) in trajectory:

        # ----------------------------
        # Simulate BLE RSSI (REALISTIC)
        # ----------------------------
        rssi_dict = {}

        for bid, (bx, by) in BEACONS.items():

            # 1. Add noisy distance BEFORE RSSI calc
            d_noisy = noisy_distance(x_gt, y_gt, bx, by)

            # 2. Convert to RSSI with multipath drift + noise
            rssi, drift[bid] = distance_to_rssi_for_pf(d_noisy, drift[bid])
            rssi_dict[bid] = rssi

        # ----------------------------
        # PF update (YOUR EXACT PF)
        # ----------------------------
        pf.predict(dt=0.25)
        pf.update_rssi(rssi_dict)

        x_pf, y_pf = pf.estimate()

        # ----------------------------
        # Heatmaps
        # ----------------------------
        accumulate_heatmap(heat_gt, NX, NY, x_gt, y_gt)
        accumulate_heatmap(heat_pf, NX, NY, x_pf, y_pf)

        err = np.sqrt((x_pf - x_gt)**2 + (y_pf - y_gt)**2)
        heat_err[int(y_gt // GRID), int(x_gt // GRID)] = err

    # ============================================================
    # Save heatmaps
    # ============================================================
    def save(name, data, title):
        plt.figure(figsize=(6, 10))
        plt.imshow(data, origin='lower', cmap='inferno')
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(name, dpi=300)
        plt.close()

    save("heatmap1_groundtruth_occupancy.png", heat_gt, "Ground Truth Occupancy")
    save("heatmap2_pf_occupancy.png", heat_pf, "PF Occupancy")
    save("heatmap3_error.png",      heat_err, "Localization Error")

    print("Simulation finished. Heatmaps saved.")


if __name__ == "__main__":
    run_simulation()