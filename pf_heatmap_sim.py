import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Room & beacon configuration
# ===========================

ROOM_W = 10.0   # meters (x direction)
ROOM_H = 16.0   # meters (y direction)

# 10 m^2 cells → 160 / 10 = 16 cells
# Choose 4 x 4 grid: 2.5 m x 4 m per cell
GRID_NX = 4
GRID_NY = 4
CELL_W = ROOM_W / GRID_NX
CELL_H = ROOM_H / GRID_NY

# Your beacons at the corners
BEACONS = {
    1: (0.0, 0.0),
    2: (0.0, 16.0),
    3: (10.0, 16.0),
    4: (10.0, 0.0),
}

# ===========================
# RSSI path-loss model
# ===========================

# RSSI(d) = A - 10 n log10(d) + noise
RSSI_A = -59.0    # dBm at 1 m
RSSI_N = 2.0      # path-loss exponent
RSSI_SIGMA = 4.0  # dB noise std


def distance_to_rssi(d):
    d = np.maximum(d, 0.5)  # avoid log(0)
    rssi_mean = RSSI_A - 10.0 * RSSI_N * np.log10(d)
    return np.random.normal(rssi_mean, RSSI_SIGMA)


def rssi_to_distance(rssi):
    """Inverse of the above path-loss model."""
    return 10 ** ((RSSI_A - rssi) / (10.0 * RSSI_N))


# ===========================
# Particle filter definition
# ===========================

class ParticleFilter2D:
    def __init__(self, beacons, x_range, y_range,
                 num_particles=2000,
                 motion_std=0.1,     # motion noise [m]
                 meas_std=0.8):      # distance noise [m]
        """
        Very simple PF for simulation.
        State: position (x, y)
        """
        self.beacons = beacons
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

        self.N = num_particles
        self.motion_std = motion_std
        self.meas_std = meas_std

        # Initialize particles uniformly in the room
        self.particles = np.zeros((self.N, 2))
        self.particles[:, 0] = np.random.uniform(self.x_min, self.x_max, self.N)
        self.particles[:, 1] = np.random.uniform(self.y_min, self.y_max, self.N)
        self.weights = np.ones(self.N) / self.N

    def predict(self):
        """Random-walk motion model."""
        noise = np.random.normal(0.0, self.motion_std, size=self.particles.shape)
        self.particles += noise

        # Keep particles inside the room
        self.particles[:, 0] = np.clip(self.particles[:, 0], self.x_min, self.x_max)
        self.particles[:, 1] = np.clip(self.particles[:, 1], self.y_min, self.y_max)

    def update(self, meas_dists):
        """
        meas_dists: dict {beacon_id: measured_distance}
        Uses Gaussian likelihood on distance.
        """
        if not meas_dists:
            return

        # Compute likelihood for each particle
        log_w = np.zeros(self.N)

        for bid, d_meas in meas_dists.items():
            bx, by = self.beacons[bid]
            dx = self.particles[:, 0] - bx
            dy = self.particles[:, 1] - by
            d_pred = np.sqrt(dx * dx + dy * dy)

            # Gaussian log-likelihood
            sigma2 = self.meas_std ** 2
            log_w += -0.5 * ((d_meas - d_pred) ** 2) / sigma2

        # Normalize
        log_w -= np.max(log_w)  # avoid overflow
        w = np.exp(log_w)
        w_sum = np.sum(w)
        if w_sum == 0:
            # Degenerate case: reset to uniform
            self.weights[:] = 1.0 / self.N
        else:
            self.weights = w / w_sum

        self.resample_if_needed()

    def resample_if_needed(self):
        # Effective N
        neff = 1.0 / np.sum(self.weights ** 2)
        if neff < self.N / 2.0:
            self.resample()

    def resample(self):
        """Systematic resampling."""
        positions = (np.arange(self.N) + np.random.uniform()) / self.N
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # guard
        idx = np.searchsorted(cumsum, positions)
        self.particles = self.particles[idx]
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        """Return weighted mean estimate (x, y)."""
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        return np.array([x, y])

    def particle_spread(self):
        """
        Return a scalar measure of uncertainty: sqrt(trace(cov)).
        """
        mean = self.estimate()
        diff = self.particles - mean
        w = self.weights[:, None]
        cov = (w * diff).T @ diff
        spread = np.sqrt(np.trace(cov))
        return spread


# ===========================
# Trajectory simulation
# ===========================

def simulate_trajectory(T):
    """
    Make a smooth, paper-friendly trajectory that explores the room.
    Returns: (T, 2) array of [x, y]
    """
    t = np.linspace(0, 2 * np.pi, T)
    # Figure-8-ish path that fills room
    x = 5.0 + 4.0 * np.sin(t)
    y = 8.0 + 7.0 * np.sin(2 * t + 0.5)

    x = np.clip(x, 0.5, ROOM_W - 0.5)
    y = np.clip(y, 0.5, ROOM_H - 0.5)

    return np.stack([x, y], axis=1)


def simulate_measurements(gt_positions):
    """
    Given ground-truth positions [T, 2], simulate RSSI
    and convert to measured distances.
    Returns:
        meas_dists_list: list of dict {beacon_id: d_meas} per timestep
        true_dists: array [T, B] true distances to each beacon (for residuals)
    """
    T = gt_positions.shape[0]
    beacon_ids = sorted(BEACONS.keys())
    B = len(beacon_ids)

    true_dists = np.zeros((T, B))
    meas_dists_list = []

    for k in range(T):
        x, y = gt_positions[k]
        meas_dict = {}

        for j, bid in enumerate(beacon_ids):
            bx, by = BEACONS[bid]
            d_true = np.hypot(x - bx, y - by)
            true_dists[k, j] = d_true

            rssi = distance_to_rssi(d_true)
            d_meas = rssi_to_distance(rssi)
            meas_dict[bid] = d_meas

        meas_dists_list.append(meas_dict)

    return meas_dists_list, true_dists


# ===========================
# Heatmap binning utilities
# ===========================

def positions_to_grid_indices(positions):
    """
    Convert positions [T, 2] (x, y) to grid indices (ix, iy)
    where 0 <= ix < GRID_NX, 0 <= iy < GRID_NY.
    """
    x = positions[:, 0]
    y = positions[:, 1]
    ix = np.floor(x / CELL_W).astype(int)
    iy = np.floor(y / CELL_H).astype(int)

    ix = np.clip(ix, 0, GRID_NX - 1)
    iy = np.clip(iy, 0, GRID_NY - 1)
    return ix, iy


def accumulate_to_grid(ix, iy, values=None):
    """
    Accumulate counts or mean of values in each cell.

    If values is None -> returns counts per cell.
    If values is given [T] -> returns mean(values) in each cell (ignoring empty cells).
    """
    grid_sum = np.zeros((GRID_NY, GRID_NX))
    grid_count = np.zeros((GRID_NY, GRID_NX))

    T = ix.shape[0]
    for k in range(T):
        gx = ix[k]
        gy = iy[k]
        if values is None:
            grid_sum[gy, gx] += 1.0
        else:
            grid_sum[gy, gx] += values[k]
        grid_count[gy, gx] += 1.0

    if values is None:
        # Just counts
        return grid_sum
    else:
        # Mean, avoiding divide-by-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_grid = grid_sum / grid_count
            mean_grid[grid_count == 0] = np.nan
        return mean_grid


# ===========================
# Plotting helper
# ===========================

def plot_heatmap(data, title, filename, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(5, 8))  # tall room
    # data indexed [iy, ix], so extent is [0, W, 0, H]
    plt.imshow(
        data,
        origin="lower",
        extent=[0, ROOM_W, 0, ROOM_H],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    # Grid lines for the 10 m^2 blocks
    for i in range(GRID_NX + 1):
        plt.axvline(i * CELL_W, color="k", linewidth=0.5, alpha=0.3)
    for j in range(GRID_NY + 1):
        plt.axhline(j * CELL_H, color="k", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


# ===========================
# Core simulation + heatmaps
# ===========================

def run_simulation_and_make_heatmaps(T=800):
    # 1. Ground-truth trajectory
    gt_positions = simulate_trajectory(T)

    # 2. Simulated measurements
    meas_dists_list, true_dists = simulate_measurements(gt_positions)

    # 3. Run particle filter
    pf = ParticleFilter2D(
        beacons=BEACONS,
        x_range=(0.0, ROOM_W),
        y_range=(0.0, ROOM_H),
        num_particles=2000,
        motion_std=0.15,
        meas_std=0.8,
    )

    est_positions = np.zeros_like(gt_positions)
    spreads = np.zeros(T)
    dist_residuals = np.zeros(T)

    beacon_ids = sorted(BEACONS.keys())
    B = len(beacon_ids)

    for k in range(T):
        pf.predict()
        pf.update(meas_dists_list[k])
        est = pf.estimate()
        est_positions[k] = est
        spreads[k] = pf.particle_spread()

        # Distance residuals: |d_meas - d_pred(est)|
        res = []
        for j, bid in enumerate(beacon_ids):
            bx, by = BEACONS[bid]
            d_pred = np.hypot(est[0] - bx, est[1] - by)
            d_meas = meas_dists_list[k][bid]
            res.append(abs(d_meas - d_pred))
        dist_residuals[k] = np.mean(res)

    # 4. Compute errors
    errors = np.linalg.norm(gt_positions - est_positions, axis=1)

    # 5. Bin everything into 10 m^2 grid blocks (using ground truth cell for stats)
    ix_gt, iy_gt = positions_to_grid_indices(gt_positions)
    ix_est, iy_est = positions_to_grid_indices(est_positions)

    heat_gt_occ = accumulate_to_grid(ix_gt, iy_gt, values=None)
    heat_est_occ = accumulate_to_grid(ix_est, iy_est, values=None)
    heat_err = accumulate_to_grid(ix_gt, iy_gt, values=errors)
    heat_dist_resid = accumulate_to_grid(ix_gt, iy_gt, values=dist_residuals)
    heat_spread = accumulate_to_grid(ix_gt, iy_gt, values=spreads)

    # 6. Plot heatmaps (5 maps)
    plot_heatmap(heat_gt_occ, "Ground-truth occupancy (counts)",
                 "heatmap1_groundtruth_occupancy.png")
    plot_heatmap(heat_est_occ, "PF estimated occupancy (counts)",
                 "heatmap2_pf_occupancy.png")
    plot_heatmap(heat_err, "Mean localization error (m)",
                 "heatmap3_error.png", cmap="magma")
    plot_heatmap(heat_dist_resid, "Mean distance residual |d_meas - d_pred| (m)",
                 "heatmap4_dist_residual.png", cmap="plasma")
    plot_heatmap(heat_spread, "Mean particle spread (m)",
                 "heatmap5_uncertainty.png", cmap="cividis")


# ===========================
# Re-usable function for REAL DATA
# ===========================

def make_heatmaps_from_logs(gt_positions, est_positions,
                            dist_residuals=None,
                            spreads=None,
                            prefix="real_"):
    """
    Use this for real experiments.

    gt_positions: [T, 2] ground-truth (x, y)
    est_positions: [T, 2] PF estimates (x, y)
    dist_residuals: optional [T] mean |d_meas - d_pred| per timestep
    spreads: optional [T] particle spread per timestep

    Saves up to 5 heatmaps with the given prefix.
    """
    T = gt_positions.shape[0]
    assert est_positions.shape[0] == T

    errors = np.linalg.norm(gt_positions - est_positions, axis=1)

    ix_gt, iy_gt = positions_to_grid_indices(gt_positions)
    ix_est, iy_est = positions_to_grid_indices(est_positions)

    heat_gt_occ = accumulate_to_grid(ix_gt, iy_gt, values=None)
    heat_est_occ = accumulate_to_grid(ix_est, iy_est, values=None)
    heat_err = accumulate_to_grid(ix_gt, iy_gt, values=errors)

    plot_heatmap(heat_gt_occ, "Ground-truth occupancy (counts)",
                 f"{prefix}heatmap1_groundtruth_occupancy.png")
    plot_heatmap(heat_est_occ, "PF estimated occupancy (counts)",
                 f"{prefix}heatmap2_pf_occupancy.png")
    plot_heatmap(heat_err, "Mean localization error (m)",
                 f"{prefix}heatmap3_error.png", cmap="magma")

    if dist_residuals is not None:
        heat_dist_resid = accumulate_to_grid(ix_gt, iy_gt, values=dist_residuals)
        plot_heatmap(heat_dist_resid, "Mean distance residual |d_meas - d_pred| (m)",
                     f"{prefix}heatmap4_dist_residual.png", cmap="plasma")

    if spreads is not None:
        heat_spread = accumulate_to_grid(ix_gt, iy_gt, values=spreads)
        plot_heatmap(heat_spread, "Mean particle spread (m)",
                     f"{prefix}heatmap5_uncertainty.png", cmap="cividis")


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    # Synthetic simulation: produces 5 PNGs
    run_simulation_and_make_heatmaps(T=800)

    # Example of how you'd use real logs:
    # gt = np.load("gt_positions.npy")        # shape [T, 2]
    # est = np.load("pf_estimates.npy")       # shape [T, 2]
    # dist_resid = np.load("dist_residuals.npy")   # shape [T]
    # spreads = np.load("particle_spread.npy")     # shape [T]
    # make_heatmaps_from_logs(gt, est, dist_resid, spreads, prefix="real_")