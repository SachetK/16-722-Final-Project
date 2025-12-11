import numpy as np
from statistics import median
from trilateration import rssi_to_distance

def robust_likelihood(d_meas, d_pred, sigma=0.8):
    """Cauchy likelihood: much less sensitive to outliers than Gaussian."""
    return 1.0 / (1.0 + ((d_meas - d_pred) / sigma) ** 2)


class ParticleFilter2D:
    def __init__(
        self,
        beacons,
        x_range,
        y_range,
        num_particles=2000,
        motion_std=0.03,
        meas_std=0.8,
        likelihood_temp=0.5,
    ):
        """
        beacons: dict {minor_id: (bx, by)}
        x_range, y_range: (min, max)

        motion_std: std dev (m) per second of random walk.
        meas_std:   scale (m) for Cauchy likelihood.
        likelihood_temp: 0 < T <= 1.  Smaller T => flatter likelihood (less collapse).
        """

        self.beacons = beacons
        self.x_range = x_range
        self.y_range = y_range

        self.N = num_particles
        self.motion_std = motion_std
        self.meas_std = meas_std
        self.likelihood_temp = likelihood_temp

        # Particles
        self.particles = np.empty((self.N, 2), dtype=np.float32)
        self.particles[:, 0] = np.random.uniform(x_range[0], x_range[1], self.N)
        self.particles[:, 1] = np.random.uniform(y_range[0], y_range[1], self.N)

        self.weights = np.ones(self.N, dtype=np.float32) / self.N

        # Per-beacon RSSI history for smoothing (used in real-time code if needed)
        self.rssi_hist = {bid: [] for bid in beacons}

    # -----------------------------------------------------
    # Motion model
    # -----------------------------------------------------
    def predict(self, dt=0.25):
        """
        Add a time-scaled random walk.
        Larger dt → larger positional uncertainty.
        """
        std = self.motion_std * dt
        noise = np.random.normal(0, std, size=self.particles.shape)
        self.particles += noise

        # Clamp to bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], self.x_range[0], self.x_range[1])
        self.particles[:, 1] = np.clip(self.particles[:, 1], self.y_range[0], self.y_range[1])

    # -----------------------------------------------------
    # RSSI-based update
    # -----------------------------------------------------
    def update_rssi(self, rssi_dict):
        """
        rssi_dict: {minor_id: rssi_value} for THIS WINDOW ONLY.
        Already aggregated upstream (median, mean, etc.).
        Missing beacons simply do not appear.
        """

        # Convert aggregated RSSI → distance
        dist_meas = {}
        for minor, rssi in rssi_dict.items():

            # Throw out obviously bad values
            if rssi >= -40 or rssi < -95:
                continue

            d = rssi_to_distance(rssi)
            d = float(np.clip(d, 0.2, 6.0))  # sanity clamp
            dist_meas[minor] = d

        # Not enough signals → skip measurement update
        if len(dist_meas) < 2:
            return

        # Compute weights update
        new_w = np.ones(self.N, dtype=np.float64)

        for minor, d_meas in dist_meas.items():
            bx, by = self.beacons[minor]

            dx = self.particles[:, 0] - bx
            dy = self.particles[:, 1] - by
            d_pred = np.sqrt(dx * dx + dy * dy)

            L = robust_likelihood(d_meas, d_pred, sigma=self.meas_std)

            # ---- soften likelihood to avoid over-collapse ----
            # L in (0,1]; raising to T<1 makes it flatter (less peaky)
            L = np.power(L, self.likelihood_temp)

            new_w *= L

        # Normalize
        new_w += 1e-20
        total = np.sum(new_w)
        if total <= 0.0 or not np.isfinite(total):
            # In pathological cases, reset to uniform
            self.weights.fill(1.0 / self.N)
        else:
            new_w /= total
            self.weights = new_w.astype(np.float32)

        self._resample_if_needed()

    # -----------------------------------------------------
    # Resampling
    # -----------------------------------------------------
    def _resample_if_needed(self):
        ESS = 1.0 / np.sum(self.weights ** 2)
        if ESS < 0.6 * self.N:
            self.resample()

    def resample(self):
        N = self.N
        positions = (np.arange(N) + np.random.uniform()) / N
        cum = np.cumsum(self.weights)
        cum[-1] = 1.0
        idx = np.searchsorted(cum, positions)

        self.particles = self.particles[idx]
        self.weights.fill(1.0 / N)

    def particle_spread(self):
        """
        Compute PF uncertainty as mean particle distance from the weighted mean.
        (Simple scalar uncertainty metric.)
        """
        mean = self.estimate()  # weighted [x, y]
        dx = self.particles[:, 0] - mean[0]
        dy = self.particles[:, 1] - mean[1]
        dist = np.sqrt(dx*dx + dy*dy)
        return float(np.average(dist, weights=self.weights))

    # -----------------------------------------------------
    # Estimate
    # -----------------------------------------------------
    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights)