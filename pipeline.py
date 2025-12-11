import asyncio
import threading
from PyQt6.QtWidgets import QApplication
from bleak import BleakScanner
import time
import statistics

from analysis_plots import plot_pf_trajectory, plot_rssi_timeseries, plot_rssi_variance
from filter import ParticleFilter2D
from trilateration import parse_ibeacon
from visualizer_pg import PGVisualizer
import csv

RSSI_LOG = open("rssi_log.csv", "w", newline="")
RSSI_WRITER = csv.writer(RSSI_LOG)
RSSI_WRITER.writerow(["time", "minor", "rssi"])

EST_LOG = open("pf_estimates.csv", "w", newline="")
EST_WRITER = csv.writer(EST_LOG)
EST_WRITER.writerow(["time", "x", "y"])

# Apartment measurements
# BEACONS = {
#     1: (7.1628, 2.8956),
#     2: (2.9464, 3.1242),
#     3: (0.3302, 3.1242),
#     4: (2.9464, 0.4826),
# }

BEACONS = {
    1: (0.0, 0.0),
    2: (0.0, 16.0),
    3: (10.0, 16.0),
    4: (10.0, 0.0),
}

TARGET_UUID = "12345678-1234-1234-1234-1234567890AB"
X_MIN, X_MAX = -0.2, 12
Y_MIN, Y_MAX = -0.2, 20

async def scanner_loop(pf, visualizer):
    """
    Collect RSSI in a sliding time window, then do one PF update.
    Uses per-beacon RSSI windows and aggregates them to reduce noise
    and avoid bias toward chatty beacons.
    """
    WINDOW_SEC = 0.25  # ~4 Hz PF update

    # per-beacon RSSI buffer: minor -> [rssi, rssi, ...]
    window_rssi = {minor: [] for minor in BEACONS.keys()}

    last_update = time.monotonic()

    async with BleakScanner() as scanner:
        async for device, adv in scanner.advertisement_data():
            parsed = parse_ibeacon(adv.manufacturer_data)
            if not parsed:
                continue

            uuid, major, minor, txp = parsed
            if uuid != TARGET_UUID or minor not in BEACONS:
                continue

            rssi = adv.rssi

            # --- log raw packet ---
            now = time.monotonic()
            RSSI_WRITER.writerow([now, minor, rssi])
            RSSI_LOG.flush()

            # --- put into per-beacon window ---
            window_rssi[minor].append(rssi)

            # --- PF update at fixed rate ---
            if now - last_update >= WINDOW_SEC:
                dt = now - last_update  # in case you want time-based motion model
                last_update = now

                # 1) motion step
                pf.predict(dt=dt)

                # 2) aggregate RSSI per beacon for this window
                #    dict: minor -> aggregated_rssi (median here)
                aggregated_rssi = {}
                for b_minor, samples in window_rssi.items():
                    if not samples:
                        # No packets this window for this beacon → treat as missing
                        continue
                    # You can use mean() instead of median() if you like
                    aggregated_rssi[b_minor] = statistics.median(samples)

                # 3) measurement update (PF must handle missing beacons gracefully)
                pf.update_rssi(aggregated_rssi)

                # 4) estimate + visualize + log
                est = pf.estimate()  # returns (x, y) or (x, y, theta)

                EST_WRITER.writerow([now, est[0], est[1]])
                EST_LOG.flush()

                visualizer.update(pf.particles.copy(), est)

                # 5) clear *values* but keep dict structure
                for samples in window_rssi.values():
                    samples.clear()


def start_scanner_async(pf, visualizer):
    """Runs asyncio loop in separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(scanner_loop(pf, visualizer))


def main():
    app = QApplication([])

    # Init PF
    pf = ParticleFilter2D(
        beacons=BEACONS,
        x_range=(X_MIN, X_MAX),
        y_range=(Y_MIN, Y_MAX),
        num_particles=5000,
        motion_std=0.02,
        meas_std=0.05,
    )

    # Start visualizer
    vis = PGVisualizer(BEACONS, (X_MIN, X_MAX), (Y_MIN, Y_MAX))
    vis.show()

    # Start BLE scanner thread
    threading.Thread(target=start_scanner_async, args=(pf, vis), daemon=True).start()

    app.exec()

    plot_rssi_timeseries("rssi_log.csv")
    plot_rssi_variance("rssi_log.csv")
    plot_pf_trajectory("pf_estimates.csv")


if __name__ == "__main__":
    main()