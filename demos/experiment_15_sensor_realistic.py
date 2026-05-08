"""Experiment 15: sensor-realistic re-run of Exp. 13–14 at 1 m cube cage.

Compares MLX90394 and AK09940A at carrier frequencies that respect each
sensor's max ODR, with per-sample noise scaled from the sensor's
spectral density (nT/sqrt(Hz)) at the actual sample rate.

The original Exp. 13–14 at 0.5 m cube assumed 50 kSPS sampling and a
fixed 0.5 µT noise floor — neither physically realisable on the
MLX90393.  This run uses the SensorProfile abstraction so each row is a
faithful prediction of what the named part would actually deliver.

Run as a script:
    python demos/experiment_15_sensor_realistic.py [--quick]

--quick reduces the field-table resolution and skips a few rows for
fast iteration; full run is ~3-5 minutes on a modern desktop.
"""

import argparse
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source import sensor_profile
from source.circular_current_loop import CircularCurrentLoop
from source.simulation import Simulation
from source.path import SplinePath
from source.inversion import (
    FieldTable, demodulate, generate_rotations, apply_rotation_to_field,
    apply_sensor_profile_noise, invert_trace, invert_trace_6dof,
)


def build_1m_cube_cage(carrier_triple, drive_current=5.0):
    """3-axis anti-Helmholtz pairs for a 1m cube cage.

    Coil diameter 1.4 m, separation ±50 cm.  Asymmetric currents
    (+I, -0.7·I) break mirror symmetry as in the original cage demo.
    """
    fx, fy, fz = carrier_triple
    I = drive_current
    return Simulation([
        CircularCurrentLoop(diameter=1.4, center=[-0.5, 0, 0],
                            normal=[1, 0, 0], current=+I, frequency=fx),
        CircularCurrentLoop(diameter=1.4, center=[+0.5, 0, 0],
                            normal=[1, 0, 0], current=-0.7 * I, frequency=fx),
        CircularCurrentLoop(diameter=1.4, center=[0, -0.5, 0],
                            normal=[0, 1, 0], current=+I, frequency=fy),
        CircularCurrentLoop(diameter=1.4, center=[0, +0.5, 0],
                            normal=[0, 1, 0], current=-0.7 * I, frequency=fy),
        CircularCurrentLoop(diameter=1.4, center=[0, 0, -0.5],
                            normal=[0, 0, 1], current=+I, frequency=fz),
        CircularCurrentLoop(diameter=1.4, center=[0, 0, +0.5],
                            normal=[0, 0, 1], current=-0.7 * I, frequency=fz),
    ])


def cage_path():
    """A spline through ~±35 cm in the cage interior."""
    pts = np.array([
        [-0.30, -0.24, -0.20],
        [+0.10, +0.30, -0.16],
        [+0.36, -0.20, +0.24],
        [-0.16, +0.16, +0.36],
        [+0.24, +0.10, -0.30],
    ])
    return SplinePath(pts)


def _path_fractions(t, speed, path_length, traversal_mode):
    """Map sample times to fractional positions along the path.

    ``"pingpong"`` (default) reverses direction at each endpoint, valid for
    any path. ``"loop"`` wraps around at the end, valid only for paths
    where the start and end coincide (otherwise produces a teleport).
    """
    raw = speed * t
    if traversal_mode == "pingpong":
        cycle = np.mod(raw / path_length, 2.0)
        return np.where(cycle <= 1.0, cycle, 2.0 - cycle)
    elif traversal_mode == "loop":
        return np.mod(raw / path_length, 1.0)
    else:
        raise ValueError(f"unknown traversal_mode: {traversal_mode!r}")


def synthesize_signal(sim, path, duration, fs, speed=0.1, rotate_deg=30,
                      seed=0, traversal_mode="pingpong"):
    """Sample the path at fs and compute lab-frame B(t).

    Returns (t, points, Bx, By, Bz, rotations).
    """
    n_samples = int(duration * fs) + 1
    t = np.linspace(0, duration, n_samples)

    L = path.length
    frac = _path_fractions(t, speed, L, traversal_mode)

    n_dense = max(2000, n_samples)
    dense = path.get_points(n_dense)
    idx = np.clip((frac * (n_dense - 1)).astype(int), 0, n_dense - 1)
    points = dense[idx]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    Bx = np.zeros(n_samples)
    By = np.zeros(n_samples)
    Bz = np.zeros(n_samples)
    for loop in sim.loops:
        bx, by, bz = loop.magnetic_field(x, y, z)
        bx = np.asarray(bx).ravel()
        by = np.asarray(by).ravel()
        bz = np.asarray(bz).ravel()
        mod = np.cos(2.0 * np.pi * loop.frequency * t + loop.phase)
        Bx += bx * mod
        By += by * mod
        Bz += bz * mod

    rotations = None
    if rotate_deg > 0:
        rotations = generate_rotations(points, n_samples,
                                       max_perturbation_deg=rotate_deg,
                                       seed=seed)
        Bx, By, Bz = apply_rotation_to_field(Bx, By, Bz, rotations)
    return t, points, Bx, By, Bz, rotations


def errors_against_truth(t, true_pts, t_est, est_pts):
    """Median/max position error in mm, interpolating ground truth onto
    the inversion timestamps."""
    from scipy.interpolate import interp1d
    interp = interp1d(t, true_pts, axis=0, fill_value="extrapolate")
    truth_at = interp(t_est)
    errs = np.linalg.norm(est_pts - truth_at, axis=1)
    return float(np.median(errs)) * 1e3, float(errs.max()) * 1e3


def orientation_errors(t, true_rotations, t_est, est_rotations):
    from scipy.spatial.transform import Rotation, Slerp
    if true_rotations is None or est_rotations is None:
        return None, None
    times = np.asarray(t)
    key_rots = Rotation.concatenate(true_rotations)
    slerp = Slerp(times, key_rots)
    t_clamped = np.clip(t_est, times[0], times[-1])
    truth_at = slerp(t_clamped)
    deg = []
    for true_rot, est_rot in zip(truth_at, est_rotations):
        diff = est_rot * true_rot.inv()
        deg.append(np.rad2deg(diff.magnitude()))
    deg = np.array(deg)
    return float(np.median(deg)), float(deg.max())


def run_one(profile_name, carrier_triple, drive_current, mode,
            resolution=25, duration=3.0, rotate_deg=30, seed=0,
            path=None, bounds=None, traversal_mode="pingpong"):
    """Execute one (sensor, carrier, drive, mode) cell of the matrix.

    Parameters
    ----------
    path : SamplePath, optional
        Trajectory the simulated bird follows. Defaults to ``cage_path()``,
        which spans roughly ±0.36 m inside the 1 m cube cage.
    bounds : tuple of 6 floats, optional
        Field-table search bounds (x_min, x_max, y_min, y_max, z_min, z_max).
        Defaults to ±0.45 m to match the original 1 m cube setup.
    traversal_mode : {"pingpong", "loop"}, optional
        How the bird traverses the path over time. ``"pingpong"`` reverses
        direction at each endpoint (default, valid for any path).
        ``"loop"`` wraps around — only valid when ``path.is_closed``.
    """
    prof = sensor_profile.get(profile_name)
    fs = prof.max_odr_hz if np.isfinite(prof.max_odr_hz) else 50000.0
    sim = build_1m_cube_cage(carrier_triple, drive_current=drive_current)
    if path is None:
        path = cage_path()

    rotate = (mode == "6-DOF")
    t, points, Bx, By, Bz, rotations = synthesize_signal(
        sim, path, duration, fs, speed=0.10,
        rotate_deg=rotate_deg if rotate else 0, seed=seed,
        traversal_mode=traversal_mode,
    )

    # Inject sensor-realistic noise
    Bx, By, Bz, sigma_T = apply_sensor_profile_noise(
        Bx, By, Bz, prof, fs, seed=seed + 1,
    )

    # Field strength at path midpoint (for SNR readout)
    bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    mean_B_uT = float(np.mean(bmag)) * 1e6
    sigma_uT = sigma_T * 1e6
    snr = mean_B_uT / sigma_uT if sigma_uT > 0 else float("inf")

    # Field table over the search volume
    if bounds is None:
        bounds = (-0.45, 0.45, -0.45, 0.45, -0.45, 0.45)
    table = FieldTable(sim, bounds, resolution=resolution)

    # Inversion (window 30 ms, matching Exp. 13)
    min_freq = min(carrier_triple)
    window_periods = 0.030 * min_freq

    signal = np.column_stack([Bx, By, Bz])
    if rotate:
        t_est, est_pts, est_rots, _ = invert_trace_6dof(
            table, t, signal, window_periods=window_periods,
        )
    else:
        t_est, est_pts, _ = invert_trace(
            table, t, signal, window_periods=window_periods,
        )
        est_rots = None

    pos_med, pos_max = errors_against_truth(t, points, t_est, est_pts)
    orient_med, orient_max = orientation_errors(t, rotations, t_est, est_rots)
    return {
        "sensor": profile_name,
        "carrier": carrier_triple,
        "drive": drive_current,
        "mode": mode,
        "fs": fs,
        "sigma_uT": sigma_uT,
        "mean_B_uT": mean_B_uT,
        "snr": snr,
        "pos_med_mm": pos_med,
        "pos_max_mm": pos_max,
        "orient_med_deg": orient_med,
        "orient_max_deg": orient_max,
        "n_estimates": len(t_est),
    }


# ----------------------------------------------------------------------
# Experiment matrix
# ----------------------------------------------------------------------

# Carrier triples that fit each sensor's Nyquist with reasonable headroom
TRIPLES = {
    "low":  (100.0, 137.0, 173.0),   # max 173 Hz; needs fs >= ~870 Hz
    "high": (300.0, 411.0, 519.0),   # max 519 Hz; needs fs >= ~2600 Hz
}

MATRIX = [
    # (sensor, triple_key, drive_A, mode)
    # --- Phase A: carrier sweep at 5 A drive ---
    ("MLX90394", "low",  5.0, "3-DOF"),
    ("MLX90394", "low",  5.0, "6-DOF"),
    ("AK09940A", "low",  5.0, "3-DOF"),
    ("AK09940A", "low",  5.0, "6-DOF"),
    ("AK09940A", "high", 5.0, "3-DOF"),
    ("AK09940A", "high", 5.0, "6-DOF"),
    # --- Phase B: drive-current sweep at each sensor's best frequency ---
    ("MLX90394", "low",  1.0,  "3-DOF"),
    ("MLX90394", "low",  10.0, "3-DOF"),
    ("AK09940A", "high", 1.0,  "3-DOF"),
    ("AK09940A", "high", 10.0, "3-DOF"),
    # --- Reference: ideal sensor at the high triple ---
    ("ideal",    "high", 5.0, "3-DOF"),
    ("ideal",    "high", 5.0, "6-DOF"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Lower-resolution table for fast iteration")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Signal duration in seconds (default 3.0)")
    args = parser.parse_args()

    resolution = 18 if args.quick else 25

    print(f"Experiment 15: 1 m cube cage, sensor-realistic noise.")
    print(f"  Field table resolution: {resolution}^3")
    print(f"  Signal duration:        {args.duration:.1f} s")
    print(f"  Window:                 30 ms (Exp. 13 standard)")
    print()

    results = []
    t_total = time.time()
    for i, (sensor_name, triple_key, drive, mode) in enumerate(MATRIX, 1):
        triple = TRIPLES[triple_key]
        label = f"{sensor_name:9s} | {triple_key:4s} ({triple[0]:>4.0f}/" \
                f"{triple[1]:>4.0f}/{triple[2]:>4.0f}) | {drive:>4.1f}A | {mode}"
        print(f"[{i:2d}/{len(MATRIX)}] {label} ...", flush=True)
        t0 = time.time()
        try:
            r = run_one(sensor_name, triple, drive, mode,
                        resolution=resolution, duration=args.duration)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        results.append(r)
        elapsed = time.time() - t0
        orient_str = (f"  orient {r['orient_med_deg']:.1f}/{r['orient_max_deg']:.1f} deg"
                      if r["orient_med_deg"] is not None else "")
        print(f"    sigma {r['sigma_uT']:.2f} uT  |B| {r['mean_B_uT']:.2f} uT  "
              f"SNR {r['snr']:.1f}  |  pos {r['pos_med_mm']:.2f}/"
              f"{r['pos_max_mm']:.2f} mm{orient_str}  ({elapsed:.0f}s)")

    print()
    print(f"Total: {time.time() - t_total:.0f} s")

    # Print Markdown table for the inversion log
    print()
    print("Markdown summary (paste into docs/inversion_log.md):")
    print()
    print("| Sensor | Carrier (Hz) | Drive | fs | sigma | SNR | Mode | "
          "Pos err (med/max) | Orient err (med/max) |")
    print("|--------|--------------|-------|-----|-------|-----|------|"
          "-------------------|---------------------|")
    for r in results:
        c = r["carrier"]
        carrier_str = f"{c[0]:.0f}/{c[1]:.0f}/{c[2]:.0f}"
        orient = (f"{r['orient_med_deg']:.1f}deg / {r['orient_max_deg']:.1f}deg"
                  if r["orient_med_deg"] is not None else "n/a")
        print(f"| {r['sensor']} | {carrier_str} | {r['drive']:.1f} A | "
              f"{r['fs']:.0f} Hz | {r['sigma_uT']:.2f} uT | "
              f"{r['snr']:.1f} | {r['mode']} | "
              f"{r['pos_med_mm']:.2f} / {r['pos_max_mm']:.2f} mm | "
              f"{orient} |")


if __name__ == "__main__":
    main()
