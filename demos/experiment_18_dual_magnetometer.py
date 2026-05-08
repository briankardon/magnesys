"""Experiment 18: dual-magnetometer backpack vs single-sensor headgear.

Tests whether two magnetometers at a fixed 3 cm baseline on a rigid
backpack PCB fix the deterministic 171 deg orientation flip seen in
Exp. 15-17.  For each test cell we synthesize the *same* trajectory,
rotation history, and per-sensor noise realisation, then run two
inversions on that data:

  1. Single-sensor (sensor 1 alone) — baseline matching Exp. 16
  2. Dual-sensor (sensor 1 + sensor 2 at 3 cm offset along sensor +x)

Both pipelines are exposed in source/inversion.py; the original
single-sensor code is unchanged.

Run as a script:
    python demos/experiment_18_dual_magnetometer.py [--quick]
"""

import argparse
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source import sensor_profile
from source.inversion import (
    FieldTable, DualSensorConfig,
    apply_rotation_to_field, apply_sensor_profile_noise,
    generate_rotations,
    invert_trace, invert_trace_6dof, invert_trace_6dof_robust,
    invert_trace_dual_3dof, invert_trace_dual_6dof,
    invert_trace_dual_6dof_robust,
)
from experiment_15_sensor_realistic import (
    build_1m_cube_cage, cage_path, errors_against_truth, orientation_errors,
    _path_fractions,
)


# Default backpack baseline — 3 cm along sensor-frame +x
DEFAULT_OFFSET_SENSOR = np.array([0.030, 0.0, 0.0])


def synthesize_dual_signal(sim, path, duration, fs, offset_sensor,
                           speed=0.10, rotate_deg=30, seed=0,
                           traversal_mode="pingpong"):
    """Sample the path at fs and compute B(t) at two co-rotating sensors.

    See ``experiment_15_sensor_realistic.synthesize_signal`` for the
    meaning of ``traversal_mode``.

    Returns
    -------
    t : (N,) ndarray
    points_1 : (N, 3) ndarray
        Lab-frame positions of sensor 1 (the trajectory we'll compare
        position estimates against).
    sig_1 : (N, 3) ndarray   — sensor 1 sensor-frame Bx, By, Bz
    sig_2 : (N, 3) ndarray   — sensor 2 sensor-frame Bx, By, Bz
    rotations : list of Rotation, length N (or None if rotate_deg <= 0)
    """
    n_samples = int(duration * fs) + 1
    t = np.linspace(0, duration, n_samples)

    L = path.length
    frac = _path_fractions(t, speed, L, traversal_mode)
    n_dense = max(2000, n_samples)
    dense = path.get_points(n_dense)
    idx = np.clip((frac * (n_dense - 1)).astype(int), 0, n_dense - 1)
    points_1 = dense[idx]

    # Determine sensor 2's lab-frame position at every sample
    offset_sensor = np.asarray(offset_sensor, dtype=float)
    if rotate_deg > 0:
        rotations = generate_rotations(points_1, n_samples,
                                       max_perturbation_deg=rotate_deg,
                                       seed=seed)
        offsets_lab = np.array([R.inv().apply(offset_sensor) for R in rotations])
    else:
        rotations = None
        offsets_lab = np.broadcast_to(offset_sensor, (n_samples, 3))
    points_2 = points_1 + offsets_lab

    # Compute lab-frame B at both sensor positions
    x1, y1, z1 = points_1[:, 0], points_1[:, 1], points_1[:, 2]
    x2, y2, z2 = points_2[:, 0], points_2[:, 1], points_2[:, 2]

    Bx1 = np.zeros(n_samples); By1 = np.zeros(n_samples); Bz1 = np.zeros(n_samples)
    Bx2 = np.zeros(n_samples); By2 = np.zeros(n_samples); Bz2 = np.zeros(n_samples)
    for loop in sim.loops:
        bx1, by1, bz1 = loop.magnetic_field(x1, y1, z1)
        bx2, by2, bz2 = loop.magnetic_field(x2, y2, z2)
        mod = np.cos(2.0 * np.pi * loop.frequency * t + loop.phase)
        Bx1 += np.asarray(bx1).ravel() * mod
        By1 += np.asarray(by1).ravel() * mod
        Bz1 += np.asarray(bz1).ravel() * mod
        Bx2 += np.asarray(bx2).ravel() * mod
        By2 += np.asarray(by2).ravel() * mod
        Bz2 += np.asarray(bz2).ravel() * mod

    # Rotate into sensor frame (both sensors share R)
    if rotations is not None:
        Bx1, By1, Bz1 = apply_rotation_to_field(Bx1, By1, Bz1, rotations)
        Bx2, By2, Bz2 = apply_rotation_to_field(Bx2, By2, Bz2, rotations)

    return (t, points_1,
            np.column_stack([Bx1, By1, Bz1]),
            np.column_stack([Bx2, By2, Bz2]),
            rotations)


def run_one_dual(profile_name, carrier_triple, drive_current, mode,
                 resolution=35, duration=3.0, rotate_deg=30, seed=0,
                 offset_sensor=DEFAULT_OFFSET_SENSOR,
                 path=None, bounds=None, traversal_mode="pingpong",
                 use_robust=False):
    """Run single- and dual-sensor inversions on the same trajectory.

    ``use_robust`` selects the multi-start 6-DOF inversions
    (``invert_trace_6dof_robust`` / ``invert_trace_dual_6dof_robust``)
    instead of the originals. 3-DOF mode is unaffected.
    """
    prof = sensor_profile.get(profile_name)
    fs = prof.max_odr_hz if np.isfinite(prof.max_odr_hz) else 50000.0
    sim = build_1m_cube_cage(carrier_triple, drive_current=drive_current)
    if path is None:
        path = cage_path()
    rotate = (mode == "6-DOF")

    t, points, sig1, sig2, rotations = synthesize_dual_signal(
        sim, path, duration, fs, offset_sensor,
        speed=0.10,
        rotate_deg=rotate_deg if rotate else 0,
        seed=seed,
        traversal_mode=traversal_mode,
    )

    # Independent noise realisations per sensor
    Bx1, By1, Bz1, sigma_T = apply_sensor_profile_noise(
        sig1[:, 0], sig1[:, 1], sig1[:, 2], prof, fs, seed=seed + 1,
    )
    Bx2, By2, Bz2, _ = apply_sensor_profile_noise(
        sig2[:, 0], sig2[:, 1], sig2[:, 2], prof, fs, seed=seed + 2,
    )
    signal_1 = np.column_stack([Bx1, By1, Bz1])
    signal_2 = np.column_stack([Bx2, By2, Bz2])

    bmag = np.sqrt(Bx1**2 + By1**2 + Bz1**2)
    mean_B_uT = float(np.mean(bmag)) * 1e6
    sigma_uT = sigma_T * 1e6
    snr = mean_B_uT / sigma_uT if sigma_uT > 0 else float("inf")

    if bounds is None:
        bounds = (-0.45, 0.45, -0.45, 0.45, -0.45, 0.45)
    table = FieldTable(sim, bounds, resolution=resolution)

    min_freq = min(carrier_triple)
    window_periods = 0.030 * min_freq

    single_6dof = invert_trace_6dof_robust if use_robust else invert_trace_6dof
    dual_6dof = (invert_trace_dual_6dof_robust if use_robust
                 else invert_trace_dual_6dof)

    # --- Single-sensor inversion (sensor 1 alone) -----------------------
    if rotate:
        t_s, pos_s, rot_s, _ = single_6dof(
            table, t, signal_1, window_periods=window_periods,
        )
    else:
        t_s, pos_s, _ = invert_trace(
            table, t, signal_1, window_periods=window_periods,
        )
        rot_s = None

    # --- Dual-sensor inversion -----------------------------------------
    dual_cfg = DualSensorConfig(offset_sensor_frame=tuple(offset_sensor))
    if rotate:
        t_d, pos_d, rot_d, _ = dual_6dof(
            table, t, signal_1, signal_2, dual_cfg,
            window_periods=window_periods,
        )
    else:
        t_d, pos_d, _ = invert_trace_dual_3dof(
            table, t, signal_1, signal_2, dual_cfg,
            window_periods=window_periods,
        )
        rot_d = None

    pos_med_s, pos_max_s = errors_against_truth(t, points, t_s, pos_s)
    pos_med_d, pos_max_d = errors_against_truth(t, points, t_d, pos_d)
    if rotate:
        om_s, oM_s = orientation_errors(t, rotations, t_s, rot_s)
        om_d, oM_d = orientation_errors(t, rotations, t_d, rot_d)
    else:
        om_s = oM_s = om_d = oM_d = None

    return {
        "sensor": profile_name,
        "carrier": carrier_triple,
        "drive": drive_current,
        "mode": mode,
        "fs": fs,
        "sigma_uT": sigma_uT,
        "snr": snr,
        "single": {"pos_med": pos_med_s, "pos_max": pos_max_s,
                    "orient_med": om_s, "orient_max": oM_s},
        "dual":   {"pos_med": pos_med_d, "pos_max": pos_max_d,
                    "orient_med": om_d, "orient_max": oM_d},
    }


TRIPLES = {
    "low":       (100.0,  137.0,  173.0),
    "very_high": (1000.0, 1373.0, 1747.0),
}

# (sensor, triple_key, drive_A, mode)
MATRIX = [
    # 3-DOF sanity: dual should be ~sqrt(2) better than single
    ("TMR_analog_nRF52", "very_high",  5.0, "3-DOF"),
    # 6-DOF: the headline test — dual must fix the 171 deg flip
    ("TMR_analog_nRF52", "very_high",  5.0, "6-DOF"),
    ("ideal",            "very_high",  5.0, "6-DOF"),
    ("TMR_analog_nRF52", "very_high", 10.0, "6-DOF"),
    ("TMR_analog_nRF52", "very_high",  1.0, "6-DOF"),
    # Robustness check at lower carrier
    ("TMR_analog_nRF52", "low",        5.0, "6-DOF"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Lower-resolution table for fast iteration")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Signal duration in seconds (default 3.0)")
    parser.add_argument("--resolution", type=int, default=35,
                        help="Field-table resolution (default 35)")
    parser.add_argument("--baseline", type=float, default=0.030,
                        help="Inter-sensor offset in meters (default 0.03)")
    args = parser.parse_args()

    resolution = 22 if args.quick else args.resolution
    offset = np.array([args.baseline, 0.0, 0.0])

    print(f"Experiment 18: dual-magnetometer backpack at "
          f"{args.baseline*100:.1f} cm baseline.")
    print(f"  Field table resolution: {resolution}^3")
    print(f"  Signal duration:        {args.duration:.1f} s")
    print(f"  Window:                 30 ms")
    print()

    results = []
    t_total = time.time()
    for i, (sensor_name, triple_key, drive, mode) in enumerate(MATRIX, 1):
        triple = TRIPLES[triple_key]
        label = (f"{sensor_name:18s} | {triple_key:9s} "
                 f"({triple[0]:>4.0f}/{triple[1]:>4.0f}/{triple[2]:>4.0f}) "
                 f"| {drive:>4.1f}A | {mode}")
        print(f"[{i}/{len(MATRIX)}] {label} ...", flush=True)
        t0 = time.time()
        try:
            r = run_one_dual(sensor_name, triple, drive, mode,
                             resolution=resolution, duration=args.duration,
                             offset_sensor=offset)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        results.append(r)
        elapsed = time.time() - t0
        s = r["single"]; d = r["dual"]
        if r["mode"] == "6-DOF":
            print(f"    SINGLE pos {s['pos_med']:.2f}/{s['pos_max']:.2f} mm "
                  f"orient {s['orient_med']:.1f}/{s['orient_max']:.1f}°")
            print(f"    DUAL   pos {d['pos_med']:.2f}/{d['pos_max']:.2f} mm "
                  f"orient {d['orient_med']:.1f}/{d['orient_max']:.1f}°  "
                  f"({elapsed:.0f}s)")
        else:
            print(f"    SINGLE pos {s['pos_med']:.2f}/{s['pos_max']:.2f} mm")
            print(f"    DUAL   pos {d['pos_med']:.2f}/{d['pos_max']:.2f} mm  "
                  f"({elapsed:.0f}s)")

    print()
    print(f"Total: {time.time() - t_total:.0f} s")

    print()
    print("Markdown summary:")
    print()
    print("| Sensor | Carrier | Drive | Mode | Single (med/max) | "
          "Dual (med/max) | Single orient (med/max) | Dual orient (med/max) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        c = r["carrier"]
        carrier_str = f"{c[0]:.0f}/{c[1]:.0f}/{c[2]:.0f}"
        s = r["single"]; d = r["dual"]
        if r["mode"] == "6-DOF":
            so = f"{s['orient_med']:.1f}° / {s['orient_max']:.1f}°"
            do = f"{d['orient_med']:.1f}° / {d['orient_max']:.1f}°"
        else:
            so = "n/a"
            do = "n/a"
        print(f"| {r['sensor']} | {carrier_str} | {r['drive']:.1f} A | "
              f"{r['mode']} | "
              f"{s['pos_med']:.2f} / {s['pos_max']:.2f} mm | "
              f"{d['pos_med']:.2f} / {d['pos_max']:.2f} mm | "
              f"{so} | {do} |")


if __name__ == "__main__":
    main()
