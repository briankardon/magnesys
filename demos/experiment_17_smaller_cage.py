"""Experiment 17: shrink tracking volume, keep coil geometry.

Tests the hypothesis from Exp. 15-16 that the 6-DOF orientation
catastrophic outliers (170 deg max even for an ideal sensor) come from
the bird visiting cage edges where field directions deviate from
cardinal axes — degrading the orientation-first SVD initializer.

Setup is identical to Exp. 16 except the bird is restricted to a
0.75 m cube (path stays inside ±0.375 m, field-table bounds match).
Coils, drive currents, frequencies, sensor profile, window, and field-
table resolution are all unchanged so any improvement is attributable
to the smaller tracking volume.

Run as a script:
    python demos/experiment_17_smaller_cage.py
"""

import argparse
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source.path import SplinePath
from experiment_15_sensor_realistic import run_one


def cage_path_075m():
    """Spline through ~±0.30 m — fully interior to a 0.75 m cube cage."""
    pts = np.array([
        [-0.25, -0.20, -0.16],
        [+0.08, +0.25, -0.13],
        [+0.30, -0.16, +0.20],
        [-0.13, +0.13, +0.30],
        [+0.20, +0.08, -0.25],
    ])
    return SplinePath(pts)


# Field-table bounds matching the 0.75 m cube cage (±0.375 m, with a
# small margin so the optimizer can move slightly outside the cage).
BOUNDS_075M = (-0.40, 0.40, -0.40, 0.40, -0.40, 0.40)

# Carrier triples reused from Exp. 16
TRIPLES = {
    "low":       (100.0,  137.0,  173.0),
    "very_high": (1000.0, 1373.0, 1747.0),
}

# (sensor, triple_key, drive_A, mode)
MATRIX = [
    # 3-DOF sanity checks (should be similar to Exp. 16)
    ("TMR_analog_nRF52", "very_high", 5.0, "3-DOF"),
    ("ideal",            "very_high", 5.0, "3-DOF"),
    # The actual test: 6-DOF at the smaller volume
    ("TMR_analog_nRF52", "very_high", 5.0, "6-DOF"),
    ("ideal",            "very_high", 5.0, "6-DOF"),
    # Drive sweep at 6-DOF — does more SNR rescue the orientation init?
    ("TMR_analog_nRF52", "very_high", 10.0, "6-DOF"),
    # Low-carrier 6-DOF as a robustness check
    ("TMR_analog_nRF52", "low",       5.0, "6-DOF"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Signal duration in seconds (default 3.0)")
    parser.add_argument("--resolution", type=int, default=35,
                        help="Field-table resolution (default 35)")
    args = parser.parse_args()

    print("Experiment 17: 0.75 m cube tracking volume, coils unchanged.")
    print(f"  Field table resolution: {args.resolution}^3")
    print(f"  Field table bounds:     ±0.40 m")
    print(f"  Path span:              ~±0.30 m (interior to 0.75 m cube)")
    print(f"  Signal duration:        {args.duration:.1f} s")
    print()

    path = cage_path_075m()

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
            r = run_one(sensor_name, triple, drive, mode,
                        resolution=args.resolution, duration=args.duration,
                        path=path, bounds=BOUNDS_075M)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        results.append(r)
        elapsed = time.time() - t0
        orient_str = (f"  orient {r['orient_med_deg']:.1f}/{r['orient_max_deg']:.1f} deg"
                      if r["orient_med_deg"] is not None else "")
        print(f"    sigma {r['sigma_uT']:.3f} uT  |B| {r['mean_B_uT']:.2f} uT  "
              f"SNR {r['snr']:.1f}  |  pos {r['pos_med_mm']:.2f}/"
              f"{r['pos_max_mm']:.2f} mm{orient_str}  ({elapsed:.0f}s)")

    print()
    print(f"Total: {time.time() - t_total:.0f} s")

    # Markdown summary
    print()
    print("Markdown summary:")
    print()
    print("| Sensor | Carrier (Hz) | Drive | Mode | Pos err (med/max) | "
          "Orient err (med/max) |")
    print("|--------|--------------|-------|------|-------------------|"
          "---------------------|")
    for r in results:
        c = r["carrier"]
        carrier_str = f"{c[0]:.0f}/{c[1]:.0f}/{c[2]:.0f}"
        orient = (f"{r['orient_med_deg']:.1f}deg / {r['orient_max_deg']:.1f}deg"
                  if r["orient_med_deg"] is not None else "n/a")
        print(f"| {r['sensor']} | {carrier_str} | {r['drive']:.1f} A | "
              f"{r['mode']} | "
              f"{r['pos_med_mm']:.2f} / {r['pos_max_mm']:.2f} mm | "
              f"{orient} |")


if __name__ == "__main__":
    main()
