"""Experiment 19: dual-sensor inversion on a smooth Lissajous trajectory.

Replays the Exp. 18 dual-vs-single matrix on a LissajousPath traversed
in "loop" mode. The Lissajous figure is naturally closed (start point
== end point after one period) and smooth everywhere, so:

  - The bird's path tangent never reverses (no 180° flip in the
    base sensor orientation generated from the path tangent)
  - The "loop" traversal_mode wraps without ping-pong, eliminating any
    endpoint-bounce artifact from synthesize_signal
  - The true rotation history fed to the error metric is smooth, so the
    SLERP truth interpolation is well-defined throughout

If the deterministic ~170° orientation max errors from Exp. 17-18
*disappear* on this path, the cause was the bouncing tangent / SLERP
discontinuity (despite my earlier analysis suggesting the bounce
shouldn't trigger at this duration).

If the ~170° max errors *persist*, the cause is in the inversion
pipeline itself — most likely the SVD rotation initializer or the
rotvec parameterization in `_refine_6dof`.

Either outcome is informative.

Run as a script:
    python demos/experiment_19_lissajous_trajectory.py [--quick]
"""

import argparse
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source.path import LissajousPath
from experiment_18_dual_magnetometer import (
    DEFAULT_OFFSET_SENSOR, run_one_dual,
)


# Lissajous trajectory tuned for a 1 m cube cage interior. Coprime
# integer ratios (3, 4, 5) give a varied 3D figure; amplitudes ±0.35 m
# keep the bird inside ±0.40 m of the cage center.
def lissajous_cage_path():
    return LissajousPath(
        amplitudes=[0.35, 0.30, 0.25],
        ratios=[3, 4, 5],
        phases=[0.0, np.pi / 2, np.pi / 4],
    )


# Same carrier triples as Exp. 18 for direct comparison
TRIPLES = {
    "low":       (100.0,  137.0,  173.0),
    "very_high": (1000.0, 1373.0, 1747.0),
}

MATRIX = [
    # 3-DOF baseline on Lissajous (sanity check)
    ("TMR_analog_nRF52", "very_high",  5.0, "3-DOF"),
    # 6-DOF: the headline test — does the ~170° max disappear?
    ("TMR_analog_nRF52", "very_high",  5.0, "6-DOF"),
    ("ideal",            "very_high",  5.0, "6-DOF"),
    ("TMR_analog_nRF52", "very_high", 10.0, "6-DOF"),
    ("TMR_analog_nRF52", "very_high",  1.0, "6-DOF"),
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
    path = lissajous_cage_path()

    print(f"Experiment 19: Lissajous trajectory, loop traversal.")
    print(f"  Path: {path}")
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
                             offset_sensor=offset,
                             path=path, traversal_mode="loop",
                             use_robust=True)
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
