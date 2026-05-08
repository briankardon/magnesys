"""Experiment 16: analog TMR + nRF52832 SAADC vs integrated digital parts.

Re-runs Exp. 15's matrix with a new SensorProfile representing a
realistic wearable analog signal chain (analog TMR magnetometer +
AD8421 instrumentation amp + nRF52832 SAADC at 50 kSPS/axis). This
profile lifts both Exp. 15 constraints simultaneously:

  * sample rate: 50 kSPS instead of 1-2.5 kSPS -> can run a true
    1 kHz lock-in like the original Exp. 13-14 simulation assumed
  * noise floor: 5 nT/sqrt(Hz) instead of 30-70 nT/sqrt(Hz) -> ~10x
    less per-sample sigma at the same sample rate

Field-table resolution is bumped to 35 to push the grid-quantization
floor below the ~17 mm seen in Exp. 15 at res 25.

Run as a script:
    python demos/experiment_16_analog_frontend.py [--quick]
"""

import argparse
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the rig + run-one helpers from Exp. 15
from experiment_15_sensor_realistic import (
    build_1m_cube_cage, cage_path, run_one,
)


# Carrier triples we want to test on the analog chain.  The "very_high"
# row reproduces the original Exp. 13/14 carriers — the whole point of
# moving to an analog frontend is to make this triple actually feasible.
TRIPLES = {
    "low":       (100.0,  137.0,  173.0),
    "med":       (300.0,  411.0,  519.0),
    "high":      (500.0,  687.0,  873.0),
    "very_high": (1000.0, 1373.0, 1747.0),
}

# (sensor, triple_key, drive_A, mode)
MATRIX = [
    # --- carrier sweep at 5 A drive, 3-DOF ---
    ("TMR_analog_nRF52", "low",       5.0, "3-DOF"),
    ("TMR_analog_nRF52", "med",       5.0, "3-DOF"),
    ("TMR_analog_nRF52", "high",      5.0, "3-DOF"),
    ("TMR_analog_nRF52", "very_high", 5.0, "3-DOF"),
    # --- 6-DOF at the highest carrier ---
    ("TMR_analog_nRF52", "very_high", 5.0, "6-DOF"),
    # --- drive-current sweep at the highest carrier, 3-DOF ---
    ("TMR_analog_nRF52", "very_high", 1.0,  "3-DOF"),
    ("TMR_analog_nRF52", "very_high", 10.0, "3-DOF"),
    # --- ideal upper bound at the same carrier (geometric floor) ---
    ("ideal", "very_high", 5.0, "3-DOF"),
    ("ideal", "very_high", 5.0, "6-DOF"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Lower-resolution table for fast iteration")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Signal duration in seconds (default 3.0)")
    parser.add_argument("--resolution", type=int, default=35,
                        help="Field-table resolution (default 35)")
    args = parser.parse_args()

    resolution = 22 if args.quick else args.resolution

    print(f"Experiment 16: analog frontend + nRF52832 SAADC.")
    print(f"  Field table resolution: {resolution}^3")
    print(f"  Signal duration:        {args.duration:.1f} s")
    print(f"  Window:                 30 ms (Exp. 13 standard)")
    print()

    results = []
    t_total = time.time()
    for i, (sensor_name, triple_key, drive, mode) in enumerate(MATRIX, 1):
        triple = TRIPLES[triple_key]
        label = (f"{sensor_name:18s} | {triple_key:9s} "
                 f"({triple[0]:>4.0f}/{triple[1]:>4.0f}/{triple[2]:>4.0f}) "
                 f"| {drive:>4.1f}A | {mode}")
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
        print(f"    sigma {r['sigma_uT']:.3f} uT  |B| {r['mean_B_uT']:.2f} uT  "
              f"SNR {r['snr']:.1f}  |  pos {r['pos_med_mm']:.2f}/"
              f"{r['pos_max_mm']:.2f} mm{orient_str}  ({elapsed:.0f}s)")

    print()
    print(f"Total: {time.time() - t_total:.0f} s")

    # Markdown summary
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
              f"{r['fs']:.0f} Hz | {r['sigma_uT']:.3f} uT | "
              f"{r['snr']:.1f} | {r['mode']} | "
              f"{r['pos_med_mm']:.2f} / {r['pos_max_mm']:.2f} mm | "
              f"{orient} |")


if __name__ == "__main__":
    main()
