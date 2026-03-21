"""Invert a B-vs-time CSV export back to a position trajectory.

Usage:
    python invert.py project.mag signal.csv [output.csv]

Arguments:
    project.mag   - The .mag file containing the coil configuration
    signal.csv    - A B-vs-time CSV exported from magnesys
                    (columns: t, x, y, z, Bx, By, Bz, Bmag[, qw, qx, qy, qz])
    output.csv    - Output trajectory CSV (default: signal_inverted.csv)

The output CSV has columns: t, x, y, z  and can be imported into
magnesys via Edit -> Import trajectory CSV.

If the signal CSV contains quaternion columns (qw, qx, qy, qz), the
inverter automatically uses 6-DOF mode (position + orientation).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from source import project
from source.inversion import FieldTable, invert_trace, invert_trace_6dof


def main():
    parser = argparse.ArgumentParser(
        description="Invert a magnetometer trace to a position trajectory",
    )
    parser.add_argument("mag_file", help="Path to the .mag project file")
    parser.add_argument("signal_csv", help="Path to the B-vs-time CSV")
    parser.add_argument("output_csv", nargs="?", default=None,
                        help="Output trajectory CSV (default: <signal>_inverted.csv)")
    parser.add_argument("--resolution", type=int, default=30,
                        help="Field table grid resolution per axis (default: 30)")
    parser.add_argument("--window-periods", type=float, default=3.0,
                        help="Demodulation window in periods of lowest freq (default: 3)")
    parser.add_argument("--bounds", type=float, nargs=6, default=None,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="Search volume bounds in meters (default: auto from coils)")
    args = parser.parse_args()

    # Load simulation
    print(f"Loading {args.mag_file}...")
    sim, viz_settings, sample_paths, trajectories = project.load(args.mag_file)
    print(f"  {len(sim.loops)} sources")

    # Load signal CSV
    print(f"Loading {args.signal_csv}...")
    with open(args.signal_csv) as f:
        skip = 0
        header_line = ""
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                skip += 1
            elif not stripped[0].lstrip("-").replace(".", "").replace("e", "").replace("+", "").isdigit():
                header_line = stripped
                skip += 1
            else:
                break
    data = np.genfromtxt(args.signal_csv, delimiter=",", skip_header=skip)
    if data.shape[1] < 7:
        print("Error: CSV must have at least 7 columns (t, x, y, z, Bx, By, Bz)")
        sys.exit(1)

    t = data[:, 0]
    signal = data[:, 4:7]  # Bx, By, Bz

    # Detect if rotation was applied (12 columns = has quaternion ground truth)
    has_rotation = data.shape[1] >= 12
    if has_rotation:
        print(f"  {len(t)} samples, {t[-1] - t[0]:.4f} s duration (with rotation)")
    else:
        print(f"  {len(t)} samples, {t[-1] - t[0]:.4f} s duration")

    # Determine bounds
    if args.bounds:
        bounds = tuple(args.bounds)
    else:
        centers = np.array([loop.center for loop in sim.loops])
        margin = 0.08
        mins = centers.min(axis=0) - margin
        maxs = centers.max(axis=0) + margin
        bounds = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
    print(f"  Search bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}] x "
          f"[{bounds[2]:.3f}, {bounds[3]:.3f}] x "
          f"[{bounds[4]:.3f}, {bounds[5]:.3f}]")

    # Build field table
    print(f"Building field table ({args.resolution}^3 = "
          f"{args.resolution**3} grid points)...")
    t0 = time.time()
    table = FieldTable(sim, bounds, resolution=args.resolution)
    print(f"  Done in {time.time() - t0:.1f} s")
    print(f"  {len(table.frequencies)} frequency channels: {table.frequencies}")

    # Invert
    if has_rotation:
        print(f"Inverting 6-DOF (window={args.window_periods} periods)...")
        t0 = time.time()
        t_pos, positions, est_rotations = invert_trace_6dof(
            table, t, signal, window_periods=args.window_periods,
        )
        print(f"  Done in {time.time() - t0:.1f} s")
    else:
        print(f"Inverting 3-DOF (window={args.window_periods} periods)...")
        t0 = time.time()
        t_pos, positions = invert_trace(
            table, t, signal, window_periods=args.window_periods,
        )
        print(f"  Done in {time.time() - t0:.1f} s")

    print(f"  {len(t_pos)} position estimates")

    # Compare to ground truth
    true_positions = data[:, 1:4]
    from scipy.interpolate import interp1d
    interp_x = interp1d(t, true_positions[:, 0], fill_value="extrapolate")
    interp_y = interp1d(t, true_positions[:, 1], fill_value="extrapolate")
    interp_z = interp1d(t, true_positions[:, 2], fill_value="extrapolate")
    true_at_inv = np.column_stack([
        interp_x(t_pos), interp_y(t_pos), interp_z(t_pos),
    ])
    errors = np.linalg.norm(positions - true_at_inv, axis=1)
    print(f"  Position error:")
    print(f"    Mean:   {errors.mean() * 1000:.2f} mm")
    print(f"    Median: {np.median(errors) * 1000:.2f} mm")
    print(f"    Max:    {errors.max() * 1000:.2f} mm")

    # Orientation error if applicable
    if has_rotation and est_rotations is not None:
        from scipy.spatial.transform import Rotation
        true_quats = data[:, 8:12]  # qw, qx, qy, qz
        interp_qw = interp1d(t, true_quats[:, 0], fill_value="extrapolate")
        interp_qx = interp1d(t, true_quats[:, 1], fill_value="extrapolate")
        interp_qy = interp1d(t, true_quats[:, 2], fill_value="extrapolate")
        interp_qz = interp1d(t, true_quats[:, 3], fill_value="extrapolate")
        true_q_at_inv = np.column_stack([
            interp_qw(t_pos), interp_qx(t_pos),
            interp_qy(t_pos), interp_qz(t_pos),
        ])
        angle_errors = []
        for i in range(len(t_pos)):
            true_rot = Rotation.from_quat(
                [true_q_at_inv[i, 1], true_q_at_inv[i, 2],
                 true_q_at_inv[i, 3], true_q_at_inv[i, 0]],  # scipy uses xyzw
            )
            diff = est_rotations[i] * true_rot.inv()
            angle_errors.append(np.rad2deg(diff.magnitude()))
        angle_errors = np.array(angle_errors)
        print(f"  Orientation error:")
        print(f"    Mean:   {angle_errors.mean():.1f} deg")
        print(f"    Median: {np.median(angle_errors):.1f} deg")
        print(f"    Max:    {angle_errors.max():.1f} deg")

    # Write output
    if args.output_csv is None:
        stem = Path(args.signal_csv).stem
        args.output_csv = str(Path(args.signal_csv).parent / f"{stem}_inverted.csv")

    print(f"Writing {args.output_csv}...")
    with open(args.output_csv, "w") as f:
        f.write("# Magnesys inversion result\n")
        f.write("t,x,y,z\n")
        for i in range(len(t_pos)):
            f.write(f"{t_pos[i]:.8e},{positions[i, 0]:.8e},"
                    f"{positions[i, 1]:.8e},{positions[i, 2]:.8e}\n")

    print("Done! Import the trajectory into magnesys via:")
    print(f"  Edit -> Import trajectory CSV -> {args.output_csv}")


if __name__ == "__main__":
    main()
