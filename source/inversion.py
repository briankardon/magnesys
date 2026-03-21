"""Magnetic field inversion: reconstruct position from magnetometer readings.

This module implements the pipeline for inverting a time-varying magnetic
field measurement back to a position trace.  It assumes multi-frequency
AC excitation where each current source oscillates at a distinct known
frequency, so individual source contributions can be separated via
lock-in detection (demodulation).

Sources that share a frequency (e.g. an anti-Helmholtz pair) are grouped
together — their combined field at each frequency is what the demodulator
measures.

Pipeline
--------
1. **FieldTable** — precompute per-frequency combined field on a 3D grid
2. **demodulate** — extract per-frequency field vectors from a time-domain signal
3. **invert_trace** — find the position that best matches the measurements

The inversion uses a two-stage approach:
  - Coarse: brute-force nearest-neighbor search on the precomputed table
  - Fine: scipy.optimize.least_squares refinement from the coarse estimate
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation, Slerp


class FieldTable:
    """Precomputed per-frequency combined field vectors on a 3D grid.

    Sources sharing the same frequency are summed, matching what
    lock-in demodulation actually measures.

    Parameters
    ----------
    simulation : Simulation
        The simulation containing the current sources.
    bounds : tuple of 6 floats
        (x_min, x_max, y_min, y_max, z_min, z_max) in meters.
    resolution : int
        Number of grid points per axis.
    """

    def __init__(self, simulation, bounds, resolution=30):
        self.simulation = simulation
        self.bounds = bounds
        self.resolution = resolution

        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        zs = np.linspace(z_min, z_max, resolution)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        self.grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        n_pts = len(self.grid_points)
        gx = self.grid_points[:, 0]
        gy = self.grid_points[:, 1]
        gz = self.grid_points[:, 2]

        # Group AC sources by frequency
        ac_sources = [
            loop for loop in simulation.loops if loop.frequency > 0
        ]
        freq_groups = {}  # freq → [list of sources]
        for source in ac_sources:
            freq_groups.setdefault(source.frequency, []).append(source)

        self.unique_frequencies = np.array(sorted(freq_groups.keys()))
        n_channels = len(self.unique_frequencies)
        self._freq_groups = freq_groups

        # Precompute combined field per frequency channel at every grid point.
        # Shape: (n_channels, n_grid_points, 3)
        self.channel_fields = np.zeros((n_channels, n_pts, 3))

        for ch, freq in enumerate(self.unique_frequencies):
            for source in freq_groups[freq]:
                bx, by, bz = source.magnetic_field(gx, gy, gz)
                self.channel_fields[ch, :, 0] += np.asarray(bx).ravel()
                self.channel_fields[ch, :, 1] += np.asarray(by).ravel()
                self.channel_fields[ch, :, 2] += np.asarray(bz).ravel()

        # Build concatenated feature vector per grid point:
        # [ch0_Bx, ch0_By, ch0_Bz, ch1_Bx, ...] — length 3 * n_channels
        self.feature_vectors = self.channel_fields.transpose(1, 0, 2).reshape(
            n_pts, n_channels * 3,
        )

        # Normalize for KD-tree so each channel has equal weight
        self._feature_scales = np.empty(n_channels * 3)
        for ch in range(n_channels):
            cols = self.feature_vectors[:, ch * 3:(ch + 1) * 3]
            scale = np.std(cols) if np.std(cols) > 0 else 1.0
            self._feature_scales[ch * 3:(ch + 1) * 3] = scale

        normalized = self.feature_vectors / self._feature_scales
        self._kdtree = KDTree(normalized)

        # Rotation-invariant KD-tree: per-channel field magnitudes
        # |B_ch| is invariant under sensor rotation
        self._mag_features = np.empty((n_pts, n_channels))
        for ch in range(n_channels):
            self._mag_features[:, ch] = np.linalg.norm(
                self.channel_fields[ch], axis=1,
            )
        mag_scales = np.empty(n_channels)
        for ch in range(n_channels):
            s = np.std(self._mag_features[:, ch])
            mag_scales[ch] = s if s > 0 else 1.0
        self._mag_scales = mag_scales
        self._kdtree_mag = KDTree(self._mag_features / mag_scales)

    @property
    def frequencies(self):
        """Unique excitation frequencies (Hz)."""
        return self.unique_frequencies

    def query_coarse(self, measurements):
        """Find the grid point closest to the measurement vector.

        Parameters
        ----------
        measurements : ndarray, shape (n_channels, 3)
            Demodulated field vectors, one per frequency channel.

        Returns
        -------
        position : ndarray, shape (3,)
            Estimated position (nearest grid point).
        """
        feat = measurements.ravel() / self._feature_scales
        _, idx = self._kdtree.query(feat)
        return self.grid_points[idx].copy()

    def query_coarse_rotated(self, measurements):
        """Find the nearest grid point using rotation-invariant features.

        Uses per-channel field magnitudes which are unaffected by sensor
        rotation.

        Parameters
        ----------
        measurements : ndarray, shape (n_channels, 3)
            Demodulated field vectors (in sensor frame).

        Returns
        -------
        position : ndarray, shape (3,)
        """
        mags = np.linalg.norm(measurements, axis=1)
        feat = mags / self._mag_scales
        _, idx = self._kdtree_mag.query(feat)
        return self.grid_points[idx].copy()

    def field_at(self, position):
        """Compute per-channel combined field at an arbitrary position.

        Parameters
        ----------
        position : ndarray, shape (3,)
            Position in meters.

        Returns
        -------
        fields : ndarray, shape (n_channels, 3)
            Combined field vector per frequency channel.
        """
        x, y, z = position
        fields = np.zeros((len(self.unique_frequencies), 3))
        for ch, freq in enumerate(self.unique_frequencies):
            for source in self._freq_groups[freq]:
                bx, by, bz = source.magnetic_field(x, y, z)
                fields[ch, 0] += float(bx)
                fields[ch, 1] += float(by)
                fields[ch, 2] += float(bz)
        return fields

    def refine(self, measurements, initial_guess):
        """Refine a position estimate using nonlinear least squares.

        Parameters
        ----------
        measurements : ndarray, shape (n_channels, 3)
            Demodulated field vectors.
        initial_guess : ndarray, shape (3,)
            Starting position (e.g. from coarse search).

        Returns
        -------
        position : ndarray, shape (3,)
            Refined position estimate.
        """
        target = measurements.ravel()
        scales = self._feature_scales

        def residual(pos):
            pred = self.field_at(pos).ravel()
            return (pred - target) / scales

        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        result = least_squares(
            residual,
            initial_guess,
            bounds=([x_min, y_min, z_min], [x_max, y_max, z_max]),
            method="trf",
        )
        return result.x


def demodulate(t, signal, frequencies):
    """Extract per-frequency field vectors from a time-domain signal.

    Uses lock-in detection: multiply by cos at each frequency and average.

    Parameters
    ----------
    t : ndarray, shape (N,)
        Time samples in seconds.
    signal : ndarray, shape (N, 3)
        Measured Bx, By, Bz at each time step.
    frequencies : ndarray, shape (K,)
        Unique frequencies to demodulate.

    Returns
    -------
    amplitudes : ndarray, shape (K, 3)
        Demodulated field vector for each frequency channel.
    """
    K = len(frequencies)
    amplitudes = np.empty((K, 3))
    for k, f in enumerate(frequencies):
        ref = np.cos(2.0 * np.pi * f * t)
        for axis in range(3):
            amplitudes[k, axis] = 2.0 * np.mean(ref * signal[:, axis])
    return amplitudes


def invert_trace(field_table, t, signal, window_periods=1.0):
    """Invert a magnetometer time series to a position trace.

    Parameters
    ----------
    field_table : FieldTable
        Precomputed field lookup table.
    t : ndarray, shape (N,)
        Time samples in seconds.
    signal : ndarray, shape (N, 3)
        Measured Bx, By, Bz at each time step.
    window_periods : float
        Demodulation window length in periods of the lowest frequency.
        Longer windows give better frequency separation but coarser
        time resolution.

    Returns
    -------
    t_positions : ndarray, shape (M,)
        Time at the center of each demodulation window.
    positions : ndarray, shape (M, 3)
        Estimated positions.
    """
    freqs = field_table.frequencies
    min_freq = freqs.min()
    window_duration = window_periods / min_freq
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    window_samples = max(int(window_duration / dt), 2)

    # Sliding window with 50% overlap
    step = max(window_samples // 2, 1)
    n_windows = max((len(t) - window_samples) // step + 1, 1)

    t_positions = np.empty(n_windows)
    positions = np.empty((n_windows, 3))

    prev_pos = None

    for i in range(n_windows):
        start = i * step
        end = start + window_samples
        if end > len(t):
            end = len(t)
            start = max(end - window_samples, 0)

        t_win = t[start:end]
        sig_win = signal[start:end]

        # Demodulate this window
        measurements = demodulate(t_win, sig_win, freqs)

        # Coarse search
        coarse = field_table.query_coarse(measurements)

        # Use previous position as initial guess if close enough
        if prev_pos is not None:
            dist = np.linalg.norm(coarse - prev_pos)
            grid_spacing = (
                (field_table.bounds[1] - field_table.bounds[0])
                / field_table.resolution
            )
            if dist < grid_spacing * 3:
                initial = prev_pos
            else:
                initial = coarse
        else:
            initial = coarse

        # Refine
        refined = field_table.refine(measurements, initial)

        t_positions[i] = (t_win[0] + t_win[-1]) / 2.0
        positions[i] = refined
        prev_pos = refined

    return t_positions, positions


# ------------------------------------------------------------------
# Rotation utilities
# ------------------------------------------------------------------

def generate_rotations(path_points, n_samples, max_perturbation_deg=30.0,
                       n_keypoints=8, seed=None):
    """Generate smooth random rotations along a path.

    The base orientation aligns the sensor's +x axis with the path tangent
    (like a bird looking where it's going).  Smooth random roll/pitch/yaw
    perturbations are added on top.

    Parameters
    ----------
    path_points : ndarray, shape (N, 3)
        Points along the path (used to compute tangent direction).
    n_samples : int
        Number of rotation samples to generate.
    max_perturbation_deg : float
        Maximum random perturbation in degrees for each Euler angle.
    n_keypoints : int
        Number of random keypoints for SLERP interpolation.
        More keypoints = faster orientation changes.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    rotations : list of Rotation
        Rotation objects (lab-to-sensor frame) at each sample point.
    """
    rng = np.random.default_rng(seed)

    # Compute path tangents
    if len(path_points) < 2:
        return [Rotation.identity()] * n_samples

    # Tangent at each sample (forward difference, normalized)
    diffs = np.diff(path_points, axis=0)
    # Extend last tangent
    tangents = np.vstack([diffs, diffs[-1:]])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    tangents = tangents / norms

    # Resample tangents to n_samples
    frac_orig = np.linspace(0, 1, len(tangents))
    frac_out = np.linspace(0, 1, n_samples)
    tangents_resampled = np.column_stack([
        np.interp(frac_out, frac_orig, tangents[:, i]) for i in range(3)
    ])
    # Re-normalize after interpolation
    tn = np.linalg.norm(tangents_resampled, axis=1, keepdims=True)
    tn = np.where(tn > 0, tn, 1.0)
    tangents_resampled = tangents_resampled / tn

    # Build base rotations: align +x with tangent
    base_rotations = []
    for tang in tangents_resampled:
        # Rotation that takes [1,0,0] to tang
        v = np.cross([1, 0, 0], tang)
        c = np.dot([1, 0, 0], tang)
        if np.linalg.norm(v) < 1e-10:
            if c > 0:
                base_rotations.append(Rotation.identity())
            else:
                base_rotations.append(Rotation.from_rotvec([0, np.pi, 0]))
        else:
            angle = np.arctan2(np.linalg.norm(v), c)
            axis = v / np.linalg.norm(v)
            base_rotations.append(Rotation.from_rotvec(axis * angle))

    # Generate smooth random perturbations via SLERP between keypoints
    max_rad = np.deg2rad(max_perturbation_deg)
    key_fracs = np.linspace(0, 1, n_keypoints)
    key_angles = rng.uniform(-max_rad, max_rad, size=(n_keypoints, 3))
    # Ensure start/end are mild
    key_angles[0] *= 0.3
    key_angles[-1] *= 0.3
    key_rots = Rotation.from_euler("xyz", key_angles)
    slerp = Slerp(key_fracs, key_rots)
    perturbations = slerp(frac_out)

    # Combine: sensor_rotation = perturbation * base
    rotations = []
    for base, pert in zip(base_rotations, perturbations):
        rotations.append(pert * base)

    return rotations


def apply_rotation_to_field(Bx, By, Bz, rotations):
    """Rotate lab-frame field vectors into the sensor frame.

    Parameters
    ----------
    Bx, By, Bz : ndarray, shape (N,)
        Lab-frame field components.
    rotations : list of Rotation, length N
        Lab-to-sensor rotation at each sample.

    Returns
    -------
    Bx_rot, By_rot, Bz_rot : ndarray, shape (N,)
        Sensor-frame field components.
    """
    n = len(Bx)
    B_lab = np.column_stack([Bx, By, Bz])
    B_sensor = np.empty_like(B_lab)
    for i in range(n):
        B_sensor[i] = rotations[i].apply(B_lab[i])
    return B_sensor[:, 0], B_sensor[:, 1], B_sensor[:, 2]


# ------------------------------------------------------------------
# 6-DOF inversion (position + orientation)
# ------------------------------------------------------------------

def _estimate_rotation(lab_fields, sensor_fields):
    """Estimate rotation from lab-frame to sensor-frame fields (Wahba's problem).

    Given pairs of corresponding vectors in two frames, find the rotation R
    such that sensor_fields[k] ~ R @ lab_fields[k] for all k.

    Uses SVD of the cross-covariance matrix.

    Parameters
    ----------
    lab_fields : ndarray, shape (K, 3)
    sensor_fields : ndarray, shape (K, 3)

    Returns
    -------
    rotvec : ndarray, shape (3,)
    """
    # Cross-covariance: H = sum( sensor_k @ lab_k^T )
    H = sensor_fields.T @ lab_fields  # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    # Ensure proper rotation (det = +1)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    return Rotation.from_matrix(R).as_rotvec()


def _refine_6dof(field_table, measurements, initial_pos, initial_rotvec=None):
    """Refine position and orientation using nonlinear least squares.

    Parameters
    ----------
    field_table : FieldTable
    measurements : ndarray, shape (n_channels, 3)
        Demodulated field in the sensor frame.
    initial_pos : ndarray, shape (3,)
    initial_rotvec : ndarray, shape (3,), optional
        Initial rotation vector (default: identity).

    Returns
    -------
    position : ndarray, shape (3,)
    rotation : Rotation
    """
    if initial_rotvec is None:
        initial_rotvec = np.zeros(3)

    target = measurements.ravel()
    scales = field_table._feature_scales

    def residual(params):
        pos = params[:3]
        rotvec = params[3:6]
        R = Rotation.from_rotvec(rotvec)
        lab_fields = field_table.field_at(pos)  # (n_ch, 3)
        # Rotate each channel's field into sensor frame
        sensor_fields = np.empty_like(lab_fields)
        for ch in range(len(lab_fields)):
            sensor_fields[ch] = R.apply(lab_fields[ch])
        return (sensor_fields.ravel() - target) / scales

    x0 = np.concatenate([initial_pos, initial_rotvec])

    x_min, x_max, y_min, y_max, z_min, z_max = field_table.bounds
    lb = [x_min, y_min, z_min, -np.pi, -np.pi, -np.pi]
    ub = [x_max, y_max, z_max, np.pi, np.pi, np.pi]

    result = least_squares(residual, x0, bounds=(lb, ub), method="trf")

    pos = result.x[:3]
    rot = Rotation.from_rotvec(result.x[3:6])
    return pos, rot


def invert_trace_6dof(field_table, t, signal, window_periods=1.0):
    """Invert a rotated magnetometer time series to position + orientation.

    Parameters
    ----------
    field_table : FieldTable
    t : ndarray, shape (N,)
    signal : ndarray, shape (N, 3)
        Sensor-frame Bx, By, Bz (field rotated by sensor orientation).
    window_periods : float

    Returns
    -------
    t_positions : ndarray, shape (M,)
    positions : ndarray, shape (M, 3)
    rotations : list of Rotation, length M
    """
    freqs = field_table.frequencies
    min_freq = freqs.min()
    window_duration = window_periods / min_freq
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    window_samples = max(int(window_duration / dt), 2)

    step = max(window_samples // 2, 1)
    n_windows = max((len(t) - window_samples) // step + 1, 1)

    t_positions = np.empty(n_windows)
    positions = np.empty((n_windows, 3))
    rotations = []

    prev_pos = None
    prev_rotvec = None

    for i in range(n_windows):
        start = i * step
        end = start + window_samples
        if end > len(t):
            end = len(t)
            start = max(end - window_samples, 0)

        t_win = t[start:end]
        sig_win = signal[start:end]

        measurements = demodulate(t_win, sig_win, freqs)

        # Coarse position search using rotation-invariant magnitudes
        coarse = field_table.query_coarse_rotated(measurements)

        if prev_pos is not None:
            dist = np.linalg.norm(coarse - prev_pos)
            grid_spacing = (
                (field_table.bounds[1] - field_table.bounds[0])
                / field_table.resolution
            )
            initial_pos = prev_pos if dist < grid_spacing * 3 else coarse
        else:
            initial_pos = coarse

        # Estimate initial rotation from coarse position using SVD
        # (Wahba's problem: find R that best maps lab fields to measurements)
        if prev_rotvec is not None:
            init_rotvec = prev_rotvec
        else:
            lab_fields = field_table.field_at(initial_pos)
            init_rotvec = _estimate_rotation(lab_fields, measurements)

        # 6-DOF refinement
        pos, rot = _refine_6dof(
            field_table, measurements, initial_pos,
            initial_rotvec=init_rotvec,
        )

        t_positions[i] = (t_win[0] + t_win[-1]) / 2.0
        positions[i] = pos
        rotations.append(rot)
        prev_pos = pos
        prev_rotvec = rot.as_rotvec()

    return t_positions, positions, rotations
