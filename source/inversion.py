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
from scipy.optimize import least_squares
from scipy.spatial import KDTree


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
