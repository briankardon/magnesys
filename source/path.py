"""Sample paths for evaluating magnetic fields along curves in 3D."""

from abc import ABC, abstractmethod

import numpy as np


class SamplePath(ABC):
    """Base class for paths through 3D space.

    Subclasses must implement get_points() and serialization.
    """

    path_type: str = ""

    @abstractmethod
    def get_points(self, n):
        """Return n evenly-spaced points along the path.

        Parameters
        ----------
        n : int
            Number of sample points.

        Returns
        -------
        points : ndarray, shape (n, 3)
            Positions in meters.
        """

    @abstractmethod
    def get_distances(self, n):
        """Return cumulative arc-length distances for n sample points.

        Parameters
        ----------
        n : int
            Number of sample points.

        Returns
        -------
        distances : ndarray, shape (n,)
            Cumulative distance along the path in meters, starting at 0.
        """

    @abstractmethod
    def to_dict(self):
        """Serialize to a JSON-compatible dict."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data):
        """Construct from a dict produced by to_dict()."""

    # ---- registry for deserialization ----

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, subclass):
        """Register a SamplePath subclass for deserialization."""
        cls._registry[subclass.path_type] = subclass
        return subclass

    @classmethod
    def create_from_dict(cls, data):
        """Create the correct SamplePath subclass from a serialized dict."""
        path_type = data.get("path_type")
        if path_type not in cls._registry:
            raise ValueError(
                f"Unknown path type '{path_type}'. "
                f"Registered types: {list(cls._registry.keys())}"
            )
        return cls._registry[path_type].from_dict(data)


@SamplePath.register
class LineSegmentPath(SamplePath):
    """A straight line segment between two endpoints.

    Parameters
    ----------
    start : array_like, shape (3,)
        Starting point [x, y, z] in meters.
    end : array_like, shape (3,)
        Ending point [x, y, z] in meters.
    """

    path_type = "line_segment"

    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        if self.start.shape != (3,):
            raise ValueError("start must have shape (3,)")
        if self.end.shape != (3,):
            raise ValueError("end must have shape (3,)")

    @property
    def length(self):
        return float(np.linalg.norm(self.end - self.start))

    def get_points(self, n):
        t = np.linspace(0, 1, n)[:, np.newaxis]
        return self.start + t * (self.end - self.start)

    def get_distances(self, n):
        return np.linspace(0, self.length, n)

    def to_dict(self):
        return {
            "path_type": self.path_type,
            "start": self.start.tolist(),
            "end": self.end.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(start=data["start"], end=data["end"])

    def __repr__(self):
        return (
            f"LineSegmentPath(start={self.start.tolist()}, "
            f"end={self.end.tolist()})"
        )


@SamplePath.register
class PolylinePath(SamplePath):
    """A piecewise-linear path through a sequence of waypoints.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Waypoint positions [x, y, z] in meters.  Must have N >= 2.
    """

    path_type = "polyline"

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if len(self.points) < 2:
            raise ValueError("polyline must have at least 2 points")

    @property
    def segment_lengths(self):
        """Length of each segment between consecutive waypoints."""
        return np.linalg.norm(np.diff(self.points, axis=0), axis=1)

    @property
    def length(self):
        return float(self.segment_lengths.sum())

    def get_points(self, n):
        seg_lens = self.segment_lengths
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cumulative[-1]
        if total == 0:
            return np.tile(self.points[0], (n, 1))

        targets = np.linspace(0, total, n)
        seg_idx = np.searchsorted(cumulative, targets, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(seg_lens) - 1)

        seg_start_d = cumulative[seg_idx]
        safe_lens = np.where(seg_lens[seg_idx] > 0, seg_lens[seg_idx], 1.0)
        t = np.where(
            seg_lens[seg_idx] > 0,
            (targets - seg_start_d) / safe_lens,
            0.0,
        )

        starts = self.points[seg_idx]
        ends = self.points[seg_idx + 1]
        return starts + t[:, np.newaxis] * (ends - starts)

    def get_distances(self, n):
        return np.linspace(0, self.length, n)

    def to_dict(self):
        return {
            "path_type": self.path_type,
            "points": self.points.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(points=data["points"])

    def __repr__(self):
        return f"PolylinePath({len(self.points)} points, length={self.length:.4g} m)"


@SamplePath.register
class SplinePath(SamplePath):
    """A smooth cubic-spline path through a sequence of waypoints.

    Uses natural cubic spline interpolation (via scipy) parametrized by
    cumulative chord length so that the curve passes exactly through every
    waypoint.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Waypoint positions [x, y, z] in meters.  Must have N >= 2.
    """

    path_type = "spline"

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if len(self.points) < 2:
            raise ValueError("spline must have at least 2 points")

    # ---- internal spline helpers ----

    def _chord_params(self):
        """Return cumulative chord-length parameter values at each waypoint."""
        diffs = np.diff(self.points, axis=0)
        chords = np.linalg.norm(diffs, axis=1)
        return np.concatenate([[0.0], np.cumsum(chords)])

    def _build_spline(self):
        """Build a cubic spline interpolator over the waypoints."""
        from scipy.interpolate import CubicSpline

        t = self._chord_params()
        # For 2 points, CubicSpline needs at least 2 knots — works fine
        return CubicSpline(t, self.points, bc_type="not-a-knot")

    def _evaluate(self, n):
        """Return n evenly-spaced-in-parameter points on the spline."""
        t = self._chord_params()
        cs = self._build_spline()
        t_eval = np.linspace(t[0], t[-1], n)
        return cs(t_eval), t_eval

    # ---- SamplePath interface ----

    @property
    def length(self):
        # Approximate arc length by evaluating many points
        pts, _ = self._evaluate(max(len(self.points) * 50, 200))
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def get_points(self, n):
        # First pass: evaluate densely to build arc-length table
        n_dense = max(len(self.points) * 50, 500)
        dense_pts, _ = self._evaluate(n_dense)
        seg_lens = np.linalg.norm(np.diff(dense_pts, axis=0), axis=1)
        cum_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum_arc[-1]
        if total == 0:
            return np.tile(self.points[0], (n, 1))

        # Second pass: resample at uniform arc length
        targets = np.linspace(0, total, n)
        idx = np.searchsorted(cum_arc, targets, side="right") - 1
        idx = np.clip(idx, 0, len(seg_lens) - 1)
        seg_start = cum_arc[idx]
        safe = np.where(seg_lens[idx] > 0, seg_lens[idx], 1.0)
        frac = np.where(seg_lens[idx] > 0, (targets - seg_start) / safe, 0.0)

        return dense_pts[idx] + frac[:, np.newaxis] * (dense_pts[idx + 1] - dense_pts[idx])

    def get_distances(self, n):
        return np.linspace(0, self.length, n)

    # ---- waypoint distances (for plot markers) ----

    @property
    def segment_lengths(self):
        """Approximate arc length of each span between consecutive waypoints."""
        n_per_seg = 50
        cs = self._build_spline()
        t = self._chord_params()
        lengths = np.empty(len(t) - 1)
        for i in range(len(t) - 1):
            ts = np.linspace(t[i], t[i + 1], n_per_seg)
            seg_pts = cs(ts)
            lengths[i] = np.sum(np.linalg.norm(np.diff(seg_pts, axis=0), axis=1))
        return lengths

    # ---- serialization ----

    def to_dict(self):
        return {
            "path_type": self.path_type,
            "points": self.points.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(points=data["points"])

    def __repr__(self):
        return f"SplinePath({len(self.points)} points, length={self.length:.4g} m)"
