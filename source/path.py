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
