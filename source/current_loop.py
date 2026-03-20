"""Abstract base class for current loop geometries."""

from abc import ABC, abstractmethod
import numpy as np


class CurrentLoop(ABC):
    """Base class for all current loop types.

    Subclasses must implement:
        - magnetic_field(x, y, z): compute B-field at arbitrary point(s)
        - to_dict() / from_dict(): serialization
        - loop_type: class attribute identifying the geometry
    """

    loop_type: str = ""

    @abstractmethod
    def magnetic_field(self, x, y, z):
        """Compute the magnetic field vector at position(s) (x, y, z).

        Parameters
        ----------
        x, y, z : float or array_like
            Coordinates in meters. Can be scalars or arrays (e.g. from
            numpy.meshgrid). All must be broadcastable to the same shape.

        Returns
        -------
        Bx, By, Bz : ndarray
            Magnetic field components in Tesla, same shape as the broadcast
            shape of the inputs.
        """

    @abstractmethod
    def get_path(self, n_points=128):
        """Return points tracing the loop path for visualization.

        Parameters
        ----------
        n_points : int
            Number of points along the path.

        Returns
        -------
        path : ndarray, shape (n_points, 3)
            Ordered 3D coordinates tracing the loop. The path should form
            a closed curve (first point == last point).
        """

    @abstractmethod
    def to_dict(self):
        """Serialize this loop to a JSON-compatible dict."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data):
        """Construct a loop from a dict produced by to_dict()."""

    # ---- registry for deserialization ----

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, subclass):
        """Register a CurrentLoop subclass so from_dict dispatch works."""
        cls._registry[subclass.loop_type] = subclass
        return subclass

    @classmethod
    def create_from_dict(cls, data):
        """Create the correct CurrentLoop subclass from a serialized dict."""
        loop_type = data.get("loop_type")
        if loop_type not in cls._registry:
            raise ValueError(
                f"Unknown loop type '{loop_type}'. "
                f"Registered types: {list(cls._registry.keys())}"
            )
        return cls._registry[loop_type].from_dict(data)
