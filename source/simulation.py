"""Simulation: a collection of current loops and field computation."""

import json
import numpy as np
from .current_loop import CurrentLoop


class Simulation:
    """A magnetic field simulation containing one or more current loops.

    Parameters
    ----------
    loops : list of CurrentLoop, optional
        Initial collection of loops.
    """

    def __init__(self, loops=None):
        self.loops: list[CurrentLoop] = list(loops) if loops else []

    # ------------------------------------------------------------------
    # Loop management
    # ------------------------------------------------------------------

    def add_loop(self, loop):
        """Add a CurrentLoop to the simulation."""
        if not isinstance(loop, CurrentLoop):
            raise TypeError(f"Expected a CurrentLoop, got {type(loop).__name__}")
        self.loops.append(loop)

    def remove_loop(self, index):
        """Remove the loop at the given index."""
        return self.loops.pop(index)

    # ------------------------------------------------------------------
    # Field computation
    # ------------------------------------------------------------------

    def magnetic_field_at(self, x, y, z):
        """Compute the total magnetic field at point(s) (x, y, z).

        Parameters
        ----------
        x, y, z : float or array_like
            Position coordinates in meters. Scalars or arrays (must be
            broadcastable to the same shape).

        Returns
        -------
        Bx, By, Bz : ndarray
            Magnetic field components in Tesla.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        Bx = np.zeros(shape)
        By = np.zeros(shape)
        Bz = np.zeros(shape)

        for loop in self.loops:
            bx, by, bz = loop.magnetic_field(x, y, z)
            Bx += bx
            By += by
            Bz += bz

        return Bx, By, Bz

    def near_wire_mask(self, x, y, z):
        """Boolean mask that is True for points too close to any wire.

        Uses each loop's NEAR_WIRE_THRESHOLD to determine proximity.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        mask = np.zeros(shape, dtype=bool)
        for loop in self.loops:
            dist = loop.distance_to_wire(x, y, z)
            threshold = getattr(loop, 'radius', loop.diameter / 2) * loop.NEAR_WIRE_THRESHOLD
            mask |= dist < threshold
        return mask

    def magnetic_field_on_grid(self, X, Y, Z):
        """Compute the total magnetic field on a meshgrid.

        Parameters
        ----------
        X, Y, Z : ndarray
            Coordinate arrays as returned by numpy.meshgrid.

        Returns
        -------
        Bx, By, Bz : ndarray
            Field components, same shape as the input arrays.
        """
        return self.magnetic_field_at(X, Y, Z)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            "magnesys_version": 1,
            "loops": [loop.to_dict() for loop in self.loops],
        }

    @classmethod
    def from_dict(cls, data):
        version = data.get("magnesys_version", 1)
        if version != 1:
            raise ValueError(f"Unsupported file version: {version}")
        loops = [CurrentLoop.create_from_dict(d) for d in data["loops"]]
        return cls(loops=loops)

    def save(self, path):
        """Save the simulation to a JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        """Load a simulation from a JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a file previously written by save().

        Returns
        -------
        Simulation
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self):
        return f"Simulation({len(self.loops)} loop(s))"
