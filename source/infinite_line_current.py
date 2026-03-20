"""Infinite straight-line current with exact analytical field."""

import numpy as np

from .current_loop import CurrentLoop

MU_0 = 4e-7 * np.pi


@CurrentLoop.register
class InfiniteLineCurrent(CurrentLoop):
    """An infinitely long straight wire carrying a steady current.

    The magnetic field is B = (μ₀ I) / (2π r) in the azimuthal direction
    around the wire, where r is the perpendicular distance.

    Parameters
    ----------
    center : array_like, shape (3,)
        A point on the wire [x, y, z] in meters.
    normal : array_like, shape (3,)
        Direction of current flow (will be normalized).
    current : float
        Current in amperes.  Positive current flows in the direction of
        *normal*.
    """

    loop_type = "infinite_line"

    def __init__(self, center, normal, current=1.0):
        self.center = np.asarray(center, dtype=float)
        n = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(n)
        self.normal = n / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
        self.current = float(current)

    def characteristic_size(self):
        # No intrinsic size — return 1 mm as a reference scale for near-wire
        # threshold calculations.
        return 1e-3

    def distance_to_wire(self, x, y, z):
        x, y, z = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        # Vector from center to each point
        dx = x - self.center[0]
        dy = y - self.center[1]
        dz = z - self.center[2]

        # Project onto wire direction
        dot = dx * self.normal[0] + dy * self.normal[1] + dz * self.normal[2]

        # Perpendicular component
        px = dx - dot * self.normal[0]
        py = dy - dot * self.normal[1]
        pz = dz - dot * self.normal[2]

        return np.sqrt(px**2 + py**2 + pz**2)

    def magnetic_field(self, x, y, z):
        x, y, z = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )

        # Vector from center to each point
        dx = x - self.center[0]
        dy = y - self.center[1]
        dz = z - self.center[2]

        # Perpendicular distance vector: r_perp = r - (r·n̂)n̂
        dot = dx * self.normal[0] + dy * self.normal[1] + dz * self.normal[2]
        px = dx - dot * self.normal[0]
        py = dy - dot * self.normal[1]
        pz = dz - dot * self.normal[2]

        r_perp = np.sqrt(px**2 + py**2 + pz**2)

        # B direction is n̂ × r_perp_hat = n̂ × (r_perp / |r_perp|)
        # n̂ × r_perp (unnormalized)
        cx = self.normal[1] * pz - self.normal[2] * py
        cy = self.normal[2] * px - self.normal[0] * pz
        cz = self.normal[0] * py - self.normal[1] * px

        # |B| = μ₀ I / (2π r)
        # B_vec = (μ₀ I / (2π r²)) * (n̂ × r_perp)
        #       = (μ₀ I / (2π)) * (n̂ × r_perp) / r²
        safe_r2 = np.where(r_perp > 0, r_perp**2, 1.0)
        scale = np.where(
            r_perp > 0,
            MU_0 * self.current / (2 * np.pi * safe_r2),
            0.0,
        )

        Bx = scale * cx
        By = scale * cy
        Bz = scale * cz

        return Bx, By, Bz

    def get_path(self, n_points=128, half_length=None):
        # Render as a long line segment centered on self.center.
        if half_length is None:
            half_length = 10.0  # large default; visualization clips
        t = np.linspace(-half_length, half_length, n_points)[:, np.newaxis]
        return self.center + t * self.normal

    def to_dict(self):
        return {
            "loop_type": self.loop_type,
            "center": self.center.tolist(),
            "normal": self.normal.tolist(),
            "current": self.current,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            center=data["center"],
            normal=data["normal"],
            current=data["current"],
        )
