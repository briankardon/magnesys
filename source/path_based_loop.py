"""Base class for current loops whose field is computed via Biot-Savart."""

import warnings

import numpy as np
from .current_loop import CurrentLoop

# Permeability of free space (T·m/A)
MU_0 = 4e-7 * np.pi


class PathBasedLoop(CurrentLoop):
    """Intermediate base for loops computed via numerical Biot-Savart integration.

    Subclasses only need to implement get_path(), to_dict(), from_dict(),
    and set loop_type.  magnetic_field() and distance_to_wire() are provided
    automatically by integrating over the discretized wire path.

    Class attributes
    ----------------
    n_integration_points : int
        Number of points used to discretize the path for Biot-Savart
        integration.  Higher values give more accuracy at the cost of
        computation time.  Default 2048.
    """

    n_integration_points: int = 2048

    # ------------------------------------------------------------------
    # Magnetic field via Biot-Savart
    # ------------------------------------------------------------------

    def magnetic_field(self, x, y, z):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        x = np.broadcast_to(x, shape)
        y = np.broadcast_to(y, shape)
        z = np.broadcast_to(z, shape)

        path = self.get_path(self.n_integration_points)
        # path is closed: path[0] ≈ path[-1], giving n-1 segments
        n_seg = len(path) - 1

        Bx = np.zeros(shape)
        By = np.zeros(shape)
        Bz = np.zeros(shape)

        # Check for near-wire points and warn
        dist = self.distance_to_wire(x, y, z)
        min_dist = np.min(dist)
        char_size = self.characteristic_size()
        if min_dist < char_size * self.NEAR_WIRE_THRESHOLD:
            warnings.warn(
                f"One or more field points are within {min_dist:.2e} m of the "
                f"wire (characteristic size {char_size:.4g} m). The ideal "
                f"thin-wire model diverges at the wire; results this close "
                f"are not physical.",
                stacklevel=2,
            )

        # Loop over segments; each iteration is vectorized over field points
        for i in range(n_seg):
            dl = path[i + 1] - path[i]  # (3,)
            mid = 0.5 * (path[i] + path[i + 1])  # (3,)

            # Displacement from segment midpoint to field points
            rx = x - mid[0]
            ry = y - mid[1]
            rz = z - mid[2]
            r_sq = rx**2 + ry**2 + rz**2
            r_mag = np.sqrt(r_sq)

            # Avoid division by zero (point on wire)
            safe_r3 = np.where(r_mag > 0, r_mag**3, 1.0)

            # dl × r
            cx = dl[1] * rz - dl[2] * ry
            cy = dl[2] * rx - dl[0] * rz
            cz = dl[0] * ry - dl[1] * rx

            inv_r3 = 1.0 / safe_r3
            Bx += cx * inv_r3
            By += cy * inv_r3
            Bz += cz * inv_r3

        prefactor = MU_0 * self.current / (4.0 * np.pi)
        return prefactor * Bx, prefactor * By, prefactor * Bz

    # ------------------------------------------------------------------
    # Distance to wire
    # ------------------------------------------------------------------

    def distance_to_wire(self, x, y, z):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        x = np.broadcast_to(x, shape).ravel()
        y = np.broadcast_to(y, shape).ravel()
        z = np.broadcast_to(z, shape).ravel()

        pts = np.column_stack([x, y, z])  # (M, 3)

        # Use a moderate number of path points for distance checking
        path = self.get_path(min(self.n_integration_points, 512))
        n_seg = len(path) - 1

        min_dist_sq = np.full(len(pts), np.inf)

        for i in range(n_seg):
            a = path[i]      # (3,)
            b = path[i + 1]  # (3,)
            ab = b - a
            ab_sq = np.dot(ab, ab)

            if ab_sq < 1e-30:
                # Degenerate segment
                d_sq = np.sum((pts - a) ** 2, axis=1)
            else:
                # Project each point onto the segment [0, 1]
                ap = pts - a  # (M, 3)
                t = np.clip(ap @ ab / ab_sq, 0.0, 1.0)  # (M,)
                closest = a + t[:, np.newaxis] * ab  # (M, 3)
                d_sq = np.sum((pts - closest) ** 2, axis=1)

            min_dist_sq = np.minimum(min_dist_sq, d_sq)

        return np.sqrt(min_dist_sq).reshape(shape)

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def characteristic_size(self):
        """Return a characteristic length for the near-wire threshold.

        Subclasses can override this.  Default uses the bounding box of
        the path.
        """
        path = self.get_path(64)
        extent = path.max(axis=0) - path.min(axis=0)
        return np.max(extent) / 2.0
