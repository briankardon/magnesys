"""Circular current loop with exact magnetic field via elliptic integrals."""

import warnings

import numpy as np
from scipy.special import ellipk, ellipe
from .current_loop import CurrentLoop

# Permeability of free space (T·m/A)
MU_0 = 4e-7 * np.pi


@CurrentLoop.register
class CircularCurrentLoop(CurrentLoop):
    """An ideal circular current loop.

    Parameters
    ----------
    diameter : float
        Loop diameter in meters.
    center : array_like, shape (3,)
        Center position [x, y, z] in meters.
    normal : array_like, shape (3,)
        Unit vector along the loop axis (right-hand rule w.r.t. current).
    current : float
        Current in amperes.
    """

    loop_type = "circular"

    def __init__(self, diameter, center, normal, current, frequency=0.0, phase=0.0):
        self.diameter = float(diameter)
        self.center = np.asarray(center, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.current = float(current)
        self.frequency = float(frequency)
        self.phase = float(phase)

        if self.center.shape != (3,):
            raise ValueError("center must have shape (3,)")
        if self.normal.shape != (3,):
            raise ValueError("normal must have shape (3,)")

        # Normalize the normal vector
        norm = np.linalg.norm(self.normal)
        if norm == 0:
            raise ValueError("normal vector must be nonzero")
        self.normal = self.normal / norm

    @property
    def radius(self):
        return self.diameter / 2.0

    def characteristic_size(self):
        return self.radius

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _to_local_cylindrical(self, x, y, z):
        """Transform lab-frame points to local (rho, z) cylindrical coords."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        x = np.broadcast_to(x, shape).copy()
        y = np.broadcast_to(y, shape).copy()
        z = np.broadcast_to(z, shape).copy()

        x -= self.center[0]
        y -= self.center[1]
        z -= self.center[2]

        R = self._rotation_to_loop_frame()
        pts = np.stack([x, y, z], axis=-1)
        pts_local = pts @ R

        xL = pts_local[..., 0]
        yL = pts_local[..., 1]
        zL = pts_local[..., 2]
        rho = np.sqrt(xL**2 + yL**2)
        return rho, zL, xL, yL

    def distance_to_wire(self, x, y, z):
        rho, zL, _, _ = self._to_local_cylindrical(x, y, z)
        return np.sqrt((rho - self.radius) ** 2 + zL**2)

    # ------------------------------------------------------------------
    # Magnetic field
    # ------------------------------------------------------------------

    def magnetic_field(self, x, y, z):
        """Compute B-field at point(s) (x, y, z) using elliptic integrals.

        Uses the exact off-axis solution for a circular current loop.
        The field is computed in the loop's local frame (axis along z),
        then rotated back to the lab frame.
        """
        rho, zL, xL, yL = self._to_local_cylindrical(x, y, z)

        # Compute field in the loop's local frame (axial symmetry)
        Brho_local, Bz_local = self._field_axisymmetric(rho, zL)

        # Convert local cylindrical (Brho, Bz) back to local Cartesian
        # Avoid division by zero on axis
        safe_rho = np.where(rho > 0, rho, 1.0)
        cos_phi = np.where(rho > 0, xL / safe_rho, 0.0)
        sin_phi = np.where(rho > 0, yL / safe_rho, 0.0)

        BxL = Brho_local * cos_phi
        ByL = Brho_local * sin_phi
        BzL = Bz_local

        # Rotate field back to lab frame
        R = self._rotation_to_loop_frame()
        B_local = np.stack([BxL, ByL, BzL], axis=-1)
        B_lab = B_local @ R.T  # (B @ R.T)[i] = sum_j B[j]*R.T[j,i] = sum_j B[j]*R[i,j] = (R @ B)[i]

        return B_lab[..., 0], B_lab[..., 1], B_lab[..., 2]

    def _field_axisymmetric(self, rho, z):
        """Exact B-field of a z-axis-aligned loop in cylindrical coordinates.

        Returns (B_rho, B_z) in the loop's cylindrical frame.
        Uses the formulation with complete elliptic integrals K and E.
        """
        a = self.radius
        I = self.current

        # Auxiliary quantities
        alpha_sq = a**2 + rho**2 + z**2 - 2 * a * rho
        beta_sq = a**2 + rho**2 + z**2 + 2 * a * rho

        # alpha_sq is the squared distance from each field point to the
        # nearest point on the wire loop.  Warn if any points are so close
        # that the thin-wire model is unphysical and numerics degrade.
        min_dist_sq = np.min(alpha_sq)
        if min_dist_sq < (a * self.NEAR_WIRE_THRESHOLD) ** 2:
            dist = np.sqrt(max(min_dist_sq, 0.0))
            warnings.warn(
                f"One or more field points are within {dist:.2e} m of the "
                f"wire (loop radius {a:.4g} m). The ideal thin-wire model "
                f"diverges at the wire; results this close are not physical.",
                stacklevel=3,
            )

        # Prevent division by zero / numerical issues
        beta_sq = np.maximum(beta_sq, 1e-30)
        alpha_sq = np.maximum(alpha_sq, 1e-30)

        beta = np.sqrt(beta_sq)
        k_sq = 1.0 - alpha_sq / beta_sq
        # Clamp k^2 to [0, 1) for elliptic integrals
        k_sq = np.clip(k_sq, 0.0, 1.0 - 1e-15)

        K = ellipk(k_sq)
        E = ellipe(k_sq)

        prefactor = MU_0 * I / (2.0 * np.pi)

        # On-axis (rho ~ 0): B_rho = 0, use limiting form for B_z
        on_axis = rho < 1e-12

        # General off-axis formulas (Jackson / Wikipedia convention):
        #   Bz   = prefactor / beta * [ K(k^2) + (a^2 - rho^2 - z^2)/alpha^2 * E(k^2) ]
        #   Brho = prefactor * z / (rho * beta) * [ -K(k^2) + (a^2 + rho^2 + z^2)/alpha^2 * E(k^2) ]

        # B_z component
        Bz = np.where(
            on_axis,
            # On-axis formula: Bz = mu0 I a^2 / (2 (a^2+z^2)^(3/2))
            MU_0 * I * a**2 / (2.0 * (a**2 + z**2) ** 1.5),
            prefactor / beta * (K + (a**2 - rho**2 - z**2) / alpha_sq * E),
        )

        # B_rho component
        safe_rho = np.where(on_axis, 1.0, rho)
        Brho = np.where(
            on_axis,
            0.0,
            prefactor * z / (safe_rho * beta) * (-K + (a**2 + rho**2 + z**2) / alpha_sq * E),
        )

        return Brho, Bz

    def _rotation_to_loop_frame(self):
        """Rotation matrix R such that R @ [0,0,1] = self.normal.

        i.e. R rotates from the loop's local frame (axis=z) to the lab frame.
        """
        n = self.normal
        z_hat = np.array([0.0, 0.0, 1.0])

        if np.allclose(n, z_hat):
            return np.eye(3)
        if np.allclose(n, -z_hat):
            return np.diag([1.0, -1.0, -1.0])

        # Rodrigues' rotation
        v = np.cross(z_hat, n)
        s = np.linalg.norm(v)
        c = np.dot(z_hat, n)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2)
        return R

    # ------------------------------------------------------------------
    # Visualization geometry
    # ------------------------------------------------------------------

    def get_path(self, n_points=128):
        theta = np.linspace(0, 2 * np.pi, n_points)
        # Circle in the local frame (xy-plane)
        local = np.column_stack([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            np.zeros_like(theta),
        ])
        # Rotate to lab frame and translate
        R = self._rotation_to_loop_frame()
        return (local @ R.T) + self.center

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            "loop_type": self.loop_type,
            "diameter": self.diameter,
            "center": self.center.tolist(),
            "normal": self.normal.tolist(),
            "current": self.current,
            "frequency": self.frequency,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            diameter=data["diameter"],
            center=data["center"],
            normal=data["normal"],
            current=data["current"],
            frequency=data.get("frequency", 0.0),
            phase=data.get("phase", 0.0),
        )

    def __repr__(self):
        return (
            f"CircularCurrentLoop(diameter={self.diameter}, "
            f"center={self.center.tolist()}, "
            f"normal={self.normal.tolist()}, "
            f"current={self.current})"
        )
