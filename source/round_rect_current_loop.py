"""Rounded-rectangle current loop."""

import numpy as np
from .current_loop import CurrentLoop
from .path_based_loop import PathBasedLoop


@CurrentLoop.register
class RoundRectCurrentLoop(PathBasedLoop):
    """A current loop shaped as a rectangle with rounded corners.

    Parameters
    ----------
    side_lengths : tuple of two floats
        Full outer dimensions (a, b) in meters.  ``a`` is the dimension
        along the ``orientation`` vector; ``b`` is perpendicular to it
        (in the loop plane).
    corner_radius : float
        Radius of the rounded corners in meters.  Must satisfy
        0 <= corner_radius <= min(a, b) / 2.
    center : array_like, shape (3,)
        Center position [x, y, z] in meters.
    normal : array_like, shape (3,)
        Unit vector perpendicular to the loop plane (right-hand rule
        w.r.t. current direction).
    orientation : array_like, shape (3,)
        Unit vector in the loop plane along the ``a``-side.  Will be
        projected to be exactly perpendicular to ``normal``.
    current : float
        Current in amperes.
    """

    loop_type = "round_rect"

    def __init__(self, side_lengths, corner_radius, center, normal,
                 orientation, current, frequency=0.0, phase=0.0):
        a, b = side_lengths
        self.side_lengths = (float(a), float(b))
        self.corner_radius = float(corner_radius)
        self.center = np.asarray(center, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.current = float(current)
        self.frequency = float(frequency)
        self.phase = float(phase)
        orientation = np.asarray(orientation, dtype=float)

        # Validate shapes
        if self.center.shape != (3,):
            raise ValueError("center must have shape (3,)")
        if self.normal.shape != (3,):
            raise ValueError("normal must have shape (3,)")
        if orientation.shape != (3,):
            raise ValueError("orientation must have shape (3,)")

        # Normalize the normal vector
        n_norm = np.linalg.norm(self.normal)
        if n_norm == 0:
            raise ValueError("normal vector must be nonzero")
        self.normal = self.normal / n_norm

        # Project orientation onto the plane perpendicular to normal
        orientation = orientation - np.dot(orientation, self.normal) * self.normal
        o_norm = np.linalg.norm(orientation)
        if o_norm < 1e-10:
            raise ValueError(
                "orientation vector must not be parallel to normal"
            )
        self.orientation = orientation / o_norm

        # Validate dimensions
        if a <= 0 or b <= 0:
            raise ValueError("side_lengths must be positive")
        if self.corner_radius < 0:
            raise ValueError("corner_radius must be non-negative")
        if self.corner_radius > min(a, b) / 2:
            raise ValueError(
                f"corner_radius ({self.corner_radius}) must be <= "
                f"min(a, b) / 2 = {min(a, b) / 2}"
            )

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------

    def get_path(self, n_points=128):
        a, b = self.side_lengths
        r = self.corner_radius

        # Lengths of the straight segments and arcs
        straight_a = a - 2 * r  # along a-side
        straight_b = b - 2 * r  # along b-side
        arc_len = 0.5 * np.pi * r  # quarter circle

        # Total perimeter
        perimeter = 2 * straight_a + 2 * straight_b + 4 * arc_len

        # Build the 8 path pieces (4 straights + 4 arcs) and distribute
        # points proportionally to arc length.
        #
        # Convention: local x along a-side, local y along b-side.
        # Counter-clockwise when viewed from +z (normal direction).
        #
        # Start at bottom-right corner end, traverse CCW:
        #   1. Right straight  : x=+a/2,       y from -b/2+r to +b/2-r
        #   2. Top-right arc   : center (a/2-r, b/2-r)
        #   3. Top straight    : y=+b/2,       x from +a/2-r to -a/2+r
        #   4. Top-left arc    : center (-a/2+r, b/2-r)
        #   5. Left straight   : x=-a/2,       y from +b/2-r to -b/2+r
        #   6. Bottom-left arc : center (-a/2+r, -b/2+r)
        #   7. Bottom straight : y=-b/2,       x from -a/2+r to +a/2-r
        #   8. Bottom-right arc: center (a/2-r, -b/2+r)

        segments = [
            ("straight", straight_b),
            ("arc", arc_len),
            ("straight", straight_a),
            ("arc", arc_len),
            ("straight", straight_b),
            ("arc", arc_len),
            ("straight", straight_a),
            ("arc", arc_len),
        ]

        # Distribute n_points-1 intervals proportionally (last point = first)
        lengths = np.array([s[1] for s in segments])
        # Avoid zero-length issues
        total = lengths.sum()
        if total == 0:
            # Degenerate: all corners meet at a point
            pt = np.zeros((n_points, 3))
            return self._to_lab_frame(pt)

        fracs = lengths / total
        counts = np.round(fracs * (n_points - 1)).astype(int)
        # Adjust rounding so total is exactly n_points - 1
        diff = (n_points - 1) - counts.sum()
        for idx in np.argsort(-fracs):
            if diff == 0:
                break
            counts[idx] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1

        ha = a / 2
        hb = b / 2

        # Corner arc centers
        corners = [
            (ha - r, hb - r, 0.0),        # top-right, arc from 0 to π/2
            (-ha + r, hb - r, np.pi/2),    # top-left, arc from π/2 to π
            (-ha + r, -hb + r, np.pi),     # bottom-left, arc from π to 3π/2
            (ha - r, -hb + r, 3*np.pi/2),  # bottom-right, arc from 3π/2 to 2π
        ]

        points_2d = []

        for seg_idx, (seg_type, seg_len) in enumerate(segments):
            n_pts = counts[seg_idx]
            if n_pts == 0:
                continue
            t = np.linspace(0, 1, n_pts, endpoint=False)

            if seg_type == "straight":
                if seg_idx == 0:
                    # Right side: up
                    xs = np.full_like(t, ha)
                    ys = (-hb + r) + t * straight_b
                elif seg_idx == 2:
                    # Top side: left
                    xs = (ha - r) - t * straight_a
                    ys = np.full_like(t, hb)
                elif seg_idx == 4:
                    # Left side: down
                    xs = np.full_like(t, -ha)
                    ys = (hb - r) - t * straight_b
                else:  # seg_idx == 6
                    # Bottom side: right
                    xs = (-ha + r) + t * straight_a
                    ys = np.full_like(t, -hb)
                points_2d.append(np.column_stack([xs, ys]))

            else:  # arc
                corner_idx = seg_idx // 2  # 0,1,2,3
                cx, cy, start_angle = corners[corner_idx]
                angles = start_angle + t * (np.pi / 2)
                xs = cx + r * np.cos(angles)
                ys = cy + r * np.sin(angles)
                points_2d.append(np.column_stack([xs, ys]))

        pts_2d = np.vstack(points_2d)  # (n_points-1, 2)
        # Close the path
        pts_2d = np.vstack([pts_2d, pts_2d[0:1]])

        # Promote to 3D (z=0 in local frame)
        pts_3d = np.column_stack([pts_2d, np.zeros(len(pts_2d))])

        return self._to_lab_frame(pts_3d)

    def _to_lab_frame(self, local_pts):
        """Transform points from local (x=a-side, y=b-side, z=normal) to lab frame."""
        # Build rotation matrix: columns are the local axes in lab coords
        x_axis = self.orientation
        z_axis = self.normal
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])  # (3, 3)
        return (local_pts @ R.T) + self.center

    def characteristic_size(self):
        return max(self.side_lengths) / 2.0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            "loop_type": self.loop_type,
            "side_lengths": list(self.side_lengths),
            "corner_radius": self.corner_radius,
            "center": self.center.tolist(),
            "normal": self.normal.tolist(),
            "orientation": self.orientation.tolist(),
            "current": self.current,
            "frequency": self.frequency,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            side_lengths=tuple(data["side_lengths"]),
            corner_radius=data["corner_radius"],
            center=data["center"],
            normal=data["normal"],
            orientation=data["orientation"],
            current=data["current"],
            frequency=data.get("frequency", 0.0),
            phase=data.get("phase", 0.0),
        )

    def __repr__(self):
        return (
            f"RoundRectCurrentLoop(side_lengths={self.side_lengths}, "
            f"corner_radius={self.corner_radius}, "
            f"center={self.center.tolist()}, "
            f"normal={self.normal.tolist()}, "
            f"orientation={self.orientation.tolist()}, "
            f"current={self.current})"
        )
