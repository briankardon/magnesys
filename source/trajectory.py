"""Trajectory: a sequence of 3D points for visualization (e.g. reconstructed paths)."""

import numpy as np


class Trajectory:
    """A static 3D trajectory for visualization.

    Unlike sample paths, trajectories are not editable and have no handles.
    They are simply polylines rendered in the 3D view, useful for comparing
    ground-truth vs. reconstructed position traces.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Ordered positions [x, y, z] in meters.
    label : str
        Display label (e.g. "Ground truth", "Reconstructed").
    color : str
        Color for rendering (hex or named color).
    """

    def __init__(self, points, label="Trajectory", color="#ff6600"):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        self.label = str(label)
        self.color = str(color)

    @property
    def length(self):
        """Total arc length of the trajectory."""
        if len(self.points) < 2:
            return 0.0
        seg_lens = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        return float(np.nansum(seg_lens))

    def to_dict(self):
        return {
            "points": self.points.tolist(),
            "label": self.label,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            points=data["points"],
            label=data.get("label", "Trajectory"),
            color=data.get("color", "#ff6600"),
        )

    def __repr__(self):
        return f"Trajectory({len(self.points)} points, label={self.label!r})"
