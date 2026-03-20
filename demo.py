"""Demo: visualize a mixed coil simulation."""

from source import (
    CircularCurrentLoop, RoundRectCurrentLoop, Simulation, Visualizer,
)

# A circular Helmholtz pair with a rectangular coil in between
R = 0.05  # 10 cm diameter
sim = Simulation([
    CircularCurrentLoop(
        diameter=2 * R,
        center=[0, 0, R / 2],
        normal=[0, 0, 1],
        current=1.0,
    ),
    CircularCurrentLoop(
        diameter=2 * R,
        center=[0, 0, -R / 2],
        normal=[0, 0, 1],
        current=1.0,
    ),
    RoundRectCurrentLoop(
        side_lengths=(0.14, 0.08),
        corner_radius=0.01,
        center=[0, 0, 0],
        normal=[0, 0, 1],
        orientation=[1, 0, 0],
        current=0.5,
    ),
])

vis = Visualizer(sim)

# Try different arrow size modes:
#   "linear"  - arrow length proportional to |B| (can be dominated by strong fields)
#   "uniform" - all arrows same size, magnitude shown by color only
#   "log"     - arrow length proportional to log(|B|), compresses dynamic range
# Use field_scale to adjust overall arrow size (float, or "auto")
vis.show(grid_resolution=8, arrow_size_mode="uniform")
