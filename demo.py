"""Demo: visualize a Helmholtz coil pair."""

from source import CircularCurrentLoop, Simulation, Visualizer

# Helmholtz coil: two identical coils separated by one radius
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
])

vis = Visualizer(sim)

# Try different arrow size modes:
#   "linear"  - arrow length proportional to |B| (can be dominated by strong fields)
#   "uniform" - all arrows same size, magnitude shown by color only
#   "log"     - arrow length proportional to log(|B|), compresses dynamic range
# Use field_scale to adjust overall arrow size (float, or "auto")
vis.show(grid_resolution=8, arrow_size_mode="uniform")
