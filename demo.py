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
vis.show(grid_resolution=8)
