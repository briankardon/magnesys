"""Mixed loop geometries: a circular Helmholtz pair with a rectangular
coil in the midplane.

Demonstrates: combining CircularCurrentLoop and RoundRectCurrentLoop
in a single simulation.
"""

from source import (
    CircularCurrentLoop, RoundRectCurrentLoop, Simulation, Visualizer,
)

R = 0.05

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

Visualizer(sim).show(grid_resolution=8, arrow_size_mode="uniform")
