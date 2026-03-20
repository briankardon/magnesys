"""Helmholtz coil pair: two identical circular loops producing a nearly
uniform field in the region between them.

Demonstrates: CircularCurrentLoop, uniform arrow mode.
"""

from source import CircularCurrentLoop, Simulation, Visualizer

R = 0.05  # radius = 5 cm

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

Visualizer(sim).show(grid_resolution=8, arrow_size_mode="uniform")
