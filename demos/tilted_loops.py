"""Tilted loops: three orthogonal coils for generating an arbitrary
field direction at the center (as in a 3-axis coil system).

Demonstrates: arbitrary loop orientations in 3D, linear arrow mode.
"""

from source import CircularCurrentLoop, Simulation, Visualizer

D = 0.12  # 12 cm diameter

sim = Simulation([
    CircularCurrentLoop(
        diameter=D,
        center=[0, 0, 0],
        normal=[1, 0, 0],
        current=1.0,
    ),
    CircularCurrentLoop(
        diameter=D,
        center=[0, 0, 0],
        normal=[0, 1, 0],
        current=1.0,
    ),
    CircularCurrentLoop(
        diameter=D,
        center=[0, 0, 0],
        normal=[0, 0, 1],
        current=1.0,
    ),
])

Visualizer(sim).show(grid_resolution=8, arrow_size_mode="uniform")
