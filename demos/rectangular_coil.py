"""Rectangular coil: a single rounded-rectangle loop showing how
corner radius affects the field shape.

Demonstrates: RoundRectCurrentLoop with different corner radii,
field_scale manual override, save/load.
"""

from source import RoundRectCurrentLoop, Simulation, Visualizer

# Sharp corners
sim = Simulation([
    RoundRectCurrentLoop(
        side_lengths=(0.16, 0.08),
        corner_radius=0.0,
        center=[0, 0, 0],
        normal=[0, 0, 1],
        orientation=[1, 0, 0],
        current=1.0,
    ),
])

# Save and reload to exercise serialization
sim.save("demos/rectangular_coil.json")
sim = Simulation.load("demos/rectangular_coil.json")

Visualizer(sim).show(
    grid_resolution=10,
    arrow_size_mode="uniform",
    grid_extents=(-0.12, 0.12, -0.08, 0.08, -0.06, 0.06),
)
