"""Three-axis rectangular Helmholtz coils: three pairs of rounded-rectangle
loops centered at the origin, each pair oriented along a different cardinal
axis (X, Y, Z).

Demonstrates: RoundRectCurrentLoop, 3D orientations, multi-axis coil systems.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source import RoundRectCurrentLoop, Simulation, Visualizer

# Coil dimensions
A = 0.12       # long side (m)
B = 0.08       # short side (m)
R = 0.005      # corner radius (m)
SEP = 0.06     # half-separation between each pair (m)
I = 0.2        # current (A)

sim = Simulation([
    # --- Z-axis pair ---
    RoundRectCurrentLoop(
        side_lengths=(A, B),
        corner_radius=R,
        center=[0, 0, SEP],
        normal=[0, 0, 1],
        orientation=[1, 0, 0],
        current=I,
    ),
    RoundRectCurrentLoop(
        side_lengths=(A, B),
        corner_radius=R,
        center=[0, 0, -SEP],
        normal=[0, 0, 1],
        orientation=[1, 0, 0],
        current=I,
    ),

    # --- X-axis pair ---
    RoundRectCurrentLoop(
        side_lengths=(A, B),
        corner_radius=R,
        center=[SEP, 0, 0],
        normal=[1, 0, 0],
        orientation=[0, 1, 0],
        current=I,
    ),
    RoundRectCurrentLoop(
        side_lengths=(A, B),
        corner_radius=R,
        center=[-SEP, 0, 0],
        normal=[1, 0, 0],
        orientation=[0, 1, 0],
        current=I,
    ),

    # --- Y-axis pair ---
    RoundRectCurrentLoop(
        side_lengths=(A, B),
        corner_radius=R,
        center=[0, SEP, 0],
        normal=[0, 1, 0],
        orientation=[0, 0, 1],
        current=I,
    ),
    RoundRectCurrentLoop(
        side_lengths=(A, B),
        corner_radius=R,
        center=[0, -SEP, 0],
        normal=[0, 1, 0],
        orientation=[0, 0, 1],
        current=I,
    ),
])

Visualizer(sim).show(grid_resolution=8, arrow_size_mode="uniform")
