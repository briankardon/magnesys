"""Anti-Helmholtz coil: two loops with opposing currents, producing a
field gradient (zero at center) used in magneto-optical traps.

Demonstrates: opposing current directions, log arrow mode to reveal
the weak-field region at center alongside the stronger field near coils.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source import CircularCurrentLoop, Simulation, Visualizer

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
        normal=[0, 0, -1],  # reversed current
        current=1.0,
    ),
])

Visualizer(sim).show(grid_resolution=8, arrow_size_mode="log")
