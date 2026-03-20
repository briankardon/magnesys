"""Field along a sample line: drag the line endpoints in 3D to see
Bx, By, Bz, and |B| plotted in the 2D graph below.

Demonstrates: LineSegmentPath, sample line widget, pyqtgraph 2D plot.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source import CircularCurrentLoop, Simulation, Visualizer

# Helmholtz pair
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
])

# The sample line checkbox in the GUI will enable a draggable line.
# Try dragging the endpoints to explore the field profile!
Visualizer(sim).show(grid_resolution=8, arrow_size_mode="uniform")
