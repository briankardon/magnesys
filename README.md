# Magnesys

A Python-based magnetic field simulator for computing exact magnetic fields from current loop geometries.

## Features

- **Exact analytical solutions** — uses complete elliptic integrals (not numerical integration) for machine-precision off-axis field calculations
- **Arbitrary orientations** — current loops can be placed and oriented anywhere in 3D space
- **Vectorized computation** — field evaluation works on single points or NumPy meshgrids
- **Extensible geometry** — abstract `CurrentLoop` base class supports adding new loop shapes
- **Human-readable save files** — simulations serialize to JSON

## Supported Geometries

| Type | Class | Parameters |
|------|-------|------------|
| Circular loop | `CircularCurrentLoop` | diameter, center, normal, current |

## Installation

Requires Python 3.10+ with NumPy and SciPy:

```bash
pip install numpy scipy
```

## Quick Start

```python
import numpy as np
from source import CircularCurrentLoop, Simulation

# Create a 10 cm diameter loop carrying 1 A, centered at the origin
loop = CircularCurrentLoop(
    diameter=0.1,
    center=[0, 0, 0],
    normal=[0, 0, 1],
    current=1.0,
)

sim = Simulation([loop])

# Field at a single point
Bx, By, Bz = sim.magnetic_field_at(0, 0, 0.05)

# Field on a grid
X, Y, Z = np.meshgrid(
    np.linspace(-0.1, 0.1, 50),
    np.linspace(-0.1, 0.1, 50),
    [0.0],
)
Bx, By, Bz = sim.magnetic_field_on_grid(X, Y, Z)

# Save and load
sim.save("my_simulation.json")
sim = Simulation.load("my_simulation.json")
```

## Adding New Geometries

Subclass `CurrentLoop` and register it:

```python
from source.current_loop import CurrentLoop

@CurrentLoop.register
class RectangularCurrentLoop(CurrentLoop):
    loop_type = "rectangular"

    def magnetic_field(self, x, y, z):
        ...

    def to_dict(self):
        ...

    @classmethod
    def from_dict(cls, data):
        ...
```
