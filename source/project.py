"""Project-level save/load for .mag files (simulation + visualization state)."""

import json

from .simulation import Simulation

MAG_FILE_FILTER = "Magnesys project (*.mag)"
CURRENT_VERSION = 2


def save(path, simulation, viz_settings=None):
    """Save a simulation and optional visualization settings to a .mag file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file path.
    simulation : Simulation
        The simulation to serialize.
    viz_settings : dict, optional
        Visualization state (grid resolution, camera, slice plane, etc.).
    """
    data = {
        "magnesys_version": CURRENT_VERSION,
        "simulation": simulation.to_dict(),
    }
    if viz_settings is not None:
        data["visualization"] = viz_settings

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load(path):
    """Load a simulation and visualization settings from a .mag or JSON file.

    Handles both v1 (flat loop list) and v2 (simulation + visualization)
    formats gracefully.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a .mag or .json file.

    Returns
    -------
    simulation : Simulation
        The loaded simulation.
    viz_settings : dict
        Visualization settings (empty dict if not present in the file).
    """
    with open(path, "r") as f:
        data = json.load(f)

    simulation = Simulation.from_dict(data)
    viz_settings = data.get("visualization", {})

    return simulation, viz_settings
