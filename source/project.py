"""Project-level save/load for .mag files (simulation + visualization state)."""

import json

from .path import SamplePath
from .simulation import Simulation

MAG_FILE_FILTER = "Magnesys project (*.mag)"
CURRENT_VERSION = 2


def save(path, simulation, viz_settings=None, sample_path=None):
    """Save a simulation and optional visualization settings to a .mag file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file path.
    simulation : Simulation
        The simulation to serialize.
    viz_settings : dict, optional
        Visualization state (grid resolution, camera, slice plane, etc.).
    sample_path : SamplePath, optional
        A sample path to serialize alongside the simulation.
    """
    data = {
        "magnesys_version": CURRENT_VERSION,
        "simulation": simulation.to_dict(),
    }
    if viz_settings is not None:
        data["visualization"] = viz_settings
    if sample_path is not None:
        data["sample_path"] = sample_path.to_dict()

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
    sample_path : SamplePath or None
        The sample path, if one was saved.
    """
    with open(path, "r") as f:
        data = json.load(f)

    simulation = Simulation.from_dict(data)
    viz_settings = data.get("visualization", {})

    sample_path = None
    if "sample_path" in data:
        sample_path = SamplePath.create_from_dict(data["sample_path"])

    return simulation, viz_settings, sample_path
