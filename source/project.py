"""Project-level save/load for .mag files (simulation + visualization state)."""

import json

from .path import SamplePath
from .simulation import Simulation
from .trajectory import Trajectory

MAG_FILE_FILTER = "Magnesys project (*.mag)"
CURRENT_VERSION = 3


def save(path, simulation, viz_settings=None, sample_paths=None,
         trajectories=None):
    """Save a simulation and optional visualization settings to a .mag file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file path.
    simulation : Simulation
        The simulation to serialize.
    viz_settings : dict, optional
        Visualization state (grid resolution, camera, slice plane, etc.).
    sample_paths : list of SamplePath, optional
        Sample paths to serialize alongside the simulation.
    trajectories : list of Trajectory, optional
        Trajectories to serialize alongside the simulation.
    """
    data = {
        "magnesys_version": CURRENT_VERSION,
        "simulation": simulation.to_dict(),
    }
    if viz_settings is not None:
        data["visualization"] = viz_settings
    if sample_paths:
        data["sample_paths"] = [p.to_dict() for p in sample_paths]
    if trajectories:
        data["trajectories"] = [t.to_dict() for t in trajectories]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load(path):
    """Load a simulation and visualization settings from a .mag or JSON file.

    Handles v1 (flat loop list), v2 (simulation + single sample_path), and
    v3 (simulation + sample_paths list) formats gracefully.

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
    sample_paths : list of SamplePath
        The sample paths (empty list if none were saved).
    """
    with open(path, "r") as f:
        data = json.load(f)

    simulation = Simulation.from_dict(data)
    viz_settings = data.get("visualization", {})

    sample_paths = []
    if "sample_paths" in data:
        # v3: list of paths
        sample_paths = [
            SamplePath.create_from_dict(d) for d in data["sample_paths"]
        ]
    elif "sample_path" in data:
        # v2: single path → wrap in list
        sample_paths = [SamplePath.create_from_dict(data["sample_path"])]

    trajectories = []
    if "trajectories" in data:
        trajectories = [Trajectory.from_dict(d) for d in data["trajectories"]]

    return simulation, viz_settings, sample_paths, trajectories
