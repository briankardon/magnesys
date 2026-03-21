"""Launch the Magnesys GUI.

Usage:
    python magnesys.py              # open with empty simulation
    python magnesys.py project.mag  # open and load a .mag file
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from source import Simulation, Visualizer, project


def main():
    parser = argparse.ArgumentParser(description="Magnesys magnetic field simulator")
    parser.add_argument("file", nargs="?", help="Path to a .mag file to load")
    args = parser.parse_args()

    if args.file:
        sim, viz_settings, sample_paths, trajectories = project.load(args.file)
        vis = Visualizer(sim)
        vis._project_path = args.file
        vis._sample_paths = sample_paths
        vis._trajectories = trajectories
        vis.show(**_viz_show_kwargs(viz_settings))
    else:
        vis = Visualizer(Simulation())
        vis.show()


def _viz_show_kwargs(viz_settings):
    """Extract show() keyword arguments from viz_settings dict."""
    kwargs = {}
    if "grid_resolution" in viz_settings:
        kwargs["grid_resolution"] = viz_settings["grid_resolution"]
    if "field_scale" in viz_settings:
        kwargs["field_scale"] = viz_settings["field_scale"]
    if "arrow_size_mode" in viz_settings:
        kwargs["arrow_size_mode"] = viz_settings["arrow_size_mode"]
    return kwargs


if __name__ == "__main__":
    main()
