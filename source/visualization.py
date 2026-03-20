"""Visualization of magnetic field simulations using PyVista."""

import numpy as np
import pyvista as pv


class Visualizer:
    """3D visualization for a magnetic field simulation.

    Parameters
    ----------
    simulation : Simulation
        The simulation to visualize.
    """

    # Default colors for cycling through loops
    LOOP_COLORS = [
        "#e6261f",  # red
        "#eb7532",  # orange
        "#f7d038",  # yellow
        "#a3e048",  # green
        "#49da9a",  # teal
        "#34bbe6",  # cyan
        "#4355db",  # blue
        "#d23be7",  # purple
    ]

    def __init__(self, simulation):
        self.simulation = simulation
        self._plotter = None
        self._field_actor = None
        self._scalar_bar_actor = None

        # State tracked for refreshing
        self._grid_extents = None
        self._grid_resolution = 10
        self._field_scale = "auto"
        self._arrow_size_mode = "linear"
        self._auto_update = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(
        self,
        grid_extents=None,
        grid_resolution=10,
        field_scale="auto",
        arrow_size_mode="linear",
        loop_line_width=3.0,
        show_field=True,
        show_loops=True,
        background="white",
    ):
        """Display the simulation in an interactive 3D window.

        Parameters
        ----------
        grid_extents : tuple of 6 floats, optional
            (x_min, x_max, y_min, y_max, z_min, z_max) in meters for the
            field quiver plot. If None, automatically determined from loop
            positions.
        grid_resolution : int or tuple of 3 ints
            Number of grid points along each axis. A single int uses the
            same resolution for all three axes.
        field_scale : float or "auto"
            Scaling factor for quiver arrows. "auto" normalizes so arrows
            are a reasonable size relative to the grid spacing. When
            arrow_size_mode is "uniform", this controls the fixed arrow
            size (default "auto" gives half a grid cell).
        arrow_size_mode : str
            How arrow size encodes field magnitude. Color always encodes
            magnitude regardless of this setting.
            - "linear": arrow length proportional to |B| (default)
            - "uniform": all arrows the same size
            - "log": arrow length proportional to log(|B|), compressing
              the dynamic range so weak and strong fields are both visible
        loop_line_width : float
            Line width for rendering current loops.
        show_field : bool
            Whether to show the magnetic field quiver plot.
        show_loops : bool
            Whether to show the current loop geometry.
        background : str
            Background color for the plot window.
        """
        self._grid_extents = grid_extents
        self._grid_resolution = grid_resolution
        self._field_scale = field_scale
        self._arrow_size_mode = arrow_size_mode

        plotter = pv.Plotter()
        plotter.set_background(background)
        self._plotter = plotter

        if show_loops:
            self._add_loops(plotter, loop_line_width)

        if show_field and self.simulation.loops:
            self._refresh_field()

        self._add_widgets(plotter)

        plotter.add_axes()
        plotter.show()

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    def _add_widgets(self, plotter):
        """Add interactive controls to the plotter."""
        # Grid resolution slider (3 to 25)
        plotter.add_slider_widget(
            self._on_resolution_changed,
            rng=[3, 25],
            value=self._grid_resolution if isinstance(self._grid_resolution, int)
                  else self._grid_resolution[0],
            title="Grid resolution",
            pointa=(0.02, 0.92),
            pointb=(0.28, 0.92),
            style="modern",
            fmt="%.0f",
        )

        # Auto-update checkbox
        plotter.add_checkbox_button_widget(
            self._on_auto_update_toggled,
            value=self._auto_update,
            position=(10, 10),
            size=30,
            color_on="green",
            color_off="grey",
        )
        plotter.add_text(
            "Auto-update",
            position=(45, 10),
            font_size=9,
            color="black",
        )

        # Manual refresh button
        plotter.add_checkbox_button_widget(
            self._on_refresh_clicked,
            value=False,
            position=(160, 10),
            size=30,
            color_on="steelblue",
            color_off="steelblue",
        )
        plotter.add_text(
            "Refresh",
            position=(195, 10),
            font_size=9,
            color="black",
        )

    def _on_resolution_changed(self, value):
        """Callback for the grid resolution slider."""
        new_res = int(round(value))
        if new_res == self._grid_resolution:
            return
        self._grid_resolution = new_res
        if self._auto_update:
            self._refresh_field()

    def _on_auto_update_toggled(self, state):
        """Callback for the auto-update checkbox."""
        self._auto_update = bool(state)

    def _on_refresh_clicked(self, _state):
        """Callback for the manual refresh button."""
        self._refresh_field()

    # ------------------------------------------------------------------
    # Field refresh
    # ------------------------------------------------------------------

    def _refresh_field(self):
        """Recompute and redraw the magnetic field arrows."""
        plotter = self._plotter
        if plotter is None:
            return

        # Remove old field actors and scalar bar
        if self._field_actor is not None:
            plotter.remove_actor(self._field_actor)
            self._field_actor = None
        plotter.scalar_bars.clear()

        if not self.simulation.loops:
            return

        self._field_actor, self._scalar_bar_actor = self._build_field(
            plotter, self._grid_extents, self._grid_resolution,
            self._field_scale, self._arrow_size_mode,
        )

        plotter.render()

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _add_loops(self, plotter, line_width):
        """Add current loop geometry with current-direction arrowheads."""
        for i, loop in enumerate(self.simulation.loops):
            path = loop.get_path()
            n = len(path)
            # Build a polydata line strip
            points = path
            lines = np.column_stack([
                np.full(n - 1, 2),
                np.arange(n - 1),
                np.arange(1, n),
            ])
            poly = pv.PolyData(points, lines=lines)
            color = self.LOOP_COLORS[i % len(self.LOOP_COLORS)]
            plotter.add_mesh(
                poly,
                color=color,
                line_width=line_width,
                render_lines_as_tubes=True,
                label=f"Loop {i}",
            )

            # Arrowhead showing current direction at one point on the loop
            idx = 0
            tangent = path[(idx + 1) % (n - 1)] - path[idx]
            tangent = tangent / np.linalg.norm(tangent)
            # Size the cone relative to the loop's bounding extent
            extent = path.max(axis=0) - path.min(axis=0)
            cone_height = np.max(extent) * 0.08
            cone = pv.Cone(
                center=path[idx] + tangent * cone_height * 0.5,
                direction=tangent,
                height=cone_height,
                radius=cone_height * 0.4,
                resolution=20,
            )
            plotter.add_mesh(cone, color=color)

    def _build_field(self, plotter, grid_extents, grid_resolution,
                     field_scale, arrow_size_mode):
        """Compute field and add arrows to plotter.

        Returns (field_actor, scalar_bar_actor) so they can be removed later.
        """
        extents = grid_extents or self._auto_extents()
        x_min, x_max, y_min, y_max, z_min, z_max = extents

        if isinstance(grid_resolution, int):
            nx = ny = nz = grid_resolution
        else:
            nx, ny, nz = grid_resolution

        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        zs = np.linspace(z_min, z_max, nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        Bx, By, Bz = self.simulation.magnetic_field_on_grid(X, Y, Z)

        # Filter out points too close to any wire (unphysical values)
        near_wire = self.simulation.near_wire_mask(X, Y, Z).ravel()
        valid = ~near_wire

        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        vectors = np.column_stack([Bx.ravel(), By.ravel(), Bz.ravel()])

        points = points[valid]
        vectors = vectors[valid]
        magnitudes = np.linalg.norm(vectors, axis=1)

        if len(points) == 0:
            return None, None

        grid = pv.PolyData(points)

        # Color always shows the true magnitude
        grid["magnitude"] = magnitudes

        # Compute the target cell size for auto-scaling
        spacing = np.array([
            (x_max - x_min) / max(nx - 1, 1),
            (y_max - y_min) / max(ny - 1, 1),
            (z_max - z_min) / max(nz - 1, 1),
        ])
        cell_size = np.mean(spacing[spacing > 0])

        # Resolve auto scale once so we use a consistent value
        resolved_scale = field_scale

        if arrow_size_mode == "uniform":
            # All arrows same size — direction only, magnitude shown by color
            safe_mag = np.where(magnitudes > 0, magnitudes, 1.0)
            unit_vectors = vectors / safe_mag[:, np.newaxis]
            grid["B"] = unit_vectors
            grid["scale"] = np.ones(len(magnitudes))

            if resolved_scale == "auto":
                resolved_scale = 0.4 * cell_size

            arrows = grid.glyph(
                orient="B", scale="scale", factor=resolved_scale,
            )

        elif arrow_size_mode == "log":
            # Arrow length proportional to log(|B|), compressing dynamic range
            nonzero = magnitudes > 0
            if np.any(nonzero):
                min_mag = magnitudes[nonzero].min()
                log_scale = np.where(
                    nonzero,
                    np.log10(magnitudes / min_mag) + 1.0,
                    0.0,
                )
            else:
                log_scale = np.zeros_like(magnitudes)

            safe_mag = np.where(magnitudes > 0, magnitudes, 1.0)
            unit_vectors = vectors / safe_mag[:, np.newaxis]
            grid["B"] = unit_vectors
            grid["scale"] = log_scale

            if resolved_scale == "auto":
                median_log = np.median(log_scale[log_scale > 0]) if np.any(log_scale > 0) else 1.0
                resolved_scale = 0.5 * cell_size / median_log

            arrows = grid.glyph(
                orient="B", scale="scale", factor=resolved_scale,
            )

        else:
            # Linear (default): arrow length proportional to |B|
            grid["B"] = vectors
            grid["scale"] = magnitudes

            if resolved_scale == "auto":
                median_mag = np.median(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 1.0
                resolved_scale = 0.5 * cell_size / median_mag

            arrows = grid.glyph(
                orient="B", scale="scale", factor=resolved_scale,
            )

        field_actor = plotter.add_mesh(
            arrows,
            scalars="magnitude",
            cmap="coolwarm",
            scalar_bar_args={"title": "|B| (T)"},
        )

        return field_actor, None

    def _auto_extents(self):
        """Compute grid extents from loop positions and sizes."""
        if not self.simulation.loops:
            return (-0.1, 0.1, -0.1, 0.1, -0.1, 0.1)

        # Gather bounding info from all loop paths
        all_points = np.vstack([
            loop.get_path() for loop in self.simulation.loops
        ])
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)

        # Pad by 50% of the span (or a minimum absolute padding)
        span = maxs - mins
        padding = np.maximum(span * 0.5, 0.01)
        mins -= padding
        maxs += padding

        return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
