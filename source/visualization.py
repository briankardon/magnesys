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

    # Panel layout constants (normalized viewport coordinates)
    _PANEL_LEFT = 0.78   # left edge of the control panel region
    _PANEL_RIGHT = 0.98

    def __init__(self, simulation):
        self.simulation = simulation
        self._plotter = None
        self._field_actor = None

        # State tracked for updating
        self._grid_extents = None
        self._grid_resolution = 10
        self._field_scale = "auto"
        self._arrow_size_mode = "linear"
        self._auto_update = True

        # Slice plane state
        self._slice_enabled = False
        self._slice_normal = np.array([0.0, 0.0, 1.0])
        self._slice_origin = np.array([0.0, 0.0, 0.0])
        self._plane_widget = None

        # Widget references for dynamic updates
        self._slider_widget = None
        self._spacing_text_actor = None
        self._update_widget = None

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

        plotter = pv.Plotter(window_size=(1400, 900))
        plotter.set_background(background)
        self._plotter = plotter

        if show_loops:
            self._add_loops(plotter, loop_line_width)

        if show_field and self.simulation.loops:
            self._update_field()

        self._add_widgets(plotter)

        plotter.add_axes()
        plotter.show()

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    def _compute_spacing(self, resolution):
        """Compute the average grid spacing for a given resolution."""
        extents = self._grid_extents or self._auto_extents()
        x_min, x_max, y_min, y_max, z_min, z_max = extents
        n = resolution if isinstance(resolution, int) else resolution[0]
        spacings = []
        for lo, hi in [(x_min, x_max), (y_min, y_max), (z_min, z_max)]:
            span = hi - lo
            if span > 0 and n > 1:
                spacings.append(span / (n - 1))
        return np.mean(spacings) if spacings else 0.0

    def _format_spacing(self, spacing_m):
        """Format a spacing value with appropriate units."""
        if spacing_m >= 0.01:
            return f"{spacing_m * 100:.1f} cm"
        else:
            return f"{spacing_m * 1000:.2f} mm"

    def _add_widgets(self, plotter):
        """Add interactive controls in a right-side panel."""
        L = self._PANEL_LEFT
        R = self._PANEL_RIGHT

        # ---- Panel title ----
        plotter.add_text(
            "Controls",
            position=(L + 0.01, 0.94),
            viewport=True,
            font_size=11,
            color="black",
            font="courier",
        )

        # ---- Grid spacing slider ----
        initial_res = (self._grid_resolution if isinstance(self._grid_resolution, int)
                       else self._grid_resolution[0])
        self._slider_widget = plotter.add_slider_widget(
            self._on_resolution_changed,
            rng=[3, 25],
            value=initial_res,
            title="Grid points per axis",
            pointa=(L, 0.85),
            pointb=(R, 0.85),
            style="modern",
            fmt="%.0f",
        )

        # Spacing readout below slider
        spacing = self._compute_spacing(initial_res)
        self._spacing_text_actor = plotter.add_text(
            f"Spacing: {self._format_spacing(spacing)}",
            position=(L + 0.01, 0.76),
            viewport=True,
            font_size=9,
            color="gray",
        )

        # ---- Checkboxes and buttons ----
        win_w, win_h = plotter.window_size
        cb_x = int(L * win_w) + 5
        cb_size = 25

        # Auto-update checkbox
        cb_y = int(0.66 * win_h)
        plotter.add_checkbox_button_widget(
            self._on_auto_update_toggled,
            value=self._auto_update,
            position=(cb_x, cb_y),
            size=cb_size,
            color_on="green",
            color_off="grey",
        )
        plotter.add_text(
            "Auto-update",
            position=(cb_x + cb_size + 8, cb_y + 2),
            font_size=9,
            color="black",
        )

        # Update button
        rb_y = int(0.58 * win_h)
        self._update_widget = plotter.add_checkbox_button_widget(
            self._on_update_clicked,
            value=False,
            position=(cb_x, rb_y),
            size=cb_size,
            color_on="steelblue",
            color_off="steelblue",
        )
        plotter.add_text(
            "Update",
            position=(cb_x + cb_size + 8, rb_y + 2),
            font_size=9,
            color="black",
        )

        # Slice plane checkbox
        sp_y = int(0.50 * win_h)
        plotter.add_checkbox_button_widget(
            self._on_slice_toggled,
            value=self._slice_enabled,
            position=(cb_x, sp_y),
            size=cb_size,
            color_on="orange",
            color_off="grey",
        )
        plotter.add_text(
            "Slice plane",
            position=(cb_x + cb_size + 8, sp_y + 2),
            font_size=9,
            color="black",
        )

    def _update_spacing_display(self):
        """Update the spacing readout text."""
        if self._spacing_text_actor is None:
            return
        spacing = self._compute_spacing(self._grid_resolution)
        self._spacing_text_actor.input = (
            f"Spacing: {self._format_spacing(spacing)}"
        )

    def _on_resolution_changed(self, value):
        """Callback for the grid resolution slider."""
        new_res = int(round(value))
        if new_res == self._grid_resolution:
            return
        self._grid_resolution = new_res
        self._update_spacing_display()
        if self._auto_update:
            self._update_field()

    def _on_auto_update_toggled(self, state):
        """Callback for the auto-update checkbox."""
        self._auto_update = bool(state)
        if self._auto_update:
            self._update_field()

    def _on_update_clicked(self, _state):
        """Callback for the manual update button."""
        self._update_field()
        # Reset to "unpressed" appearance
        if self._update_widget is not None:
            self._update_widget.GetRepresentation().SetState(0)

    def _on_slice_toggled(self, state):
        """Callback for the slice plane checkbox."""
        self._slice_enabled = bool(state)
        plotter = self._plotter
        if plotter is None:
            return

        if self._slice_enabled:
            # Create the plane widget
            extents = self._grid_extents or self._auto_extents()
            bounds = list(extents)
            center = [
                (extents[0] + extents[1]) / 2,
                (extents[2] + extents[3]) / 2,
                (extents[4] + extents[5]) / 2,
            ]
            self._slice_origin = np.array(center)
            self._slice_normal = np.array([0.0, 0.0, 1.0])

            self._plane_widget = plotter.add_plane_widget(
                self._on_plane_moved,
                normal=self._slice_normal,
                origin=self._slice_origin,
                bounds=bounds,
                factor=1.0,
                color="orange",
                implicit=True,
                normal_rotation=True,
                origin_translation=True,
                outline_translation=True,
                interaction_event="end",
                test_callback=False,
            )

            # Shrink the plane widget handles
            self._plane_widget.SetHandleSize(
                self._plane_widget.GetHandleSize() * 0.5
            )

            self._update_field()
        else:
            # Remove the plane widget
            if self._plane_widget is not None:
                self._plane_widget.Off()
                if self._plane_widget in plotter.plane_widgets:
                    plotter.plane_widgets.remove(self._plane_widget)
                self._plane_widget = None
            self._update_field()

    def _on_plane_moved(self, normal, origin):
        """Callback when the slice plane is dragged."""
        self._slice_normal = np.array(normal)
        self._slice_origin = np.array(origin)
        if self._auto_update:
            self._update_field()

    # ------------------------------------------------------------------
    # Field update
    # ------------------------------------------------------------------

    def _update_field(self):
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

        if self._slice_enabled:
            self._field_actor = self._build_field_slice(
                plotter, self._grid_extents, self._grid_resolution,
                self._field_scale, self._arrow_size_mode,
                self._slice_normal, self._slice_origin,
            )
        else:
            self._field_actor = self._build_field(
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

    def _make_plane_grid(self, normal, origin, extents, resolution):
        """Create a 2D grid of points lying in the given plane.

        The grid spans the intersection of the plane with the bounding box
        defined by extents.

        Returns
        -------
        points : ndarray, shape (M, 3)
            Grid point positions in 3D.
        cell_size : float
            Average grid cell spacing.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = extents
        n = resolution if isinstance(resolution, int) else resolution[0]

        # Build an orthonormal basis for the plane
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)

        # Choose a vector not parallel to normal for cross product
        if abs(normal[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        u = np.cross(normal, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        # Determine the grid extent: project the bounding box corners
        # onto the u and v axes
        corners = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ])

        origin = np.asarray(origin, dtype=float)
        relative = corners - origin
        u_proj = relative @ u
        v_proj = relative @ v

        u_min, u_max = u_proj.min(), u_proj.max()
        v_min, v_max = v_proj.min(), v_proj.max()

        us = np.linspace(u_min, u_max, n)
        vs = np.linspace(v_min, v_max, n)
        U, V = np.meshgrid(us, vs)

        # Convert back to 3D
        points = (origin
                  + U.ravel()[:, np.newaxis] * u
                  + V.ravel()[:, np.newaxis] * v)

        spacing_u = (u_max - u_min) / max(n - 1, 1)
        spacing_v = (v_max - v_min) / max(n - 1, 1)
        cell_size = (spacing_u + spacing_v) / 2.0

        return points, cell_size

    def _build_field_slice(self, plotter, grid_extents, grid_resolution,
                           field_scale, arrow_size_mode, normal, origin):
        """Compute field on a 2D slice plane and add arrows.  Returns field_actor."""
        extents = grid_extents or self._auto_extents()

        points, cell_size = self._make_plane_grid(
            normal, origin, extents, grid_resolution,
        )

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        Bx, By, Bz = self.simulation.magnetic_field_at(x, y, z)

        # Filter near-wire points
        near_wire = self.simulation.near_wire_mask(x, y, z)
        valid = ~near_wire

        vectors = np.column_stack([
            np.asarray(Bx).ravel(),
            np.asarray(By).ravel(),
            np.asarray(Bz).ravel(),
        ])
        points = points[valid]
        vectors = vectors[valid]
        magnitudes = np.linalg.norm(vectors, axis=1)

        if len(points) == 0:
            return None

        return self._arrows_from_points(
            plotter, points, vectors, magnitudes, cell_size,
            field_scale, arrow_size_mode,
        )

    def _build_field(self, plotter, grid_extents, grid_resolution,
                     field_scale, arrow_size_mode):
        """Compute field on 3D grid and add arrows.  Returns field_actor."""
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
            return None

        # Compute the target cell size for auto-scaling
        spacing = np.array([
            (x_max - x_min) / max(nx - 1, 1),
            (y_max - y_min) / max(ny - 1, 1),
            (z_max - z_min) / max(nz - 1, 1),
        ])
        cell_size = np.mean(spacing[spacing > 0])

        return self._arrows_from_points(
            plotter, points, vectors, magnitudes, cell_size,
            field_scale, arrow_size_mode,
        )

    def _arrows_from_points(self, plotter, points, vectors, magnitudes,
                            cell_size, field_scale, arrow_size_mode):
        """Build glyph arrows from points/vectors and add to plotter.

        Shared by both 3D grid and 2D slice rendering paths.
        Returns the field actor.
        """
        grid = pv.PolyData(points)
        grid["magnitude"] = magnitudes

        resolved_scale = field_scale

        if arrow_size_mode == "uniform":
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

        return field_actor

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

        # Force a cube so the 3D grid isn't stretched along one axis
        center = (mins + maxs) / 2.0
        half_side = np.max(maxs - mins) / 2.0
        mins = center - half_side
        maxs = center + half_side

        return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
