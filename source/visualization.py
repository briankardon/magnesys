"""Visualization of magnetic field simulations using PyVista + Qt."""

import sys

import numpy as np
import pyqtgraph as pg
import pyvista as pv
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QKeySequence, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from . import project
from .path import LineSegmentPath


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

    # Colors for B-field plot curves
    _PLOT_COLORS = {
        "Bx": "#e6261f",
        "By": "#4355db",
        "Bz": "#49da9a",
        "|B|": "#222222",
    }

    N_PATH_SAMPLES = 200

    def __init__(self, simulation):
        self.simulation = simulation
        self._plotter = None
        self._window = None
        self._field_actor = None
        self._project_path = None

        # State tracked for updating
        self._grid_extents = None
        self._grid_resolution = 10
        self._field_scale = "auto"
        self._arrow_size_mode = "linear"
        self._auto_update = True
        self._loop_line_width = 3.0

        # Slice plane state
        self._slice_enabled = False
        self._slice_normal = np.array([0.0, 0.0, 1.0])
        self._slice_origin = np.array([0.0, 0.0, 0.0])
        self._plane_widget = None

        # Sample path state
        self._sample_path = None
        self._sample_line_enabled = False
        self._line_widget = None
        self._plot_widget = None
        self._plot_curves = {}

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
        self._loop_line_width = loop_line_width

        # ---- Build the Qt application and window ----
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        window = QMainWindow()
        window.setWindowTitle("Magnesys")
        window.resize(1400, 900)
        self._window = window

        self._build_menu_bar(window)

        # Central widget: [left: 3D + plot | right: controls]
        central = QWidget()
        window.setCentralWidget(central)
        hlayout = QHBoxLayout(central)
        hlayout.setContentsMargins(0, 0, 0, 0)

        # Left side: vertical splitter with 3D viewport on top, plot below
        splitter = QSplitter(Qt.Orientation.Vertical)

        plotter = QtInteractor(splitter)
        plotter.set_background(background)
        splitter.addWidget(plotter)
        self._plotter = plotter

        self._plot_widget = self._build_plot_widget()
        self._plot_widget.setMinimumHeight(200)
        splitter.addWidget(self._plot_widget)
        self._plot_widget.setVisible(False)  # hidden until sample line enabled

        # Set initial splitter sizes explicitly (pixels)
        splitter.setSizes([660, 240])
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        hlayout.addWidget(splitter, stretch=4)

        # Right side: controls panel
        panel = self._build_control_panel()
        hlayout.addWidget(panel, stretch=0)

        # ---- Populate the 3D scene ----
        if show_loops:
            self._add_loops(plotter, loop_line_width)

        if show_field and self.simulation.loops:
            self._update_field()

        plotter.add_axes()
        plotter.reset_camera()

        window.show()
        app.exec()

    # ------------------------------------------------------------------
    # Qt control panel
    # ------------------------------------------------------------------

    def _build_control_panel(self):
        """Build the right-side Qt control panel."""
        panel = QWidget()
        panel.setFixedWidth(260)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # ---- Loops tree ----
        loops_group = QGroupBox("Loops")
        loops_layout = QVBoxLayout(loops_group)

        self._loops_model = QStandardItemModel()
        self._loops_model.setHorizontalHeaderLabels(["Property", "Value"])

        self._loops_tree = QTreeView()
        self._loops_tree.setModel(self._loops_model)
        self._loops_tree.setEditTriggers(QTreeView.EditTrigger.NoEditTriggers)
        self._loops_tree.setAlternatingRowColors(True)
        self._loops_tree.header().setStretchLastSection(True)
        self._loops_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents,
        )
        loops_layout.addWidget(self._loops_tree)

        layout.addWidget(loops_group)

        self._refresh_loops_tree()

        # ---- Grid resolution ----
        grid_group = QGroupBox("Grid")
        grid_layout = QVBoxLayout(grid_group)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Points:"))
        self._res_slider = QSlider(Qt.Orientation.Horizontal)
        self._res_slider.setRange(3, 35)
        initial_res = (self._grid_resolution if isinstance(self._grid_resolution, int)
                       else self._grid_resolution[0])
        self._res_slider.setValue(initial_res)
        slider_row.addWidget(self._res_slider)
        self._res_label = QLabel(str(initial_res))
        self._res_label.setFixedWidth(24)
        slider_row.addWidget(self._res_label)
        grid_layout.addLayout(slider_row)

        spacing = self._compute_spacing(initial_res)
        self._spacing_label = QLabel(
            f"Spacing: {self._format_spacing(spacing)}"
        )
        self._spacing_label.setStyleSheet("color: grey; font-size: 11px;")
        grid_layout.addWidget(self._spacing_label)

        self._res_slider.valueChanged.connect(self._on_resolution_changed)

        layout.addWidget(grid_group)

        # ---- Options ----
        opts_group = QGroupBox("Options")
        opts_layout = QVBoxLayout(opts_group)

        self._auto_update_cb = QCheckBox("Auto-update")
        self._auto_update_cb.setChecked(self._auto_update)
        self._auto_update_cb.toggled.connect(self._on_auto_update_toggled)
        opts_layout.addWidget(self._auto_update_cb)

        self._slice_cb = QCheckBox("Slice plane")
        self._slice_cb.setChecked(self._slice_enabled)
        self._slice_cb.toggled.connect(self._on_slice_toggled)
        opts_layout.addWidget(self._slice_cb)

        self._sample_line_cb = QCheckBox("Sample line")
        self._sample_line_cb.setChecked(self._sample_line_enabled)
        self._sample_line_cb.toggled.connect(self._on_sample_line_toggled)
        opts_layout.addWidget(self._sample_line_cb)

        layout.addWidget(opts_group)

        # ---- Update button ----
        self._update_btn = QPushButton("Update")
        self._update_btn.clicked.connect(self._on_update_clicked)
        layout.addWidget(self._update_btn)

        layout.addStretch()

        return panel

    def _refresh_loops_tree(self):
        """Rebuild the loops tree view from the current simulation."""
        model = self._loops_model
        model.removeRows(0, model.rowCount())

        for i, loop in enumerate(self.simulation.loops):
            color = self.LOOP_COLORS[i % len(self.LOOP_COLORS)]
            label = f"Loop {i} ({loop.loop_type})"

            root = QStandardItem(label)
            root.setForeground(QColor(color))
            root_val = QStandardItem("")
            root.setEditable(False)
            root_val.setEditable(False)

            # Properties to display
            props = self._loop_properties(loop)
            for key, val in props:
                key_item = QStandardItem(key)
                val_item = QStandardItem(val)
                key_item.setEditable(False)
                val_item.setEditable(False)
                root.appendRow([key_item, val_item])

            model.appendRow([root, root_val])

    @staticmethod
    def _loop_properties(loop):
        """Return a list of (key, value_str) pairs for display in the tree."""
        props = [("Type", loop.loop_type)]

        if hasattr(loop, "diameter"):
            props.append(("Diameter", f"{loop.diameter:.4g} m"))
        if hasattr(loop, "side_lengths"):
            a, b = loop.side_lengths
            props.append(("Side lengths", f"{a:.4g} x {b:.4g} m"))
        if hasattr(loop, "corner_radius"):
            props.append(("Corner radius", f"{loop.corner_radius:.4g} m"))

        props.append(("Center", _format_vec(loop.center)))
        props.append(("Normal", _format_vec(loop.normal)))

        if hasattr(loop, "orientation"):
            props.append(("Orientation", _format_vec(loop.orientation)))

        props.append(("Current", f"{loop.current:.4g} A"))

        return props

    # ------------------------------------------------------------------
    # 2D plot widget
    # ------------------------------------------------------------------

    def _build_plot_widget(self):
        """Build the pyqtgraph plot for B-field along the sample path."""
        pw = pg.PlotWidget()
        pw.setBackground("w")
        pw.setLabel("bottom", "Distance along path", units="m")
        pw.setLabel("left", "B", units="T")
        pw.addLegend(offset=(60, 10))
        pw.showGrid(x=True, y=True, alpha=0.3)

        for name, color in self._PLOT_COLORS.items():
            pen = pg.mkPen(color=color, width=2,
                           style=Qt.PenStyle.DashLine if name == "|B|"
                           else Qt.PenStyle.SolidLine)
            self._plot_curves[name] = pw.plot([], [], pen=pen, name=name)

        return pw

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu_bar(self, window):
        """Build the File menu bar."""
        menu_bar = window.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", window)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_file_open)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", window)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_file_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", window)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._on_file_save_as)
        file_menu.addAction(save_as_action)

    # ------------------------------------------------------------------
    # Project save / load
    # ------------------------------------------------------------------

    def _viz_settings_to_dict(self):
        """Collect current visualization state into a serializable dict."""
        settings = {
            "grid_resolution": self._grid_resolution,
            "field_scale": self._field_scale,
            "arrow_size_mode": self._arrow_size_mode,
            "auto_update": self._auto_update,
            "slice_enabled": self._slice_enabled,
            "slice_normal": self._slice_normal.tolist(),
            "slice_origin": self._slice_origin.tolist(),
            "sample_line_enabled": self._sample_line_enabled,
        }

        if self._plotter is not None:
            cam = self._plotter.camera_position
            settings["camera_position"] = [list(v) for v in cam]

        return settings

    def _apply_viz_settings(self, settings):
        """Apply visualization settings from a dict, falling back to defaults."""
        self._grid_resolution = settings.get("grid_resolution", 10)
        self._field_scale = settings.get("field_scale", "auto")
        self._arrow_size_mode = settings.get("arrow_size_mode", "linear")
        self._auto_update = settings.get("auto_update", True)

        self._slice_normal = np.array(
            settings.get("slice_normal", [0.0, 0.0, 1.0])
        )
        self._slice_origin = np.array(
            settings.get("slice_origin", [0.0, 0.0, 0.0])
        )

        self._res_slider.setValue(
            self._grid_resolution if isinstance(self._grid_resolution, int)
            else self._grid_resolution[0]
        )
        self._auto_update_cb.setChecked(self._auto_update)

        slice_enabled = settings.get("slice_enabled", False)
        self._slice_cb.setChecked(slice_enabled)

        sample_line_enabled = settings.get("sample_line_enabled", False)
        self._sample_line_cb.setChecked(sample_line_enabled)

        cam = settings.get("camera_position")
        if cam is not None and self._plotter is not None:
            self._plotter.camera_position = cam

    def _update_window_title(self):
        """Update the window title to reflect the current file."""
        if self._window is None:
            return
        if self._project_path:
            from pathlib import Path
            name = Path(self._project_path).name
            self._window.setWindowTitle(f"Magnesys \u2014 {name}")
        else:
            self._window.setWindowTitle("Magnesys")

    def _on_file_open(self):
        """File → Open callback."""
        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Open project",
            "",
            f"{project.MAG_FILE_FILTER};;All files (*)",
        )
        if not path:
            return

        simulation, viz_settings, sample_path = project.load(path)
        self._project_path = path

        self.simulation = simulation
        self._sample_path = sample_path

        plotter = self._plotter
        plotter.clear()

        self._field_actor = None
        self._plane_widget = None
        self._slice_enabled = False
        self._line_widget = None
        self._sample_line_enabled = False
        self._add_loops(plotter, self._loop_line_width)
        self._apply_viz_settings(viz_settings)
        self._update_field()

        plotter.add_axes()
        plotter.reset_camera()

        cam = viz_settings.get("camera_position")
        if cam is not None:
            plotter.camera_position = cam

        self._refresh_loops_tree()
        self._update_window_title()

    def _on_file_save(self):
        """File → Save callback."""
        if self._project_path:
            project.save(
                self._project_path,
                self.simulation,
                self._viz_settings_to_dict(),
                sample_path=self._sample_path,
            )
        else:
            self._on_file_save_as()

    def _on_file_save_as(self):
        """File → Save As callback."""
        path, _ = QFileDialog.getSaveFileName(
            self._window,
            "Save project",
            "",
            f"{project.MAG_FILE_FILTER};;All files (*)",
        )
        if not path:
            return

        if not path.lower().endswith(".mag"):
            path += ".mag"

        self._project_path = path
        project.save(
            path, self.simulation, self._viz_settings_to_dict(),
            sample_path=self._sample_path,
        )
        self._update_window_title()

    # ------------------------------------------------------------------
    # Spacing helpers
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

    def _update_spacing_display(self):
        """Update the spacing readout label."""
        spacing = self._compute_spacing(self._grid_resolution)
        self._spacing_label.setText(
            f"Spacing: {self._format_spacing(spacing)}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_resolution_changed(self, value):
        """Callback for the grid resolution slider."""
        self._grid_resolution = value
        self._res_label.setText(str(value))
        self._update_spacing_display()
        if self._auto_update:
            self._update_field()

    def _on_auto_update_toggled(self, checked):
        """Callback for the auto-update checkbox."""
        self._auto_update = checked
        if self._auto_update:
            self._update_field()

    def _on_update_clicked(self):
        """Callback for the manual update button."""
        self._update_field()

    def _on_slice_toggled(self, checked):
        """Callback for the slice plane checkbox."""
        self._slice_enabled = checked
        plotter = self._plotter
        if plotter is None:
            return

        if self._slice_enabled:
            extents = self._grid_extents or self._auto_extents()
            bounds = list(extents)

            if self._plane_widget is None and np.allclose(self._slice_origin, 0.0):
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

            self._plane_widget.SetHandleSize(
                self._plane_widget.GetHandleSize() * 0.5
            )
            self._plane_widget.ScaleEnabledOff()

            self._update_field()
        else:
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

    def _on_sample_line_toggled(self, checked):
        """Callback for the sample line checkbox."""
        self._sample_line_enabled = checked
        plotter = self._plotter
        if plotter is None:
            return

        if self._sample_line_enabled:
            # Default line endpoints: span the center of the bounding box
            extents = self._grid_extents or self._auto_extents()
            if self._sample_path is None:
                cx = (extents[0] + extents[1]) / 2
                cy = (extents[2] + extents[3]) / 2
                cz = (extents[4] + extents[5]) / 2
                span = (extents[1] - extents[0]) * 0.4
                self._sample_path = LineSegmentPath(
                    start=[cx - span, cy, cz],
                    end=[cx + span, cy, cz],
                )

            self._line_widget = plotter.add_line_widget(
                self._on_line_moved,
                bounds=list(extents),
                factor=1.0,
                resolution=1,
                color="black",
                interaction_event="end",
            )
            # Set to current path endpoints
            self._line_widget.SetPoint1(self._sample_path.start.tolist())
            self._line_widget.SetPoint2(self._sample_path.end.tolist())
            self._line_widget.SetHandleSize(
                self._line_widget.GetHandleSize() * 0.5
            )

            self._plot_widget.setVisible(True)
            self._update_plot()
        else:
            if self._line_widget is not None:
                self._line_widget.Off()
                if self._line_widget in plotter.line_widgets:
                    plotter.line_widgets.remove(self._line_widget)
                self._line_widget = None
            self._plot_widget.setVisible(False)

    def _on_line_moved(self, polydata):
        """Callback when the sample line widget is dragged."""
        pts = np.array(polydata.points)
        if len(pts) >= 2:
            self._sample_path = LineSegmentPath(
                start=pts[0], end=pts[-1],
            )
            if self._auto_update:
                self._update_plot()

    # ------------------------------------------------------------------
    # Field update
    # ------------------------------------------------------------------

    def _update_field(self):
        """Recompute and redraw the magnetic field arrows."""
        plotter = self._plotter
        if plotter is None:
            return

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

        # Also refresh the line plot if active
        if self._sample_line_enabled:
            self._update_plot()

    def _update_plot(self):
        """Recompute B along the sample path and update the 2D plot."""
        if not self._plot_curves:
            return
        if self._sample_path is None or not self.simulation.loops:
            for curve in self._plot_curves.values():
                curve.setData([], [])
            return

        n = self.N_PATH_SAMPLES
        points = self._sample_path.get_points(n)
        distances = self._sample_path.get_distances(n)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        Bx, By, Bz = self.simulation.magnetic_field_at(x, y, z)
        Bx = np.asarray(Bx).ravel()
        By = np.asarray(By).ravel()
        Bz = np.asarray(Bz).ravel()
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

        self._plot_curves["Bx"].setData(distances, Bx)
        self._plot_curves["By"].setData(distances, By)
        self._plot_curves["Bz"].setData(distances, Bz)
        self._plot_curves["|B|"].setData(distances, Bmag)

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _add_loops(self, plotter, line_width):
        """Add current loop geometry with current-direction arrowheads."""
        for i, loop in enumerate(self.simulation.loops):
            path = loop.get_path()
            n = len(path)
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

            idx = 0
            tangent = path[(idx + 1) % (n - 1)] - path[idx]
            tangent = tangent / np.linalg.norm(tangent)
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

        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)

        if abs(normal[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        u = np.cross(normal, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

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

        near_wire = self.simulation.near_wire_mask(X, Y, Z).ravel()
        valid = ~near_wire

        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        vectors = np.column_stack([Bx.ravel(), By.ravel(), Bz.ravel()])

        points = points[valid]
        vectors = vectors[valid]
        magnitudes = np.linalg.norm(vectors, axis=1)

        if len(points) == 0:
            return None

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

        all_points = np.vstack([
            loop.get_path() for loop in self.simulation.loops
        ])
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)

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


def _format_vec(v):
    """Format a 3-vector for display."""
    return f"({v[0]:.4g}, {v[1]:.4g}, {v[2]:.4g})"
