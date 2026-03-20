"""Visualization of magnetic field simulations using PyVista + Qt."""

import sys

import numpy as np
import pyqtgraph as pg
import pyvista as pv
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QAction, QColor, QKeySequence, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from . import project
from .path import LineSegmentPath, PolylinePath, SplinePath


class ExportFieldAlongPathDialog(QDialog):
    """Dialog for configuring field-along-path CSV export."""

    def __init__(self, path_length, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export field along path")
        self._path_length = path_length

        layout = QVBoxLayout(self)

        # Sampling interval
        form = QFormLayout()

        self._interval_spin = QDoubleSpinBox()
        self._interval_spin.setDecimals(4)
        self._interval_spin.setSuffix(" m")
        self._interval_spin.setRange(1e-6, path_length)
        # Default: ~200 points
        default_interval = path_length / 200 if path_length > 0 else 0.001
        self._interval_spin.setValue(default_interval)
        self._interval_spin.setSingleStep(default_interval * 0.1)
        form.addRow("Sampling interval:", self._interval_spin)

        # Point count readout
        self._count_label = QLabel()
        form.addRow("Points:", self._count_label)

        # Path length readout
        form.addRow("Path length:", QLabel(f"{path_length:.6g} m"))

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
        )
        self._export_btn = buttons.addButton(
            "Export...", QDialogButtonBox.ButtonRole.AcceptRole,
        )
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        # AcceptRole doesn't auto-connect to accept(), wire it manually
        self._export_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)

        # Update count on interval change
        self._interval_spin.valueChanged.connect(self._update_count)
        self._update_count()

    def _update_count(self):
        n = self.point_count()
        self._count_label.setText(str(n))

    def interval(self):
        return self._interval_spin.value()

    def point_count(self):
        if self._path_length <= 0:
            return 0
        return max(int(self._path_length / self._interval_spin.value()) + 1, 2)


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
        self._loop_actors = []  # actors for loop geometry (lines + cones)
        self._project_path = None

        # State tracked for updating
        self._grid_extents = None
        self._grid_resolution = 10
        self._field_scale = "auto"
        self._arrow_size_mode = "log"
        self._auto_update = True
        self._loop_line_width = 3.0
        self._time = 0.0  # current time in seconds
        self._auto_scale = True  # auto-scale 3D color bar, arrows, and 2D y-axis
        self._locked_clim = None  # (vmin, vmax) when auto-scale is off
        self._locked_yrange = None  # (ymin, ymax) when auto-scale is off
        self._locked_arrow_scale = None  # resolved glyph factor when locked
        self._locked_log_min_mag = None  # log-mode reference floor when locked

        # Slice plane state
        self._slice_enabled = False
        self._slice_normal = np.array([0.0, 0.0, 1.0])
        self._slice_origin = np.array([0.0, 0.0, 0.0])
        self._plane_widget = None

        # Sample paths state (multi-path)
        self._sample_paths = []
        self._sample_paths_visible = False
        self._path_visuals = []  # parallel to _sample_paths; dicts or None
        self._selected_path_index = 0
        self._path_selector = None  # QComboBox reference
        self._plot_mode = "position"  # "position" or "time"
        self._plot_widget = None
        self._plot_curves = {}
        self._time_cursor = None  # vertical line on time-mode plot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(
        self,
        grid_extents=None,
        grid_resolution=10,
        field_scale="auto",
        arrow_size_mode="uniform",
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

        self._plot_container = self._build_plot_container()
        self._plot_container.setMinimumHeight(200)
        splitter.addWidget(self._plot_container)
        self._plot_container.setVisible(False)  # hidden until sample paths enabled

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
        panel.setFixedWidth(390)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # ---- Objects tree ----
        objects_group = QGroupBox("Objects")
        objects_layout = QVBoxLayout(objects_group)

        self._tree_model = QStandardItemModel()
        self._tree_model.setHorizontalHeaderLabels(["Property", "Value"])

        self._tree_view = QTreeView()
        self._tree_view.setModel(self._tree_model)
        self._tree_view.setEditTriggers(QTreeView.EditTrigger.DoubleClicked)
        self._tree_view.setAlternatingRowColors(True)
        self._tree_view.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu,
        )
        self._tree_view.customContextMenuRequested.connect(
            self._on_tree_context_menu,
        )
        self._tree_view.header().setStretchLastSection(True)
        self._tree_view.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents,
        )
        objects_layout.addWidget(self._tree_view)

        layout.addWidget(objects_group)

        self._refresh_tree()

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

        # ---- Time ----
        time_group = QGroupBox("Time")
        time_layout = QVBoxLayout(time_group)

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("t (s):"))
        self._time_spin = QDoubleSpinBox()
        self._time_spin.setDecimals(6)
        self._time_spin.setRange(-1e6, 1e6)
        self._time_spin.setValue(self._time)
        self._time_spin.setSingleStep(0.001)
        self._time_spin.valueChanged.connect(self._on_time_changed)
        time_row.addWidget(self._time_spin)
        time_layout.addLayout(time_row)

        self._time_slider = QSlider(Qt.Orientation.Horizontal)
        self._time_slider.setRange(0, 1000)
        self._time_slider.setValue(0)
        self._time_slider.valueChanged.connect(self._on_time_slider_moved)
        time_layout.addWidget(self._time_slider)

        self._time_range_label = QLabel("")
        self._time_range_label.setStyleSheet("color: grey; font-size: 11px;")
        time_layout.addWidget(self._time_range_label)
        self._update_time_range()

        layout.addWidget(time_group)

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

        self._sample_paths_cb = QCheckBox("Show sample paths")
        self._sample_paths_cb.setChecked(self._sample_paths_visible)
        self._sample_paths_cb.toggled.connect(self._on_sample_paths_toggled)
        opts_layout.addWidget(self._sample_paths_cb)

        self._auto_scale_cb = QCheckBox("Auto-scale")
        self._auto_scale_cb.setChecked(self._auto_scale)
        self._auto_scale_cb.toggled.connect(self._on_auto_scale_toggled)
        opts_layout.addWidget(self._auto_scale_cb)

        layout.addWidget(opts_group)

        # ---- Arrow scaling ----
        arrow_group = QGroupBox("Arrow scaling")
        arrow_layout = QVBoxLayout(arrow_group)

        self._arrow_radios = {}
        for mode, label in [
            ("uniform", "Uniform (color only)"),
            ("log", "Logarithmic"),
            ("linear", "Linear"),
        ]:
            rb = QRadioButton(label)
            rb.setChecked(mode == self._arrow_size_mode)
            rb.toggled.connect(lambda checked, m=mode: self._on_arrow_mode_changed(m, checked))
            arrow_layout.addWidget(rb)
            self._arrow_radios[mode] = rb

        layout.addWidget(arrow_group)

        # ---- Update button ----
        self._update_btn = QPushButton("Update")
        self._update_btn.clicked.connect(self._on_update_clicked)
        layout.addWidget(self._update_btn)

        layout.addStretch()

        return panel

    def _refresh_tree(self):
        """Rebuild the objects tree view from the current simulation and paths."""
        model = self._tree_model
        tree = self._tree_view

        # Disconnect while rebuilding to avoid spurious change signals
        try:
            model.itemChanged.disconnect(self._on_tree_item_changed)
        except TypeError:
            pass

        # Save expanded state keyed by ("loop", i) or ("path", j) and prop name
        expanded = set()
        for row in range(model.rowCount()):
            root_idx = model.index(row, 0)
            root_item = model.itemFromIndex(root_idx)
            meta = root_item.data(Qt.ItemDataRole.UserRole + 1)
            if meta is None:
                continue
            key = (meta["kind"], meta["index"])
            if tree.isExpanded(root_idx):
                expanded.add(key)
                for child_row in range(root_item.rowCount()):
                    child_idx = model.index(child_row, 0, root_idx)
                    if tree.isExpanded(child_idx):
                        child_name = model.itemFromIndex(child_idx).text()
                        expanded.add((*key, child_name))

        model.removeRows(0, model.rowCount())

        _ROLE = Qt.ItemDataRole.UserRole
        _KEY_ROLE = Qt.ItemDataRole.UserRole + 1  # for expansion state

        # ---- Loop rows ----
        for i, loop in enumerate(self.simulation.loops):
            color = self.LOOP_COLORS[i % len(self.LOOP_COLORS)]
            label = f"Loop {i} ({loop.loop_type})"

            root = QStandardItem(label)
            root.setForeground(QColor(color))
            root_val = QStandardItem("")
            root.setEditable(False)
            root_val.setEditable(False)
            root.setData({"kind": "loop", "index": i}, _KEY_ROLE)

            # Type (read-only)
            key_item = QStandardItem("Type")
            val_item = QStandardItem(loop.loop_type)
            key_item.setEditable(False)
            val_item.setEditable(False)
            root.appendRow([key_item, val_item])

            # Scalar properties
            scalars = self._loop_scalar_props(loop)
            for attr, display_name, value in scalars:
                key_item = QStandardItem(display_name)
                val_item = QStandardItem(f"{value:.6g}")
                key_item.setEditable(False)
                val_item.setEditable(True)
                val_item.setData({"kind": "loop", "loop": i, "attr": attr}, _ROLE)
                root.appendRow([key_item, val_item])

            # Vector properties
            vectors = self._loop_vector_props(loop)
            for attr, display_name, vec in vectors:
                vec_key = QStandardItem(display_name)
                vec_val = QStandardItem(_format_vec(vec))
                vec_key.setEditable(False)
                vec_val.setEditable(False)
                for ci, comp in enumerate("xyz"):
                    comp_key = QStandardItem(comp)
                    comp_val = QStandardItem(f"{vec[ci]:.6g}")
                    comp_key.setEditable(False)
                    comp_val.setEditable(True)
                    comp_val.setData(
                        {"kind": "loop", "loop": i, "attr": attr, "component": ci}, _ROLE,
                    )
                    vec_key.appendRow([comp_key, comp_val])
                root.appendRow([vec_key, vec_val])

            model.appendRow([root, root_val])

        # ---- Path rows ----
        for j, sp in enumerate(self._sample_paths):
            label = f"Path {j} ({sp.path_type})"

            root = QStandardItem(label)
            root_val = QStandardItem("")
            root.setEditable(False)
            root_val.setEditable(False)
            root.setData({"kind": "path", "index": j}, _KEY_ROLE)

            # Type (read-only)
            key_item = QStandardItem("Type")
            val_item = QStandardItem(sp.path_type)
            key_item.setEditable(False)
            val_item.setEditable(False)
            root.appendRow([key_item, val_item])

            # Path-specific properties
            if isinstance(sp, (PolylinePath, SplinePath)):
                for k, pt in enumerate(sp.points):
                    pt_key = QStandardItem(f"Point {k} (m)")
                    pt_val = QStandardItem(_format_vec(pt))
                    pt_key.setEditable(False)
                    pt_val.setEditable(False)
                    pt_key.setData(
                        {"kind": "path_point", "path": j, "point_index": k}, _ROLE,
                    )
                    for ci, comp in enumerate("xyz"):
                        comp_key = QStandardItem(comp)
                        comp_val = QStandardItem(f"{pt[ci]:.6g}")
                        comp_key.setEditable(False)
                        comp_val.setEditable(True)
                        comp_val.setData(
                            {"kind": "path", "path": j, "attr": "points",
                             "point_index": k, "component": ci}, _ROLE,
                        )
                        pt_key.appendRow([comp_key, comp_val])
                    root.appendRow([pt_key, pt_val])
            elif hasattr(sp, "start") and hasattr(sp, "end"):
                for attr, display_name in [("start", "Start (m)"), ("end", "End (m)")]:
                    vec = getattr(sp, attr)
                    vec_key = QStandardItem(display_name)
                    vec_val = QStandardItem(_format_vec(vec))
                    vec_key.setEditable(False)
                    vec_val.setEditable(False)
                    for ci, comp in enumerate("xyz"):
                        comp_key = QStandardItem(comp)
                        comp_val = QStandardItem(f"{vec[ci]:.6g}")
                        comp_key.setEditable(False)
                        comp_val.setEditable(True)
                        comp_val.setData(
                            {"kind": "path", "path": j, "attr": attr, "component": ci}, _ROLE,
                        )
                        vec_key.appendRow([comp_key, comp_val])
                    root.appendRow([vec_key, vec_val])

            model.appendRow([root, root_val])

        # Restore expanded state
        for row in range(model.rowCount()):
            root_idx = model.index(row, 0)
            root_item = model.itemFromIndex(root_idx)
            meta = root_item.data(_KEY_ROLE)
            if meta is None:
                continue
            key = (meta["kind"], meta["index"])
            if key in expanded:
                tree.setExpanded(root_idx, True)
                for child_row in range(root_item.rowCount()):
                    child_idx = model.index(child_row, 0, root_idx)
                    child_name = model.itemFromIndex(child_idx).text()
                    if (*key, child_name) in expanded:
                        tree.setExpanded(child_idx, True)

        model.itemChanged.connect(self._on_tree_item_changed)

    @staticmethod
    def _loop_scalar_props(loop):
        """Return [(attr_name, display_name, value), ...] for scalar properties."""
        props = []
        if hasattr(loop, "diameter"):
            props.append(("diameter", "Diameter (m)", loop.diameter))
        if hasattr(loop, "side_lengths"):
            a, b = loop.side_lengths
            props.append(("side_length_a", "Side A (m)", a))
            props.append(("side_length_b", "Side B (m)", b))
        if hasattr(loop, "corner_radius"):
            props.append(("corner_radius", "Corner radius (m)", loop.corner_radius))
        props.append(("current", "Current (A)", loop.current))
        props.append(("frequency", "Frequency (Hz)", getattr(loop, "frequency", 0.0)))
        props.append(("phase", "Phase (rad)", getattr(loop, "phase", 0.0)))
        return props

    @staticmethod
    def _loop_vector_props(loop):
        """Return [(attr_name, display_name, ndarray), ...] for vector properties."""
        props = [
            ("center", "Center (m)", loop.center),
            ("normal", "Normal", loop.normal),
        ]
        if hasattr(loop, "orientation"):
            props.append(("orientation", "Orientation", loop.orientation))
        return props

    def _on_tree_item_changed(self, item):
        """Handle an edit in the objects tree view."""
        meta = item.data(Qt.ItemDataRole.UserRole)
        if meta is None:
            return

        text = item.text().strip()
        try:
            value = float(text)
        except ValueError:
            self._refresh_tree()
            return

        kind = meta.get("kind")
        if kind == "loop":
            self._on_loop_property_edited(meta, value)
        elif kind == "path":
            self._on_path_property_edited(meta, value)

    def _on_loop_property_edited(self, meta, value):
        """Apply a loop property edit from the tree."""
        loop_idx = meta["loop"]
        attr = meta["attr"]

        if loop_idx >= len(self.simulation.loops):
            return
        loop = self.simulation.loops[loop_idx]

        if "component" in meta:
            ci = meta["component"]
            vec = getattr(loop, attr).copy()
            vec[ci] = value
            setattr(loop, attr, vec)
            if attr in ("normal", "orientation"):
                norm = np.linalg.norm(vec)
                if norm > 0:
                    setattr(loop, attr, vec / norm)
        elif attr == "side_length_a":
            loop.side_lengths = (value, loop.side_lengths[1])
        elif attr == "side_length_b":
            loop.side_lengths = (loop.side_lengths[0], value)
        else:
            setattr(loop, attr, value)

        self._rebuild_scene()

    def _on_path_property_edited(self, meta, value):
        """Apply a path property edit from the tree."""
        path_idx = meta["path"]
        attr = meta["attr"]

        if path_idx >= len(self._sample_paths):
            return
        sp = self._sample_paths[path_idx]

        if attr == "points" and "point_index" in meta:
            # Polyline waypoint edit
            k = meta["point_index"]
            ci = meta["component"]
            sp.points[k, ci] = value
        elif "component" in meta:
            ci = meta["component"]
            vec = getattr(sp, attr).copy()
            vec[ci] = value
            setattr(sp, attr, vec)
        else:
            setattr(sp, attr, value)

        self._sync_path_visual(path_idx)
        self._refresh_tree()
        if path_idx == self._selected_path_index:
            self._update_plot()

    # ------------------------------------------------------------------
    # 2D plot widget with path selector
    # ------------------------------------------------------------------

    def _build_plot_container(self):
        """Build the plot area with a path selector combo overlaid."""
        self._plot_widget = self._build_plot_widget()

        # Overlay the combo box as a child of the plot widget (not in a layout)
        self._path_selector = QComboBox(self._plot_widget)
        self._path_selector.setMinimumWidth(120)
        self._path_selector.currentIndexChanged.connect(self._on_path_selected)
        self._path_selector.raise_()

        # Plot mode radio buttons (overlaid, below the combo)
        self._mode_position_rb = QRadioButton("B vs. position", self._plot_widget)
        self._mode_position_rb.setChecked(True)
        self._mode_position_rb.setStyleSheet("background: rgba(255,255,255,180); padding: 1px 4px;")
        self._mode_position_rb.toggled.connect(
            lambda checked: self._on_plot_mode_changed("position") if checked else None
        )

        self._mode_time_rb = QRadioButton("B vs. time", self._plot_widget)
        self._mode_time_rb.setStyleSheet("background: rgba(255,255,255,180); padding: 1px 4px;")
        self._mode_time_rb.toggled.connect(
            lambda checked: self._on_plot_mode_changed("time") if checked else None
        )

        # Time cursor (vertical line, hidden until time mode)
        self._time_cursor = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("#888888", width=1, style=Qt.PenStyle.DashLine),
            movable=False,
        )
        self._plot_widget.addItem(self._time_cursor)
        self._time_cursor.setVisible(False)

        # Reposition overlaid widgets on resize
        self._resize_filter = _ResizeFilter(self._reposition_plot_overlays)
        self._plot_widget.installEventFilter(self._resize_filter)

        self._refresh_path_selector()

        return self._plot_widget

    def _reposition_plot_overlays(self):
        """Anchor the overlaid widgets to the upper-right corner of the plot."""
        if self._path_selector is None or self._plot_widget is None:
            return
        margin = 6
        pw = self._plot_widget

        # Path selector combo — top right
        combo = self._path_selector
        x = pw.width() - combo.width() - margin
        combo.move(x, margin)

        # Radio buttons — below the combo, right-aligned
        rb_y = margin + combo.height() + 2
        self._mode_position_rb.adjustSize()
        self._mode_time_rb.adjustSize()
        rb_w = max(self._mode_position_rb.width(), self._mode_time_rb.width())
        rb_x = pw.width() - rb_w - margin
        self._mode_position_rb.move(rb_x, rb_y)
        self._mode_time_rb.move(rb_x, rb_y + self._mode_position_rb.height() + 1)

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

        # Waypoint markers (shown for polyline paths)
        self._waypoint_markers = pw.plot(
            [], [], pen=None,
            symbol="o", symbolSize=7,
            symbolBrush=pg.mkBrush(0, 0, 0),
            symbolPen=pg.mkPen(0, 0, 0, width=1),
        )

        return pw

    def _refresh_path_selector(self):
        """Rebuild the path selector combo box items."""
        if self._path_selector is None:
            return
        self._path_selector.blockSignals(True)
        self._path_selector.clear()
        for j in range(len(self._sample_paths)):
            self._path_selector.addItem(f"Path {j}")
        # Clamp selected index
        if self._sample_paths:
            self._selected_path_index = min(
                self._selected_path_index, len(self._sample_paths) - 1,
            )
            self._path_selector.setCurrentIndex(self._selected_path_index)
        else:
            self._selected_path_index = 0
        self._path_selector.setEnabled(len(self._sample_paths) > 0)
        self._path_selector.blockSignals(False)

    def _on_path_selected(self, index):
        """Callback when user selects a different path in the combo."""
        if index < 0:
            return
        self._selected_path_index = index
        self._update_plot()

    def _on_plot_mode_changed(self, mode):
        """Callback when the plot mode radio button is toggled."""
        self._plot_mode = mode
        self._time_cursor.setVisible(mode == "time")
        if mode == "position":
            self._plot_widget.setLabel("bottom", "Distance along path", units="m")
        else:
            self._plot_widget.setLabel("bottom", "Time", units="s")
        self._update_plot()

    def _update_time_cursor(self):
        """Move the time cursor to the current time value."""
        if self._time_cursor is not None and self._plot_mode == "time":
            self._time_cursor.setValue(self._time)

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

        # ---- Export menu ----
        export_menu = menu_bar.addMenu("&Export")

        export_path_action = QAction("Export field along &path...", window)
        export_path_action.triggered.connect(self._on_export_field_along_path)
        export_menu.addAction(export_path_action)

        # ---- Edit menu ----
        edit_menu = menu_bar.addMenu("&Edit")

        add_loop_menu = edit_menu.addMenu("&Add loop")
        add_circular = QAction("&Circular", window)
        add_circular.triggered.connect(self._on_add_circular_loop)
        add_loop_menu.addAction(add_circular)

        add_round_rect = QAction("&Rounded rectangle", window)
        add_round_rect.triggered.connect(self._on_add_round_rect_loop)
        add_loop_menu.addAction(add_round_rect)

        add_inf_line = QAction("&Infinite line", window)
        add_inf_line.triggered.connect(self._on_add_infinite_line)
        add_loop_menu.addAction(add_inf_line)

        add_path_menu = edit_menu.addMenu("Add &path")
        add_line_seg = QAction("&Line segment", window)
        add_line_seg.triggered.connect(self._on_add_line_segment_path)
        add_path_menu.addAction(add_line_seg)

        add_polyline = QAction("&Polyline", window)
        add_polyline.triggered.connect(self._on_add_polyline_path)
        add_path_menu.addAction(add_polyline)

        add_spline = QAction("&Spline", window)
        add_spline.triggered.connect(self._on_add_spline_path)
        add_path_menu.addAction(add_spline)

        edit_menu.addSeparator()

        delete_action = QAction("&Delete selected object", window)
        delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        delete_action.triggered.connect(self._on_delete_selected_object)
        edit_menu.addAction(delete_action)

        edit_menu.addSeparator()

        clear_action = QAction("&Clear all", window)
        clear_action.triggered.connect(self._on_clear_all)
        edit_menu.addAction(clear_action)

    # ------------------------------------------------------------------
    # Add / delete loops and paths
    # ------------------------------------------------------------------

    def _selected_item_info(self):
        """Return ("loop", i), ("path", j), or None for the selected tree item."""
        idx = self._tree_view.currentIndex()
        if not idx.isValid():
            return None
        # Walk up to the top-level row
        while idx.parent().isValid():
            idx = idx.parent()
        item = self._tree_model.itemFromIndex(idx)
        meta = item.data(Qt.ItemDataRole.UserRole + 1)
        if meta is None:
            return None
        return (meta["kind"], meta["index"])

    def _on_add_circular_loop(self):
        """Add a circular loop with default parameters."""
        from .circular_current_loop import CircularCurrentLoop

        extents = self._auto_extents()
        cx = (extents[0] + extents[1]) / 2
        cy = (extents[2] + extents[3]) / 2
        cz = (extents[4] + extents[5]) / 2
        span = (extents[1] - extents[0])

        loop = CircularCurrentLoop(
            diameter=span * 0.4,
            center=[cx, cy, cz],
            normal=[0, 0, 1],
            current=1.0,
        )
        self.simulation.add_loop(loop)
        self._rebuild_scene()

    def _on_add_round_rect_loop(self):
        """Add a rounded-rectangle loop with default parameters."""
        from .round_rect_current_loop import RoundRectCurrentLoop

        extents = self._auto_extents()
        cx = (extents[0] + extents[1]) / 2
        cy = (extents[2] + extents[3]) / 2
        cz = (extents[4] + extents[5]) / 2
        span = (extents[1] - extents[0])

        a = span * 0.4
        b = span * 0.25
        loop = RoundRectCurrentLoop(
            side_lengths=(a, b),
            corner_radius=min(a, b) * 0.1,
            center=[cx, cy, cz],
            normal=[0, 0, 1],
            orientation=[1, 0, 0],
            current=1.0,
        )
        self.simulation.add_loop(loop)
        self._rebuild_scene()

    def _on_add_infinite_line(self):
        """Add an infinite line current with default parameters."""
        from .infinite_line_current import InfiniteLineCurrent

        extents = self._auto_extents()
        cx = (extents[0] + extents[1]) / 2
        cy = (extents[2] + extents[3]) / 2
        cz = (extents[4] + extents[5]) / 2

        loop = InfiniteLineCurrent(
            center=[cx, cy, cz],
            normal=[0, 0, 1],
            current=1.0,
        )
        self.simulation.add_loop(loop)
        self._rebuild_scene()

    def _on_add_line_segment_path(self):
        """Add a line segment path with default endpoints."""
        extents = self._grid_extents or self._auto_extents()
        cx = (extents[0] + extents[1]) / 2
        cy = (extents[2] + extents[3]) / 2
        cz = (extents[4] + extents[5]) / 2
        span = (extents[1] - extents[0]) * 0.4

        path = LineSegmentPath(
            start=[cx - span, cy, cz],
            end=[cx + span, cy, cz],
        )
        self._add_path(path)

    def _on_add_polyline_path(self):
        """Add a polyline path with 3 default waypoints."""
        extents = self._grid_extents or self._auto_extents()
        cx = (extents[0] + extents[1]) / 2
        cy = (extents[2] + extents[3]) / 2
        cz = (extents[4] + extents[5]) / 2
        span = (extents[1] - extents[0]) * 0.4

        path = PolylinePath(points=[
            [cx - span, cy, cz],
            [cx, cy, cz],
            [cx + span, cy, cz],
        ])
        self._add_path(path)

    def _on_add_spline_path(self):
        """Add a spline path with 3 default waypoints."""
        extents = self._grid_extents or self._auto_extents()
        cx = (extents[0] + extents[1]) / 2
        cy = (extents[2] + extents[3]) / 2
        cz = (extents[4] + extents[5]) / 2
        span = (extents[1] - extents[0]) * 0.4

        path = SplinePath(points=[
            [cx - span, cy, cz],
            [cx, cy, cz],
            [cx + span, cy, cz],
        ])
        self._add_path(path)

    def _add_path(self, path):
        """Append a path and update visuals/tree/selector."""
        was_empty = len(self._sample_paths) == 0
        self._sample_paths.append(path)

        if self._sample_paths_visible:
            self._create_path_visual(len(self._sample_paths) - 1)
        elif was_empty:
            # Auto-enable visibility when adding the first path
            self._sample_paths_cb.setChecked(True)

        self._refresh_tree()
        self._refresh_path_selector()
        self._update_plot()

    def _insert_polyline_point(self, path_idx, point_idx, before=True):
        """Insert a new waypoint in a polyline at the midpoint between neighbors."""
        sp = self._sample_paths[path_idx]
        pts = sp.points
        n = len(pts)

        if before:
            if point_idx == 0:
                # Before first point: mirror outward from first segment
                delta = pts[0] - pts[1] if n > 1 else np.array([0.01, 0, 0])
                new_pt = pts[0] + delta * 0.5
                insert_idx = 0
            else:
                new_pt = (pts[point_idx - 1] + pts[point_idx]) / 2
                insert_idx = point_idx
        else:
            if point_idx == n - 1:
                # After last point: mirror outward from last segment
                delta = pts[-1] - pts[-2] if n > 1 else np.array([0.01, 0, 0])
                new_pt = pts[-1] + delta * 0.5
                insert_idx = n
            else:
                new_pt = (pts[point_idx] + pts[point_idx + 1]) / 2
                insert_idx = point_idx + 1

        sp.points = np.insert(sp.points, insert_idx, new_pt, axis=0)

        # Rebuild visual (number of handles changed)
        if self._sample_paths_visible and path_idx < len(self._path_visuals):
            self._teardown_path_visual(path_idx)
            self._create_path_visual(path_idx)

        self._refresh_tree()
        if path_idx == self._selected_path_index:
            self._update_plot()

    def _delete_polyline_point(self, path_idx, point_idx):
        """Remove a waypoint from a polyline (must keep at least 2)."""
        sp = self._sample_paths[path_idx]
        if len(sp.points) <= 2:
            return
        sp.points = np.delete(sp.points, point_idx, axis=0)

        # Rebuild visual (number of handles changed)
        if self._sample_paths_visible and path_idx < len(self._path_visuals):
            self._teardown_path_visual(path_idx)
            self._create_path_visual(path_idx)

        self._refresh_tree()
        if path_idx == self._selected_path_index:
            self._update_plot()

    def _randomize_path(self, path_idx):
        """Randomize all points of a path within the current grid extents."""
        if path_idx >= len(self._sample_paths):
            return
        sp = self._sample_paths[path_idx]
        extents = self._grid_extents or self._auto_extents()
        lo = np.array([extents[0], extents[2], extents[4]])
        hi = np.array([extents[1], extents[3], extents[5]])

        if isinstance(sp, (PolylinePath, SplinePath)):
            n = len(sp.points)
            sp.points = lo + (hi - lo) * np.random.rand(n, 3)
        elif isinstance(sp, LineSegmentPath):
            sp.start = lo + (hi - lo) * np.random.rand(3)
            sp.end = lo + (hi - lo) * np.random.rand(3)

        # Rebuild visual and refresh
        if self._sample_paths_visible and path_idx < len(self._path_visuals):
            self._teardown_path_visual(path_idx)
            self._create_path_visual(path_idx)

        self._refresh_tree()
        if path_idx == self._selected_path_index:
            self._update_plot()

    def _on_clear_all(self):
        """Remove all loops and paths."""
        self.simulation.loops.clear()
        self._teardown_all_path_visuals()
        self._sample_paths.clear()
        self._selected_path_index = 0
        self._rebuild_scene()
        self._refresh_path_selector()
        self._update_plot()
        self._update_time_range()

    def _on_delete_selected_object(self):
        """Delete the object currently selected in the tree."""
        info = self._selected_item_info()
        if info is None:
            return
        kind, idx = info
        if kind == "loop":
            if idx < len(self.simulation.loops):
                self.simulation.remove_loop(idx)
                self._rebuild_scene()
        elif kind == "path":
            self._delete_path(idx)

    def _delete_path(self, path_idx):
        """Remove a path by index, tearing down its visual."""
        if path_idx >= len(self._sample_paths):
            return

        # Tear down visual
        if path_idx < len(self._path_visuals):
            self._teardown_path_visual(path_idx)

        # Remove from lists
        self._sample_paths.pop(path_idx)
        if path_idx < len(self._path_visuals):
            self._path_visuals.pop(path_idx)

        # Rebuild all visuals since indices shifted (widget callbacks capture index)
        if self._sample_paths_visible:
            self._rebuild_all_path_visuals()

        # Clamp selected index
        if self._sample_paths:
            self._selected_path_index = min(
                self._selected_path_index, len(self._sample_paths) - 1,
            )
        else:
            self._selected_path_index = 0

        self._refresh_tree()
        self._refresh_path_selector()
        self._update_plot()

    def _selected_point_index(self):
        """If a polyline point or its child is selected, return (path_idx, point_idx)."""
        idx = self._tree_view.currentIndex()
        if not idx.isValid():
            return None
        # Walk up to 3 levels checking for point_index metadata
        for _ in range(3):
            for col in range(2):
                test_idx = idx.sibling(idx.row(), col)
                if not test_idx.isValid():
                    continue
                item = self._tree_model.itemFromIndex(test_idx)
                if item is None:
                    continue
                meta = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(meta, dict) and "point_index" in meta and "path" in meta:
                    return (meta["path"], meta["point_index"])
            if not idx.parent().isValid():
                break
            idx = idx.parent()
        return None

    def _on_tree_context_menu(self, position):
        """Right-click context menu on the objects tree."""
        info = self._selected_item_info()
        if info is None:
            return

        from PyQt6.QtWidgets import QMenu
        kind, idx = info
        menu = QMenu(self._tree_view)

        if kind == "loop":
            delete_action = menu.addAction("Delete loop")
            action = menu.exec(self._tree_view.viewport().mapToGlobal(position))
            if action == delete_action:
                if idx < len(self.simulation.loops):
                    self.simulation.remove_loop(idx)
                    self._rebuild_scene()
        elif kind == "path":
            point_info = self._selected_point_index()
            sp = self._sample_paths[idx] if idx < len(self._sample_paths) else None

            # Polyline point-level actions
            add_before = add_after = delete_point = None
            if point_info and isinstance(sp, (PolylinePath, SplinePath)):
                add_before = menu.addAction("Add point before")
                add_after = menu.addAction("Add point after")
                if len(sp.points) > 2:
                    delete_point = menu.addAction("Delete point")
                menu.addSeparator()

            randomize_action = menu.addAction("Randomize")
            menu.addSeparator()
            delete_path_action = menu.addAction("Delete path")
            action = menu.exec(self._tree_view.viewport().mapToGlobal(position))

            if action is None:
                pass
            elif action == add_before:
                self._insert_polyline_point(point_info[0], point_info[1], before=True)
            elif action == add_after:
                self._insert_polyline_point(point_info[0], point_info[1], before=False)
            elif action == delete_point:
                self._delete_polyline_point(point_info[0], point_info[1])
            elif action == randomize_action:
                self._randomize_path(idx)
            elif action == delete_path_action:
                self._delete_path(idx)

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
            "sample_paths_visible": self._sample_paths_visible,
            "selected_path_index": self._selected_path_index,
            "time": self._time,
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

        # Restore arrow mode radio button
        if self._arrow_size_mode in self._arrow_radios:
            self._arrow_radios[self._arrow_size_mode].setChecked(True)

        slice_enabled = settings.get("slice_enabled", False)
        self._slice_cb.setChecked(slice_enabled)

        # Restore selected path index before toggling visibility
        self._selected_path_index = settings.get("selected_path_index", 0)

        # Support both new key and old key for backward compatibility
        paths_visible = settings.get(
            "sample_paths_visible",
            settings.get("sample_line_enabled", False),
        )
        self._sample_paths_cb.setChecked(paths_visible)

        # Restore time
        self._time = settings.get("time", 0.0)
        self._time_spin.setValue(self._time)
        self._update_time_range()

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

        simulation, viz_settings, sample_paths = project.load(path)
        self._project_path = path

        self.simulation = simulation
        self._sample_paths = sample_paths

        plotter = self._plotter
        plotter.clear()

        self._field_actor = None
        self._loop_actors.clear()
        self._plane_widget = None
        self._slice_enabled = False

        # Tear down all path visuals
        self._teardown_all_path_visuals()
        self._sample_paths_visible = False

        self._add_loops(plotter, self._loop_line_width)
        self._apply_viz_settings(viz_settings)
        self._update_field()

        plotter.add_axes()
        plotter.reset_camera()

        cam = viz_settings.get("camera_position")
        if cam is not None:
            plotter.camera_position = cam

        self._refresh_tree()
        self._refresh_path_selector()
        self._update_window_title()

    def _on_file_save(self):
        """File → Save callback."""
        if self._project_path:
            project.save(
                self._project_path,
                self.simulation,
                self._viz_settings_to_dict(),
                sample_paths=self._sample_paths,
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
            sample_paths=self._sample_paths,
        )
        self._update_window_title()

    def _on_export_field_along_path(self):
        """Export → Export field along path callback."""
        if not self._sample_paths:
            QMessageBox.information(
                self._window,
                "No sample path",
                "Add a sample path first (Edit \u2192 Add path).",
            )
            return

        sp = self._sample_paths[self._selected_path_index]
        path_length = sp.length

        dlg = ExportFieldAlongPathDialog(path_length, parent=self._window)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        n_points = dlg.point_count()

        # Ask for output file
        csv_path, _ = QFileDialog.getSaveFileName(
            self._window,
            "Export CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not csv_path:
            return
        if not csv_path.lower().endswith(".csv"):
            csv_path += ".csv"

        # Compute field
        points = sp.get_points(n_points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        Bx, By, Bz = self.simulation.magnetic_field_at(x, y, z)
        Bx = np.asarray(Bx).ravel()
        By = np.asarray(By).ravel()
        Bz = np.asarray(Bz).ravel()
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

        # Write CSV
        with open(csv_path, "w") as f:
            f.write(
                f"# Magnesys v{project.CURRENT_VERSION} — "
                f"field along path, {n_points} points, "
                f"interval {dlg.interval():.6g} m, "
                f"path length {path_length:.6g} m\n"
            )
            f.write("x,y,z,Bx,By,Bz,Bmag\n")
            for i in range(n_points):
                f.write(
                    f"{x[i]:.8e},{y[i]:.8e},{z[i]:.8e},"
                    f"{Bx[i]:.8e},{By[i]:.8e},{Bz[i]:.8e},{Bmag[i]:.8e}\n"
                )

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

    def _on_arrow_mode_changed(self, mode, checked):
        """Callback for arrow scaling radio buttons."""
        if not checked:
            return
        self._arrow_size_mode = mode
        if self._auto_update:
            self._update_field()

    def _on_auto_scale_toggled(self, checked):
        """Callback for the auto-scale checkbox."""
        self._auto_scale = checked
        if self._auto_scale:
            # Clear locked ranges so next update auto-scales fresh
            self._locked_clim = None
            self._locked_yrange = None
            self._locked_arrow_scale = None
            self._locked_log_min_mag = None
            if self._plot_widget is not None:
                self._plot_widget.enableAutoRange(axis="y")
            if self._auto_update:
                self._update_field()

    def _on_update_clicked(self):
        """Callback for the manual update button."""
        self._update_field()

    def _on_time_changed(self, value):
        """Callback for the time spin box."""
        self._time = value
        # Sync slider position (0–1000 maps to the auto time range)
        t_max = self._auto_time_range()
        if t_max > 0:
            pos = int(1000 * value / t_max)
            self._time_slider.blockSignals(True)
            self._time_slider.setValue(max(0, min(1000, pos)))
            self._time_slider.blockSignals(False)
        self._update_time_cursor()
        if self._auto_update:
            self._update_field()

    def _on_time_slider_moved(self, pos):
        """Callback for the time slider (0–1000 maps to one full period)."""
        t_max = self._auto_time_range()
        self._time = t_max * pos / 1000.0
        self._time_spin.blockSignals(True)
        self._time_spin.setValue(self._time)
        self._time_spin.blockSignals(False)
        self._update_time_cursor()
        if self._auto_update:
            self._update_field()

    def _auto_time_range(self):
        """Compute a sensible time range from source frequencies."""
        freqs = [
            getattr(loop, "frequency", 0.0)
            for loop in self.simulation.loops
        ]
        freqs = [f for f in freqs if f > 0]
        if not freqs:
            return 1.0  # default 1 second range for DC-only
        min_freq = min(freqs)
        return 1.0 / min_freq  # one full period of the slowest frequency

    def _update_time_range(self):
        """Update the time range label and spin box step size."""
        t_max = self._auto_time_range()
        freqs = [getattr(loop, "frequency", 0.0) for loop in self.simulation.loops]
        freqs = [f for f in freqs if f > 0]
        if freqs:
            self._time_range_label.setText(
                f"Range: 0 \u2013 {t_max:.6g} s (1/{min(freqs):.4g} Hz)"
            )
            self._time_spin.setSingleStep(t_max / 100)
        else:
            self._time_range_label.setText("No AC sources")
            self._time_spin.setSingleStep(0.001)

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

    # ------------------------------------------------------------------
    # Sample path visuals (line widgets for segments, actors for polylines)
    # ------------------------------------------------------------------

    def _on_sample_paths_toggled(self, checked):
        """Callback for the Show sample paths checkbox."""
        self._sample_paths_visible = checked
        plotter = self._plotter
        if plotter is None:
            return

        if self._sample_paths_visible:
            self._rebuild_all_path_visuals()
            self._plot_container.setVisible(True)
            self._refresh_path_selector()
            self._update_plot()
        else:
            self._teardown_all_path_visuals()
            self._plot_container.setVisible(False)

    def _create_path_visual(self, path_idx):
        """Create the appropriate 3D visual for the path at path_idx."""
        plotter = self._plotter
        if plotter is None:
            return
        sp = self._sample_paths[path_idx]

        # Ensure list is long enough
        while len(self._path_visuals) <= path_idx:
            self._path_visuals.append(None)

        if isinstance(sp, (PolylinePath, SplinePath)):
            line_actor = self._make_polyline_line_actor(plotter, sp)
            sphere_widgets = self._make_polyline_handles(plotter, sp, path_idx)
            self._path_visuals[path_idx] = {
                "kind": "polyline",
                "line_actor": line_actor,
                "sphere_widgets": sphere_widgets,
            }
        else:
            # LineSegmentPath — interactive line widget
            extents = self._grid_extents or self._auto_extents()

            def on_moved(polydata, idx=path_idx):
                self._on_line_moved(idx, polydata)

            widget = plotter.add_line_widget(
                on_moved,
                bounds=list(extents),
                factor=1.0,
                resolution=1,
                color="black",
                interaction_event="end",
            )

            if hasattr(sp, "start") and hasattr(sp, "end"):
                widget.SetPoint1(sp.start.tolist())
                widget.SetPoint2(sp.end.tolist())

            widget.SetHandleSize(widget.GetHandleSize() * 0.5)
            widget.GetHandleProperty().SetColor(0.2, 0.2, 0.2)
            widget.GetSelectedHandleProperty().SetColor(1.0, 0.0, 0.0)
            widget.GetSelectedLineProperty().SetColor(0.2, 0.2, 0.2)
            widget.ClampToBoundsOff()

            self._path_visuals[path_idx] = {"kind": "widget", "widget": widget}

    @staticmethod
    def _make_polyline_line_actor(plotter, sp):
        """Add a polyline/spline path as a rendered 3D line and return the actor."""
        if isinstance(sp, SplinePath):
            # Sample the smooth curve densely for rendering
            pts = sp.get_points(max(len(sp.points) * 30, 100))
        else:
            pts = sp.points
        n = len(pts)
        lines = np.column_stack([
            np.full(n - 1, 2),
            np.arange(n - 1),
            np.arange(1, n),
        ])
        poly = pv.PolyData(pts, lines=lines)
        return plotter.add_mesh(
            poly,
            color="black",
            line_width=2.0,
            render_lines_as_tubes=True,
            reset_camera=False,
        )

    def _make_polyline_handles(self, plotter, sp, path_idx):
        """Create sphere widgets for draggable polyline waypoints."""
        pts = sp.points
        extents = pts.max(axis=0) - pts.min(axis=0)
        radius = max(np.max(extents) * 0.012, 1e-4)

        sphere_widgets = []
        for k in range(len(pts)):
            def on_moved(center, idx=path_idx, point_k=k):
                self._on_polyline_sphere_moved(idx, point_k, center)

            widget = plotter.add_sphere_widget(
                on_moved,
                center=pts[k],
                radius=radius,
                color="#333333",
                style="surface",
                selected_color="red",
                interaction_event="always",
            )

            # Add end-interaction observer for tree/plot refresh
            def on_released(_obj, _event, idx=path_idx):
                self._on_polyline_sphere_released(idx)

            widget.AddObserver("EndInteractionEvent", on_released)
            sphere_widgets.append(widget)

        return sphere_widgets

    def _on_polyline_sphere_moved(self, path_idx, point_idx, center):
        """Callback when a polyline sphere handle is dragged.

        Fires continuously during drag — only rebuild the lightweight 3D line.
        Tree and plot are updated on release via the end-interaction callback.
        """
        if path_idx >= len(self._sample_paths):
            return
        sp = self._sample_paths[path_idx]
        sp.points[point_idx] = np.array(center)

        vis = self._path_visuals[path_idx]
        if vis and vis["kind"] == "polyline":
            plotter = self._plotter
            if plotter is not None:
                plotter.remove_actor(vis["line_actor"])
                vis["line_actor"] = self._make_polyline_line_actor(plotter, sp)
                plotter.render()

    def _on_polyline_sphere_released(self, path_idx):
        """Callback when a polyline sphere handle drag ends."""
        self._refresh_tree()
        if path_idx == self._selected_path_index and self._auto_update:
            self._update_plot()

    def _rebuild_all_path_visuals(self):
        """Tear down and recreate all path visuals."""
        self._teardown_all_path_visuals()
        for i in range(len(self._sample_paths)):
            self._create_path_visual(i)

    def _teardown_path_visual(self, path_idx):
        """Tear down a single path visual by index."""
        if path_idx >= len(self._path_visuals):
            return
        vis = self._path_visuals[path_idx]
        if vis is None:
            return
        plotter = self._plotter
        if vis["kind"] == "widget":
            w = vis["widget"]
            w.Off()
            if plotter is not None and w in plotter.line_widgets:
                plotter.line_widgets.remove(w)
        elif vis["kind"] == "polyline":
            if plotter is not None:
                plotter.remove_actor(vis["line_actor"])
                for sw in vis["sphere_widgets"]:
                    sw.Off()
                    if sw in plotter.sphere_widgets:
                        plotter.sphere_widgets.remove(sw)
        self._path_visuals[path_idx] = None

    def _teardown_all_path_visuals(self):
        """Remove all path visuals from the plotter."""
        for i in range(len(self._path_visuals)):
            self._teardown_path_visual(i)
        self._path_visuals.clear()

    def _sync_path_visual(self, path_idx):
        """Push updated path data to its 3D visual."""
        if path_idx >= len(self._path_visuals):
            return
        vis = self._path_visuals[path_idx]
        if vis is None:
            return
        sp = self._sample_paths[path_idx]
        plotter = self._plotter

        if vis["kind"] == "widget":
            w = vis["widget"]
            if hasattr(sp, "start") and hasattr(sp, "end"):
                w.SetPoint1(sp.start.tolist())
                w.SetPoint2(sp.end.tolist())
                if plotter is not None:
                    plotter.render()
        elif vis["kind"] == "polyline" and isinstance(sp, (PolylinePath, SplinePath)):
            # Rebuild entire visual (sphere count may differ, simpler than
            # repositioning individual widgets)
            self._teardown_path_visual(path_idx)
            self._create_path_visual(path_idx)
            if plotter is not None:
                plotter.render()

    def _on_line_moved(self, path_index, polydata):
        """Callback when a sample line widget is dragged."""
        pts = np.array(polydata.points)
        if len(pts) >= 2 and path_index < len(self._sample_paths):
            self._sample_paths[path_index] = LineSegmentPath(
                start=pts[0], end=pts[-1],
            )
            self._refresh_tree()
            if path_index == self._selected_path_index and self._auto_update:
                self._update_plot()

    # ------------------------------------------------------------------
    # Field update
    # ------------------------------------------------------------------

    def _rebuild_scene(self):
        """Rebuild 3D loop geometry, field, plot, and tree after loop edits."""
        plotter = self._plotter
        if plotter is None:
            return

        # Remove only the loop geometry actors (not axes, widgets, etc.)
        for actor in self._loop_actors:
            plotter.remove_actor(actor)
        self._loop_actors.clear()

        self._add_loops(plotter, self._loop_line_width)
        self._update_field()

        # Refresh the tree to show updated values (e.g. re-normalized normals)
        self._refresh_tree()
        self._update_time_range()

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
        if self._sample_paths_visible:
            self._update_plot()

    def _update_plot(self):
        """Recompute B along the selected sample path and update the 2D plot."""
        if not self._plot_curves:
            return

        sp = None
        if (self._sample_paths
                and 0 <= self._selected_path_index < len(self._sample_paths)):
            sp = self._sample_paths[self._selected_path_index]

        if sp is None or not self.simulation.loops:
            for curve in self._plot_curves.values():
                curve.setData([], [])
            if self._waypoint_markers is not None:
                self._waypoint_markers.setData([], [])
            return

        if self._plot_mode == "time":
            self._update_plot_time_mode(sp)
        else:
            self._update_plot_position_mode(sp)

        # Lock or auto-scale the 2D y-axis
        pw = self._plot_widget
        if pw is not None:
            if self._auto_scale:
                pw.enableAutoRange(axis="y")
                yrange = pw.viewRange()[1]
                self._locked_yrange = tuple(yrange)
            else:
                pw.disableAutoRange(axis="y")
                if self._locked_yrange is not None:
                    pw.setYRange(*self._locked_yrange, padding=0)

    def _update_plot_position_mode(self, sp):
        """B vs. position along path at the current time."""
        n = self.N_PATH_SAMPLES
        points = sp.get_points(n)
        distances = sp.get_distances(n)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        Bx, By, Bz = self.simulation.magnetic_field_at(x, y, z, t=self._time)
        Bx = np.asarray(Bx).ravel()
        By = np.asarray(By).ravel()
        Bz = np.asarray(Bz).ravel()
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

        self._plot_curves["Bx"].setData(distances, Bx)
        self._plot_curves["By"].setData(distances, By)
        self._plot_curves["Bz"].setData(distances, Bz)
        self._plot_curves["|B|"].setData(distances, Bmag)

        # Waypoint markers for polyline/spline paths
        if self._waypoint_markers is not None:
            if isinstance(sp, (PolylinePath, SplinePath)):
                wp_dists = np.concatenate([[0.0], np.cumsum(sp.segment_lengths)])
                self._waypoint_markers.setData(
                    wp_dists, np.zeros(len(wp_dists)),
                )
            else:
                self._waypoint_markers.setData([], [])

    def _update_plot_time_mode(self, sp):
        """B vs. time as measured by a sensor moving along the path."""
        n = self.N_PATH_SAMPLES
        t_max = self._auto_time_range()
        t_samples = np.linspace(0, t_max, n)

        # Sample path points (sensor moves at constant speed)
        points = sp.get_points(n)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Precompute each source's spatial field (expensive, done once)
        # Then apply time modulation per sample (cheap)
        Bx_total = np.zeros(n)
        By_total = np.zeros(n)
        Bz_total = np.zeros(n)

        for loop in self.simulation.loops:
            bx, by, bz = loop.magnetic_field(x, y, z)
            bx = np.asarray(bx).ravel()
            by = np.asarray(by).ravel()
            bz = np.asarray(bz).ravel()

            f = getattr(loop, "frequency", 0.0)
            phi = getattr(loop, "phase", 0.0)
            if f != 0.0 or phi != 0.0:
                mod = np.cos(2.0 * np.pi * f * t_samples + phi)
                bx = bx * mod
                by = by * mod
                bz = bz * mod

            Bx_total += bx
            By_total += by
            Bz_total += bz

        Bmag = np.sqrt(Bx_total**2 + By_total**2 + Bz_total**2)

        self._plot_curves["Bx"].setData(t_samples, Bx_total)
        self._plot_curves["By"].setData(t_samples, By_total)
        self._plot_curves["Bz"].setData(t_samples, Bz_total)
        self._plot_curves["|B|"].setData(t_samples, Bmag)

        # Hide waypoint markers in time mode
        if self._waypoint_markers is not None:
            self._waypoint_markers.setData([], [])

        # Update time cursor position
        self._update_time_cursor()

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _add_loops(self, plotter, line_width):
        """Add current loop geometry with current-direction arrowheads."""
        from .infinite_line_current import InfiniteLineCurrent

        extents = self._grid_extents or self._auto_extents()
        grid_span = max(
            extents[1] - extents[0],
            extents[3] - extents[2],
            extents[5] - extents[4],
        )

        for i, loop in enumerate(self.simulation.loops):
            if isinstance(loop, InfiniteLineCurrent):
                path = loop.get_path(half_length=grid_span)
            else:
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
            actor = plotter.add_mesh(
                poly,
                color=color,
                line_width=line_width,
                render_lines_as_tubes=True,
                label=f"Loop {i}",
            )
            self._loop_actors.append(actor)

            # Current-direction arrowhead at the center of the path
            mid = n // 2
            tangent = path[min(mid + 1, n - 1)] - path[max(mid - 1, 0)]
            t_norm = np.linalg.norm(tangent)
            if t_norm > 0:
                tangent = tangent / t_norm
            extent = path.max(axis=0) - path.min(axis=0)
            if isinstance(loop, InfiniteLineCurrent):
                cone_height = grid_span * 0.03
            else:
                cone_height = np.max(extent) * 0.08
            cone = pv.Cone(
                center=path[mid] + tangent * cone_height * 0.5,
                direction=tangent,
                height=cone_height,
                radius=cone_height * 0.4,
                resolution=20,
            )
            actor = plotter.add_mesh(cone, color=color)
            self._loop_actors.append(actor)

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
        Bx, By, Bz = self.simulation.magnetic_field_at(x, y, z, t=self._time)

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

        Bx, By, Bz = self.simulation.magnetic_field_on_grid(X, Y, Z, t=self._time)

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

        # Use locked arrow scale if auto-scale is off and we have a saved value
        if not self._auto_scale and self._locked_arrow_scale is not None:
            resolved_scale = self._locked_arrow_scale
        else:
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
                current_min = magnitudes[nonzero].min()
                # Use locked floor when auto-scale is off
                if not self._auto_scale and self._locked_log_min_mag is not None:
                    min_mag = self._locked_log_min_mag
                else:
                    min_mag = current_min
                    if self._auto_scale:
                        self._locked_log_min_mag = min_mag
                log_scale = np.where(
                    nonzero,
                    np.maximum(np.log10(magnitudes / min_mag) + 1.0, 0.0),
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
                resolved_scale = 0.15 * cell_size / median_mag

            arrows = grid.glyph(
                orient="B", scale="scale", factor=resolved_scale,
            )

        # Capture the resolved scale for locking
        if self._auto_scale:
            self._locked_arrow_scale = resolved_scale

        # Compute color limits using percentiles to ignore near-wire outliers
        mag = arrows["magnitude"]
        if self._auto_scale:
            if len(mag) > 0:
                vmin, vmax = np.percentile(mag, [2, 98])
                if vmax <= vmin:
                    vmax = vmin + 1e-20
                self._locked_clim = (float(vmin), float(vmax))
            else:
                self._locked_clim = None

        mesh_kwargs = dict(
            scalars="magnitude",
            cmap="coolwarm",
            scalar_bar_args={"title": "|B| (T)"},
            reset_camera=False,
        )

        if self._locked_clim is not None:
            mesh_kwargs["clim"] = self._locked_clim

        field_actor = plotter.add_mesh(arrows, **mesh_kwargs)

        return field_actor

    def _auto_extents(self):
        """Compute grid extents from loop positions and sizes."""
        from .infinite_line_current import InfiniteLineCurrent

        if not self.simulation.loops:
            return (-0.1, 0.1, -0.1, 0.1, -0.1, 0.1)

        paths = []
        for loop in self.simulation.loops:
            if isinstance(loop, InfiniteLineCurrent):
                # Use a small region around the center point
                paths.append(loop.get_path(half_length=0.1))
            else:
                paths.append(loop.get_path())
        all_points = np.vstack(paths)
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


class _ResizeFilter(QObject):
    """Event filter that calls a callback on resize events."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self._callback()
        return False


def _format_vec(v):
    """Format a 3-vector for display."""
    return f"({v[0]:.4g}, {v[1]:.4g}, {v[2]:.4g})"
