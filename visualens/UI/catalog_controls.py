import os
from functools import partial

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from .color_picker import ColorControls
from .controls import ControlBuilder
from .ui_state import MARKER_CHOICES, CatalogUIState


class CatalogControlsTab(QWidget) :
    """
    Plotting options, column selection, interactive selection panel and export
    buttons for a single ``Catalog``. The state lives in ``CatalogUIState`` on
    the catalog, so rebuilding this widget never loses the user's choices.
    """
    def __init__(self, workspace, cat, parent=None) :
        super().__init__(parent)
        self.workspace = workspace
        self.cat = cat
        self.state = CatalogUIState.of(cat)
        self._build()

    def _build(self) :
        state = self.state
        colnames = [str(c) for c in self.cat.cat.colnames]
        layout = QVBoxLayout(self)

        #layout.addWidget(ControlBuilder.separator())
        #layout.addWidget(QLabel('Plotting options'))

        self.color_controls = ColorControls(state.h, state.s, state.v, state.opacity, parent=self)
        self.color_controls.changed.connect(self._on_color_changed)
        self.color_controls.released.connect(self._on_color_released)
        self.color_controls.set_enabled(not state.use_default_color)

        default_color_box = QCheckBox('Use default color')
        default_color_box.setChecked(state.use_default_color)
        default_color_box.toggled.connect(self._on_use_default_color_toggled)

        left_block = QVBoxLayout()
        left_block.addWidget(ControlBuilder.toggle_button('Plot', self._on_plot_toggled, state.plot))
        left_block.addWidget(ControlBuilder.toggle_button('Plot column', self._on_plot_column_toggled, state.plot_column))
        left_block.addWidget(default_color_box)
        left_block.addLayout(self.color_controls.brightness_row)
        left_block.addLayout(self.color_controls.opacity_row)
        left_block.addStretch()

        top_row = QHBoxLayout()
        top_row.addLayout(left_block, 1)
        top_row.addWidget(self.color_controls.disk, 0, Qt.AlignTop)
        layout.addLayout(top_row)
        layout.addLayout(self._make_marker_row())

        colname_list = ControlBuilder.list_widget(colnames, 150, state.colname)
        colname_list.currentTextChanged.connect(self._on_colname_changed)
        layout.addWidget(colname_list)

        export_mult_btn = QPushButton('Export to multiple image file…')
        export_mult_btn.clicked.connect(self._export_mult_file)
        layout.addWidget(export_mult_btn)

        layout.addWidget(ControlBuilder.separator())
        layout.addWidget(QLabel('Interactive selection panel'))

        x_layout, x_list = ControlBuilder.labeled_list('x-axis', colnames, 120, state.x_colname)
        y_layout, y_list = ControlBuilder.labeled_list('y-axis', colnames, 120, state.y_colname)
        x_list.currentTextChanged.connect(self._on_x_colname_changed)
        y_list.currentTextChanged.connect(self._on_y_colname_changed)
        xy_row = QHBoxLayout()
        xy_row.addLayout(x_layout)
        xy_row.addLayout(y_layout)
        layout.addLayout(xy_row)

        selection_row = QHBoxLayout()
        self.selection_btn = ControlBuilder.toggle_button(
            'Make selection panel', self._on_selection_panel_toggled, state.selection_panel,
        )
        selection_row.addWidget(self.selection_btn)

        export_potfile_btn = QPushButton('Export to potfile…')
        export_potfile_btn.clicked.connect(self._export_potfile)
        selection_row.addWidget(export_potfile_btn)
        layout.addLayout(selection_row)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Plotting options
    # ------------------------------------------------------------------
    def _make_marker_row(self) :
        """Marker drop-down, scale and linewidth boxes (arguments of ``cat.plot``) on a single line."""
        state = self.state

        marker_box = QComboBox()
        marker_box.addItems(MARKER_CHOICES)
        if state.marker not in MARKER_CHOICES :
            marker_box.addItem(state.marker)
        marker_box.setCurrentText(state.marker)
        marker_box.currentTextChanged.connect(self._on_marker_changed)

        self.scale_edit = self._make_number_edit(state.scale, 'scale')
        self.linewidth_edit = self._make_number_edit(state.linewidth, 'linewidth')

        row = QHBoxLayout()
        row.addWidget(QLabel('Marker'))
        row.addWidget(marker_box, 2)
        row.addWidget(QLabel('Scale'))
        row.addWidget(self.scale_edit, 1)
        row.addWidget(QLabel('Linewidth'))
        row.addWidget(self.linewidth_edit, 1)
        return row

    def _make_number_edit(self, value, attr) :
        edit = QLineEdit(f"{value:g}")
        edit.setValidator(QDoubleValidator(0., 1e6, 6, edit))
        edit.setMinimumWidth(30)
        edit.editingFinished.connect(partial(self._on_number_edited, edit, attr))
        return edit

    def _on_marker_changed(self, marker) :
        if not marker or marker == self.state.marker :
            return
        self.state.marker = marker
        self.state.replot(self.cat)

    def _on_number_edited(self, edit, attr) :
        current = getattr(self.state, attr)
        try :
            value = float(edit.text().replace(',', '.'))
        except ValueError :
            value = current
        edit.setText(f"{value:g}")
        if value == current :
            return
        setattr(self.state, attr, value)
        self.state.replot(self.cat)

    def _on_plot_toggled(self, checked) :
        self.state.plot = checked
        if checked :
            self.state.replot(self.cat)
        else :
            self.cat.clear_ellipses()

    def _on_plot_column_toggled(self, checked) :
        self.state.plot_column = checked
        if checked :
            self.state.replot_column(self.cat)
        else :
            self.cat.clear_column()

    def _on_colname_changed(self, colname) :
        if not colname :
            return
        self.state.colname = colname
        if self.state.plot_column :
            self.state.replot_column(self.cat)

    def _on_color_changed(self, hue, saturation, value, opacity) :
        state = self.state
        state.h, state.s, state.v, state.opacity = hue, saturation, value, opacity

    def _on_color_released(self) :
        self.state.replot(self.cat)

    def _on_use_default_color_toggled(self, checked) :
        self.state.use_default_color = checked
        self.color_controls.set_enabled(not checked)
        self.state.replot(self.cat)

    # ------------------------------------------------------------------
    # Interactive selection panel
    # ------------------------------------------------------------------
    def _on_x_colname_changed(self, colname) :
        if not colname :
            return
        self.state.x_colname = colname
        self._remake_selection_panel_if_active()

    def _on_y_colname_changed(self, colname) :
        if not colname :
            return
        self.state.y_colname = colname
        self._remake_selection_panel_if_active()

    def _remake_selection_panel_if_active(self) :
        state = self.state
        if state.selection_panel and state.x_colname is not None and state.y_colname is not None :
            self._show_selection_panel()

    def _show_selection_panel(self) :
        if self.cat.Scatter_widget is not None :
            self.cat.remove_selection_panel()
        self.cat.make_selection_panel(xy_axes=[self.state.x_colname, self.state.y_colname])

    def _on_selection_panel_toggled(self, checked) :
        state = self.state
        if checked :
            if state.x_colname is None or state.y_colname is None :
                ControlBuilder.warn(
                    self.workspace._dialog_parent(), 'Missing axis',
                    'Please select both an x and y column first.',
                )
                ControlBuilder.set_checked_silently(self.selection_btn, False)
                return
            self._show_selection_panel()
            state.selection_panel = True
        else :
            if self.cat.Scatter_widget is not None :
                self.cat.remove_selection_panel()
            state.selection_panel = False

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def _export_mult_file(self) :
        cat = self.cat
        if cat.image is None or cat.image.image_path is None :
            base_dir = os.path.dirname(cat.path) if cat.path else os.getcwd()
        else :
            base_dir = os.path.dirname(cat.image.image_path)
        path_str, _ = QFileDialog.getSaveFileName(
            self.workspace._dialog_parent(),
            'Export to multiple image file',
            os.path.join(base_dir, 'mult.lenstool'),
            'Lenstool mult files (*.lenstool);;All files (*)',
        )
        if not path_str :
            return

        try :
            cat.export_to_mult_file(file_path=path_str)
        except Exception as exc :
            ControlBuilder.error(self.workspace._dialog_parent(), 'Failed to export multiple image file', exc)

    def _export_potfile(self) :
        cat = self.cat
        colnames = [str(c) for c in cat.cat.colnames]
        default_idx = colnames.index(self.state.x_colname) if self.state.x_colname in colnames else 0

        mag_col, ok = QInputDialog.getItem(
            self.workspace._dialog_parent(),
            'Export to potfile',
            'Select magnitude column',
            colnames,
            default_idx,
            False,
        )
        if not ok :
            return

        base_dir = os.path.dirname(cat.path) if cat.path else os.getcwd()
        path_str, _ = QFileDialog.getSaveFileName(
            self.workspace._dialog_parent(),
            'Export to potfile',
            os.path.join(base_dir, 'exported_potfile.lenstool'),
            'Lenstool potfiles (*.lenstool);;All files (*)',
        )
        if not path_str :
            return

        try :
            cat.export_to_potfile(file_path=path_str, mag_col=mag_col)
        except Exception as exc :
            ControlBuilder.error(self.workspace._dialog_parent(), 'Failed to export potfile', exc)
