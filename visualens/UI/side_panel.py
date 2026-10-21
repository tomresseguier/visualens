import os
from functools import partial

import numpy as np
import pyqtgraph as pg
from matplotlib import pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QCheckBox, QSlider, QComboBox,
)

from ..utils.utils_Qt.Special_QWidgets import PlotWidget_xyZooming
from ..utils.utils_Qt.custom_colormaps import get_colormap
from .catalog_controls import CatalogControlsTab
from .controls import ControlBuilder
from .ui_state import CatalogUIState, LensModelUIState


LT_Z_STEP = 0.5   # redshift slider step; 30 steps -> redshift range [0, 15]
LT_Z_N_STEPS = 30
BROAD_FAMILY_COLUMNS = 8
MAP_CMAP_CHOICES = (
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
    'gray', 'hot', 'coolwarm', 'seismic', 'RdBu',
)
MAP_OPACITY_SLIDER_MAX = 100


class ImageSidePanel(QTabWidget) :
    """
    Collapsible left-side panel shown next to the ImageView, with one tab
    each for Image, Catalog and Lens model controls.

    The panel subscribes to ``workspace.signals`` to keep itself in sync with
    the workspace; nothing needs to call its ``refresh_*`` methods by hand.
    """
    def __init__(self, workspace, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.workspace = workspace

        self._critical_curve_toggle = None
        self._caustic_curve_toggle = None
        self._map_plot_toggles = {}
        self._active_map_plot = None
        self._mag_contours_toggle = None
        self._mag_abs_checkbox = None

        self.image_tab = self._make_image_tab()
        self.catalog_tab, self.catalog_subtabs = self._make_catalog_tab()
        self.lens_model_tab = self._make_lens_model_tab()

        self.addTab(ControlBuilder.scroll_area(self.image_tab), 'Image')
        self.addTab(ControlBuilder.scroll_area(self.catalog_tab), 'Catalog')
        self.addTab(ControlBuilder.scroll_area(self.lens_model_tab), 'Lens model')

        self._connect_workspace_signals()
        self.refresh_all()

    def _connect_workspace_signals(self) :
        signals = self.workspace.signals
        signals.image_changed.connect(self.sync_link_button_states)
        signals.catalogs_changed.connect(self.refresh_catalog_tabs)
        signals.lens_model_changed.connect(self.refresh_lens_model_tab)
        signals.window_attached.connect(self.refresh_all)

    def refresh_all(self) :
        self.sync_link_button_states()
        self.refresh_catalog_tabs()
        self.refresh_lens_model_tab()

    def _dialog_parent(self) :
        return self.workspace._dialog_parent()

    # ------------------------------------------------------------------
    # Image tab
    # ------------------------------------------------------------------
    def _make_image_tab(self) :
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        for label, callback, stretch in (
            ('Open…', self.workspace.open_image_dialog, 1),
            ('Show viewer', self.workspace.reopen_main_window, 2),
            ('2nd window', self.workspace.plot_secondary_image, 2),
            ('Split', self.workspace.toggle_split4, 2),
        ) :
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            top_row.addWidget(btn, stretch)
        layout.addLayout(top_row)

        link_row = QHBoxLayout()
        self.zoom_link_btn = ControlBuilder.toggle_button('Zoom link', self._toggle_zoom_link)
        self.pan_link_btn = ControlBuilder.toggle_button('Pan link', self._toggle_pan_link)
        link_row.addWidget(self.zoom_link_btn)
        link_row.addWidget(self.pan_link_btn)
        layout.addLayout(link_row)

        layout.addWidget(ControlBuilder.separator())

        layout.addWidget(QLabel('Hand selected multiple images'))

        hand_select_row = QHBoxLayout()
        export_btn = QPushButton('export selection…')
        export_btn.clicked.connect(self._export_hand_selection)
        hand_select_row.addWidget(export_btn)

        clear_btn = QPushButton('clear_selection')
        clear_btn.clicked.connect(self._clear_hand_selection)
        hand_select_row.addWidget(clear_btn)
        layout.addLayout(hand_select_row)

        layout.addStretch()
        return tab

    def _export_hand_selection(self) :
        img = self.workspace.image
        if img is None :
            return
        base_dir = os.path.dirname(img.image_path) if img.image_path is not None else os.getcwd()
        path_str, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            'Export hand selection',
            os.path.join(base_dir, 'hand_selected_catalog.lenstool'),
            'Lenstool mult files (*.lenstool);;All files (*)',
        )
        if not path_str :
            return
        img.export_hand_selection_as_mult_file(path_str)

    def _clear_hand_selection(self) :
        img = self.workspace.image
        if img is None :
            return
        img.clear_hand_selection()

    def _toggle_zoom_link(self, linked) :
        img = self.workspace.image
        if img is None :
            return
        if linked :
            img.link_zoom()
        else :
            img.unlink_zoom()

    def _toggle_pan_link(self, linked) :
        img = self.workspace.image
        if img is None :
            return
        if linked :
            img.link_pan()
        else :
            img.unlink_pan()

    def sync_link_button_states(self) :
        img = self.workspace.image
        sync = img._view_range_sync_state() if img is not None else {'zoom' : False, 'pan' : False}
        with ControlBuilder.signals_blocked(self.zoom_link_btn, self.pan_link_btn) :
            self.zoom_link_btn.setChecked(sync['zoom'])
            self.pan_link_btn.setChecked(sync['pan'])

    # ------------------------------------------------------------------
    # Lens model tab
    # ------------------------------------------------------------------
    def _make_lens_model_tab(self) :
        tab = QWidget()
        layout = QVBoxLayout(tab)

        open_btn = QPushButton('Open…')
        open_btn.clicked.connect(self.workspace.open_lens_model_dialog)
        layout.addWidget(open_btn)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel('Set multiple image system(s):'))
        all_btn = QPushButton('All')
        all_btn.clicked.connect(self._select_all_broad_families)
        header_row.addWidget(all_btn)
        none_btn = QPushButton('None')
        none_btn.clicked.connect(self._select_no_broad_families)
        header_row.addWidget(none_btn)
        header_row.addStretch()
        layout.addLayout(header_row)

        self._lens_model_families_container = QWidget()
        layout.addWidget(self._lens_model_families_container)

        layout.addWidget(ControlBuilder.separator())

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel('Source plane redshift'))
        self._lt_z_value_label = QLabel('0.0')
        z_row.addWidget(self._lt_z_value_label)
        z_row.addStretch()
        layout.addLayout(z_row)

        self.lt_z_slider = QSlider(Qt.Horizontal)
        self.lt_z_slider.setMinimum(0)
        self.lt_z_slider.setMaximum(LT_Z_N_STEPS)
        self.lt_z_slider.setSingleStep(1)
        self.lt_z_slider.setValue(0)
        self.lt_z_slider.valueChanged.connect(self._on_lt_z_slider_value_changed)
        self.lt_z_slider.sliderReleased.connect(self._on_lt_z_slider_released)
        layout.addWidget(self.lt_z_slider)

        layout.addWidget(ControlBuilder.separator())

        self.lens_model_subtabs = QTabWidget()
        layout.addWidget(self.lens_model_subtabs, 1)

        layout.addWidget(ControlBuilder.separator())

        self._optimization_container = QWidget()
        QVBoxLayout(self._optimization_container).setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._optimization_container)

        on_the_fly_btn = QPushButton('Start on the fly predicted images')
        on_the_fly_btn.clicked.connect(self._start_on_the_fly_predicted_images)
        layout.addWidget(on_the_fly_btn)

        layout.addWidget(ControlBuilder.separator())

        open_imsim_btn = QPushButton('Open image simulator')
        open_imsim_btn.clicked.connect(self._open_image_simulator)
        layout.addWidget(open_imsim_btn)

        return tab

    def refresh_lens_model_tab(self) :
        """Rebuild the dynamic parts of the Lens model tab to match ``workspace.lens_model``."""
        self._rebuild_broad_family_checkboxes()
        self._rebuild_lens_model_subtabs()
        self._rebuild_optimization_section()

        lens_model = self.workspace.lens_model
        if lens_model is not None and getattr(lens_model, 'z_source', None) is not None :
            with ControlBuilder.signals_blocked(self.lt_z_slider) :
                self.lt_z_slider.setValue(round(lens_model.z_source / LT_Z_STEP))
            self._lt_z_value_label.setText(f"{lens_model.z_source:.1f}")

    def _rebuild_optimization_section(self) :
        """Chi2-vs-iteration plot and corner-plot button; only present if the model has burn-in/bayes tables."""
        container = self._optimization_container
        layout = container.layout()
        while layout.count() > 0 :
            widget = layout.takeAt(0).widget()
            if widget is not None :
                widget.deleteLater()

        lens_model = self.workspace.lens_model
        has_burnin = lens_model is not None and getattr(lens_model, 'burnin_table', None) is not None
        has_bayes = lens_model is not None and getattr(lens_model, 'bayes_table', None) is not None
        container.setVisible(has_burnin or has_bayes)
        if not (has_burnin or has_bayes) :
            return

        layout.addWidget(QLabel('Optimization'))

        try :
            chi2_dict = lens_model.plot_chi2(show_matplotlib=False)
        except Exception as exc :
            ControlBuilder.error(self._dialog_parent(), 'Failed to compute chi2 curve', exc)
            chi2_dict = None

        if chi2_dict is not None :
            plot_widget = PlotWidget_xyZooming()
            plot_widget.setMinimumHeight(180)
            plot_widget.setLabel('bottom', 'Iteration')
            plot_widget.setLabel('left', 'Chi2')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            legend = plot_widget.addLegend()
            #legend.setMovable(False)
            for key, label, color in (('burnin', 'Burn-in', 'c'), ('sampling', 'Sampling', 'm')) :
                curve = chi2_dict.get(key)
                if curve is not None :
                    plot_widget.plot(
                        np.asarray(curve['steps']), np.asarray(curve['chi2']),
                        pen=pg.mkPen(color, width=1.5), name=label,
                    )
            sampling = chi2_dict.get('sampling')
            if sampling is not None :
                chi2 = np.asarray(sampling['chi2'])
                y_std = np.std(chi2)
                plot_widget.setYRange(np.min(chi2) - 20 * y_std, np.max(chi2) + 20 * y_std, padding=0)
            layout.addWidget(plot_widget)

        if has_bayes :
            corner_btn = QPushButton('Corner plot')
            corner_btn.clicked.connect(self._show_corner_plot)
            layout.addWidget(corner_btn)

        layout.addWidget(ControlBuilder.separator())

    def _show_corner_plot(self) :
        lens_model = self._require_lens_model()
        if lens_model is None :
            return
        try :
            lens_model.plot_bayes()
            plt.show(block=False)
        except Exception as exc :
            ControlBuilder.error(self._dialog_parent(), 'Failed to plot corner plot', exc)

    def _rebuild_broad_family_checkboxes(self) :
        grid = ControlBuilder.reset_grid(self._lens_model_families_container)

        lens_model = self.workspace.lens_model
        if lens_model is None :
            return

        broad_families = lens_model.broad_families
        if not isinstance(broad_families, list) :
            broad_families = broad_families.tolist()

        state = LensModelUIState.of(lens_model)
        state.sync_with(broad_families)

        for i, name in enumerate(broad_families) :
            checkbox = QCheckBox(str(name))
            checkbox.setChecked(name in state.checked_broad_families)
            checkbox.toggled.connect(partial(self._on_broad_family_toggled, name))
            row, col = divmod(i, BROAD_FAMILY_COLUMNS)
            grid.addWidget(checkbox, row, col)

    def _on_broad_family_toggled(self, name, checked) :
        lens_model = self.workspace.lens_model
        if lens_model is None :
            return
        state = LensModelUIState.of(lens_model)
        state.set_checked(name, checked)

        lens_model.set_which(state.checked_in_order(lens_model.broad_families))
        self._replot_lens_model_which(lens_model)

    def _select_all_broad_families(self) :
        lens_model = self.workspace.lens_model
        if lens_model is None :
            return
        broad_families = lens_model.broad_families
        if not isinstance(broad_families, list) :
            broad_families = broad_families.tolist()
        state = LensModelUIState.of(lens_model)
        state.checked_broad_families = set(broad_families)
        lens_model.set_which(broad_families)
        self._replot_lens_model_which(lens_model)
        self._rebuild_broad_family_checkboxes()

    def _select_no_broad_families(self) :
        lens_model = self.workspace.lens_model
        if lens_model is None :
            return
        state = LensModelUIState.of(lens_model)
        state.checked_broad_families = set()
        lens_model.set_which([])
        self._replot_lens_model_which(lens_model)
        self._rebuild_broad_family_checkboxes()

    def _replot_lens_model_which(self, lens_model) :
        for cat in (lens_model.mult, lens_model.source, lens_model.images_filtered, lens_model.potfile) :
            if cat is None :
                continue
            cat_state = CatalogUIState.existing(cat)
            if cat_state is not None :
                cat_state.replot(cat)

    def _on_lt_z_slider_value_changed(self, value) :
        self._lt_z_value_label.setText(f"{value * LT_Z_STEP:.1f}")

    def _on_lt_z_slider_released(self) :
        lens_model = self.workspace.lens_model
        if lens_model is None :
            return
        z = self.lt_z_slider.value() * LT_Z_STEP
        try :
            lens_model.set_z_source(
                z,
                plot_critical=self._is_checked(self._critical_curve_toggle),
                plot_caustic=self._is_checked(self._caustic_curve_toggle),
            )
            self._replot_active_lens_map(lens_model)
        except Exception as exc :
            ControlBuilder.error(self._dialog_parent(), 'Failed to set redshift', exc)

    @staticmethod
    def _is_checked(btn) :
        return btn is not None and btn.isChecked()

    def _rebuild_lens_model_subtabs(self) :
        subtabs = self.lens_model_subtabs
        current_index = subtabs.currentIndex()
        ControlBuilder.clear_tabs(subtabs)

        lens_model = self.workspace.lens_model

        for title, attr in (
            ('Multiple images', 'mult'),
            ('Sources', 'source'),
            ('Predicted images', 'images_filtered'),
            ('Cluster members', 'potfile'),
        ) :
            cat = getattr(lens_model, attr, None) if lens_model is not None else None
            if cat is not None and attr in ('mult', 'potfile') :
                CatalogUIState.of(cat, plot_column=True, colname='id')
            subtabs.addTab(self._make_catalog_controls_tab(cat) if cat is not None else QWidget(), title)

        subtabs.addTab(self._make_maps_and_curves_tab(lens_model), 'Maps and curves')

        if 0 <= current_index < subtabs.count() :
            subtabs.setCurrentIndex(current_index)

    @staticmethod
    def _maps_section_label(text) :
        label = QLabel(text)
        label.setStyleSheet('font-weight: bold; margin-top: 4px;')
        return label

    def _make_maps_and_curves_tab(self, lens_model) :
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self._map_plot_toggles = {}
        self._active_map_plot = None
        self._mag_contours_toggle = None
        self._mag_abs_checkbox = None

        map_style_row = QHBoxLayout()
        map_style_row.addWidget(QLabel('Colormap'))
        self._map_cmap_combo = QComboBox()
        self._map_cmap_combo.addItems(MAP_CMAP_CHOICES)
        map_style_row.addWidget(self._map_cmap_combo, 1)
        map_style_row.addWidget(QLabel('Opacity'))
        self._map_opacity_slider = QSlider(Qt.Horizontal)
        self._map_opacity_slider.setMinimum(0)
        self._map_opacity_slider.setMaximum(MAP_OPACITY_SLIDER_MAX)
        self._map_opacity_slider.setSingleStep(5)
        map_style_row.addWidget(self._map_opacity_slider, 1)
        layout.addLayout(map_style_row)
        self._sync_map_style_controls_from_lens_model(lens_model)
        self._map_cmap_combo.currentTextChanged.connect(self._on_map_cmap_changed)
        self._map_opacity_slider.valueChanged.connect(self._on_map_opacity_slider_changed)

        layout.addWidget(self._maps_section_label('Magnification'))
        mag_row = QHBoxLayout()
        self._mag_abs_checkbox = QCheckBox('Abs')
        self._mag_abs_checkbox.setChecked(
            lens_model is not None and getattr(lens_model, 'magnification_abs', False),
        )
        self._mag_abs_checkbox.toggled.connect(self._on_magnification_abs_toggled)
        self._map_plot_toggles['magnification'] = ControlBuilder.toggle_button(
            'Plot', partial(self._toggle_lens_map_plot, which='magnification'),
        )
        self._mag_contours_toggle = ControlBuilder.toggle_button(
            'Plot contours', self._toggle_magnification_contours,
        )
        mag_row.addWidget(self._mag_abs_checkbox)
        mag_row.addWidget(self._map_plot_toggles['magnification'])
        mag_row.addWidget(self._mag_contours_toggle)
        layout.addLayout(mag_row)

        layout.addWidget(self._maps_section_label('Convergence'))
        conv_row = QHBoxLayout()
        self._map_plot_toggles['convergence'] = ControlBuilder.toggle_button(
            'Plot', partial(self._toggle_lens_map_plot, which='convergence'),
        )
        conv_row.addWidget(self._map_plot_toggles['convergence'])
        layout.addLayout(conv_row)

        layout.addWidget(self._maps_section_label('Shear'))
        shear_row = QHBoxLayout()
        self._map_plot_toggles['shear'] = ControlBuilder.toggle_button(
            'Plot', partial(self._toggle_lens_map_plot, which='shear'),
        )
        shear_row.addWidget(self._map_plot_toggles['shear'])
        layout.addLayout(shear_row)

        layout.addWidget(ControlBuilder.separator())

        layout.addWidget(self._maps_section_label('Curves'))
        critical_on = lens_model is not None and getattr(lens_model, 'critical_curve_plot', None) is not None
        caustic_on = lens_model is not None and getattr(lens_model, 'caustic_curve_plot', None) is not None
        curves_row = QHBoxLayout()
        self._critical_curve_toggle = ControlBuilder.toggle_button(
            'Plot critical curve', partial(self._toggle_lt_curve, which='critical'), critical_on,
        )
        self._caustic_curve_toggle = ControlBuilder.toggle_button(
            'Plot caustic curve', partial(self._toggle_lt_curve, which='caustic'), caustic_on,
        )
        curves_row.addWidget(self._critical_curve_toggle)
        curves_row.addWidget(self._caustic_curve_toggle)
        layout.addLayout(curves_row)
        layout.addStretch()
        return tab

    def _sync_map_style_controls_from_lens_model(self, lens_model) :
        cmap = 'viridis'
        opacity = 0.6
        if lens_model is not None :
            cmap = getattr(lens_model, 'map_cmap', cmap)
            cmap = getattr(cmap, 'name', cmap)   # map_cmap may be a pg.ColorMap
            opacity = getattr(lens_model, 'map_opacity', opacity)
        combo = getattr(self, '_map_cmap_combo', None)
        slider = getattr(self, '_map_opacity_slider', None)
        if combo is not None :
            with ControlBuilder.signals_blocked(combo) :
                idx = combo.findText(cmap)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
        if slider is not None :
            with ControlBuilder.signals_blocked(slider) :
                slider.setValue(round(opacity * MAP_OPACITY_SLIDER_MAX))

    def _on_map_cmap_changed(self, cmap) :
        lens_model = self.workspace.lens_model
        if lens_model is None :
            return
        lens_model.map_cmap = get_colormap(cmap)   # custom 5-point version when available
        lens_model._apply_map_style()

    def _on_map_opacity_slider_changed(self, value) :
        lens_model = self.workspace.lens_model
        if lens_model is None :
            return
        lens_model.map_opacity = value / MAP_OPACITY_SLIDER_MAX
        lens_model._apply_map_style()

    def _on_magnification_abs_toggled(self, checked) :
        lens_model = self.workspace.lens_model
        if lens_model is None :
            return
        lens_model.magnification_abs = checked
        if self._active_map_plot == 'magnification' :
            self._replot_active_lens_map(lens_model)

    def _toggle_magnification_contours(self, checked) :
        if not checked :
            return
        ControlBuilder.warn(
            self._dialog_parent(), 'Not implemented',
            'Magnification contours will be available in a future version.',
        )
        ControlBuilder.set_checked_silently(self._mag_contours_toggle, False)

    def _toggle_lens_map_plot(self, checked, which) :
        lens_model = self.workspace.lens_model
        if lens_model is None :
            toggle = self._map_plot_toggles.get(which)
            if toggle is not None :
                ControlBuilder.set_checked_silently(toggle, False)
            return

        toggle = self._map_plot_toggles[which]
        if not checked :
            if self._active_map_plot == which :
                lens_model._clear_map()
                self._active_map_plot = None
            return

        if getattr(lens_model, 'z_source', None) is None :
            ControlBuilder.warn(
                self._dialog_parent(), 'No redshift set',
                'Please set the source plane redshift with the slider first.',
            )
            ControlBuilder.set_checked_silently(toggle, False)
            return

        for key, btn in self._map_plot_toggles.items() :
            if key != which :
                ControlBuilder.set_checked_silently(btn, False)

        plotters = {
            'magnification' : lens_model.plot_magnification,
            'convergence' : lens_model.plot_convergence,
            'shear' : lens_model.plot_shear,
        }
        try :
            plotters[which]()
            self._active_map_plot = which
        except Exception as exc :
            ControlBuilder.error(self._dialog_parent(), 'Failed to plot map', exc)
            ControlBuilder.set_checked_silently(toggle, False)
            self._active_map_plot = None

    def _replot_active_lens_map(self, lens_model) :
        which = self._active_map_plot
        if which is None or lens_model is None :
            return
        plotters = {
            'magnification' : lens_model.plot_magnification,
            'convergence' : lens_model.plot_convergence,
            'shear' : lens_model.plot_shear,
        }
        try :
            plotters[which]()
        except Exception as exc :
            ControlBuilder.error(self._dialog_parent(), 'Failed to replot map', exc)

    def _toggle_lt_curve(self, checked, which) :
        lens_model = self.workspace.lens_model
        if lens_model is None :
            return
        if checked :
            if getattr(lens_model, 'z_source', None) is None :
                ControlBuilder.warn(
                    self._dialog_parent(), 'No redshift set',
                    'Please set the source plane redshift with the slider first.',
                )
                toggle = self._critical_curve_toggle if which == 'critical' else self._caustic_curve_toggle
                ControlBuilder.set_checked_silently(toggle, False)
                return
            try :
                lens_model.plot_lt_curve(which=which)
            except Exception as exc :
                ControlBuilder.error(self._dialog_parent(), 'Failed to plot curve', exc)
        else :
            lens_model.clear_lt_curve(which=which)

    def _start_on_the_fly_predicted_images(self) :
        lens_model = self._require_lens_model(need_redshift=True)
        if lens_model is None :
            return
        try :
            lens_model.start_im2source()
        except Exception as exc :
            ControlBuilder.error(self._dialog_parent(), 'Failed to start on the fly predicted images', exc)

    def _open_image_simulator(self) :
        lens_model = self._require_lens_model()
        if lens_model is None :
            return
        try :
            lens_model.start_simulate_image()
        except Exception as exc :
            ControlBuilder.error(self._dialog_parent(), 'Failed to open image simulator', exc)

    def _require_lens_model(self, need_redshift=False) :
        """Return the workspace lens model, or warn the user and return None if it is not usable."""
        lens_model = self.workspace.lens_model
        if lens_model is None :
            ControlBuilder.warn(self._dialog_parent(), 'No model', 'Please import a Lenstool model first.')
            return None
        if need_redshift and getattr(lens_model, 'z_source', None) is None :
            ControlBuilder.warn(
                self._dialog_parent(), 'No redshift set',
                'Please set the source plane redshift with the slider first.',
            )
            return None
        return lens_model

    # ------------------------------------------------------------------
    # Catalog tab: one sub-tab per catalog in workspace.catalogs
    # ------------------------------------------------------------------
    def _make_catalog_tab(self) :
        tab = QWidget()
        layout = QVBoxLayout(tab)

        open_btn = QPushButton('Open…')
        open_btn.clicked.connect(self.workspace.open_catalog_dialog)
        layout.addWidget(open_btn)

        subtabs = QTabWidget()
        layout.addWidget(subtabs, 1)
        return tab, subtabs

    def refresh_catalog_tabs(self) :
        """Rebuild the per-catalog sub-tabs to match ``workspace.catalogs``."""
        current_index = self.catalog_subtabs.currentIndex()
        ControlBuilder.clear_tabs(self.catalog_subtabs)

        for i, cat in enumerate(self.workspace.catalogs) :
            self.catalog_subtabs.addTab(self._make_catalog_controls_tab(cat), f'Cat {i + 1}')

        if 0 <= current_index < self.catalog_subtabs.count() :
            self.catalog_subtabs.setCurrentIndex(current_index)

    def _make_catalog_controls_tab(self, cat) :
        return CatalogControlsTab(self.workspace, cat)
