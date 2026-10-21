from PyQt5.QtWidgets import QPushButton, QFileDialog, QMessageBox, QWidget, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5 import sip

from .image import Image
from .catalog import Catalog
from .LensModel.LensModel import LensModel
from .UI.side_panel import ImageSidePanel
from .UI.workspace_signals import WorkspaceSignals




class Visualens :
    def __init__(self, image_path=None, main_window=None) :
        # These must exist before `Image(...)` runs, since a freshly created
        # Image plots itself right away, which calls back into
        # `_attach_side_panel()` (see Image.plot_image()).
        self.signals = WorkspaceSignals()
        self.lens_model = None
        self.catalog = None
        self.catalogs = []
        self.side_panel = None
        self.toggle_panel_btn = None
        self.image = None
        self.image = Image(image_path, main_window=main_window, workspace=self)

    def import_image(self, image_path=None, main_window=None, plot_image=True, wcs=(0, 0)) :
        if main_window is None and self.image is not None :
            main_window = getattr(self.image, 'QMainWindow', None)
        self.image = Image(image_path, main_window=main_window, plot_image=plot_image, wcs=wcs, workspace=self)
        self._attach_side_panel()
        self.signals.image_changed.emit()
        return self.image

    def main_window_is_open(self) :
        """Return whether the main image viewer window is currently open."""
        img = self.image
        if img is None or img.ImageView is None :
            return False
        main_window = getattr(img, 'QMainWindow', None)
        if main_window is not None and not main_window.isVisible() :
            return False
        return img.ImageView.isVisible()

    def reopen_main_window(self) :
        """Show or recreate the main image viewer window.

        If the main viewer was closed, this rebuilds it (image, side panel,
        toggles) and re-applies any loaded catalog and lens-model overlays.
        If it is already open, this brings it to the front.
        """
        if self.image is None :
            raise ValueError('No image loaded; import an image first.')

        was_closed = not self.main_window_is_open()
        self.image.plot_image()
        if was_closed :
            if self.catalog is not None :
                self.catalog.plot()
            if self.lens_model is not None :
                self.lens_model.plot()
        elif self.image.QMainWindow is not None :
            self.image.QMainWindow.show()
            self.image.QMainWindow.raise_()
            self.image.QMainWindow.activateWindow()
        return self.image.QMainWindow

    def plot_secondary_image(self) :
        """Open a new, independent window showing only the current image (no left panel/toggle).

        The new window is registered alongside the main one so that
        ``link_zoom()``/``link_pan()`` (called on ``self.image``) keep it in sync.
        """
        img = self.image
        if img is None :
            raise ValueError('No image loaded; import an image first.')

        img.plot_image()  # make sure there is a main window for this to be secondary *to*

        print('Creating secondary window...')
        extra_image_view, _, _, extra_window = img.create_QT_instances(force_new_window=True)
        extra_window.setWindowTitle(img._window_title() + ' (' + str(len(img.ImageView_list) + 1) + ')')
        extra_window.show()
        img.ImageView_list.append(extra_image_view)
        img.QMainWindow_list.append(extra_window)
        print('Done')

        if hasattr(img, '_view_range_sync') and (img._view_range_sync['zoom'] or img._view_range_sync['pan']) :
            img._connect_view_range_sync()
        img._connect_hand_select_sync()

        return extra_image_view, extra_window

    def toggle_split4(self) :
        """Toggle between a single main viewer and a 2x2 grid of four views.

        When splitting, the top-left ``ImageView`` stays ``self.image.ImageView``
        (the one ``Catalog``/``LensModel`` instances are linked to); the three
        other panes are secondary views. The histogram side bar is hidden on all
        four panes to save space. When unsplitting, the three extra panes are
        removed and the main view is restored as the only pane in the window.
        """
        img = self.image
        if img is None :
            raise ValueError('No image loaded; import an image first.')

        img.plot_image()  # make sure the main window/ImageView exists

        if getattr(img, '_is_split4', False) :
            return self._unsplit4(img)
        
        # If the side panel is opened, it will resize incorrectly when splitting.
        _opened_side_panel = self.side_panel.isVisible()
        if _opened_side_panel :
            self._toggle_side_panel(img.QSplitter, self.side_panel)

        main_view = img.ImageView
        main_drag_widget = img.QWidget
        img._set_right_side_bar_visible_for(main_view, False)

        idx = img.QSplitter.indexOf(main_drag_widget)

        extra_views, extra_drag_widgets = [], []
        for _ in range(3) :
            extra_view, extra_drag_widget, _ = img._create_image_view_and_drag_widget()
            img._set_right_side_bar_visible_for(extra_view, False)
            extra_views.append(extra_view)
            extra_drag_widgets.append(extra_drag_widget)

        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(2)
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (row, col), drag_widget in zip(positions, [main_drag_widget] + extra_drag_widgets) :
            grid_layout.addWidget(drag_widget, row, col)
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        img.QSplitter.insertWidget(idx if idx >= 0 else img.QSplitter.count(), grid_widget)

        img.ImageView_list = [img.ImageView_list[0]] + extra_views + img.ImageView_list[1:]
        img.QMainWindow_list = [img.QMainWindow_list[0]] + [img.QMainWindow] * 3 + img.QMainWindow_list[1:]

        img._split4_grid_widget = grid_widget
        img._split4_extra_views = extra_views
        img._is_split4 = True

        if hasattr(img, '_view_range_sync') and (img._view_range_sync['zoom'] or img._view_range_sync['pan']) :
            img._connect_view_range_sync()
        img._connect_hand_select_sync()
        
        # Now we can reopen the side panel if it was opened before splitting.
        if _opened_side_panel :
            self._toggle_side_panel(img.QSplitter, self.side_panel)

        return [main_view] + extra_views

    def _unsplit4(self, img) :
        """Restore a single main viewer after ``toggle_split4()`` split the window."""
        # If the side panel is opened, it will resize incorrectly when unsplitting.
        _opened_side_panel = self.side_panel.isVisible()
        if _opened_side_panel :
            self._toggle_side_panel(img.QSplitter, self.side_panel)
            
        grid_widget = img._split4_grid_widget
        extra_views = list(img._split4_extra_views)
        main_drag_widget = img.QWidget
        grid_layout = grid_widget.layout()

        grid_layout.removeWidget(main_drag_widget)
        main_drag_widget.setParent(None)

        for extra_view in extra_views :
            extra_drag_widget = extra_view.parentWidget()
            if extra_drag_widget is not None :
                grid_layout.removeWidget(extra_drag_widget)
                extra_drag_widget.deleteLater()

        idx = img.QSplitter.indexOf(grid_widget)
        img.QSplitter.insertWidget(idx if idx >= 0 else img.QSplitter.count(), main_drag_widget)

        grid_widget.setParent(None)
        grid_widget.deleteLater()

        img._set_right_side_bar_visible_for(img.ImageView, True)

        img.ImageView_list = [img.ImageView_list[0]] + img.ImageView_list[4:]
        img.QMainWindow_list = [img.QMainWindow_list[0]] + img.QMainWindow_list[4:]

        img._split4_grid_widget = None
        img._split4_extra_views = []
        img._is_split4 = False

        if hasattr(img, '_view_range_sync') and (img._view_range_sync['zoom'] or img._view_range_sync['pan']) :
            img._connect_view_range_sync()
        img._connect_hand_select_sync()

        # Now we can reopen the side panel if it was opened before splitting.
        if _opened_side_panel :
            self._toggle_side_panel(img.QSplitter, self.side_panel)

        return [img.ImageView]

    def import_catalog(self, cat, color=None, units='pixel') :
        if self.image is None :
            new_cat = Catalog(cat, color=color, units=units, workspace=self)
            self.image = new_cat.image   # adopt the auto-created empty Image
        else :
            new_cat = self.image.make_catalog(cat, color=color, units=units)
            new_cat.workspace = self
        self.catalog = new_cat
        self.catalogs.append(new_cat)
        self._attach_side_panel()
        self.signals.catalogs_changed.emit()
        return new_cat

    def import_lens_model(self, model_dir, compute_predictions=True, verbose=True, use_best=False) :
        self.lens_model = LensModel(model_dir, self.image, workspace=self, compute_predictions=compute_predictions, verbose=verbose, use_best=use_best)
        self.image = self.lens_model.image   # adopt the auto-created empty Image if there wasn't one
        self._attach_side_panel()
        self.lens_model.clear()
        self.lens_model.plot()
        main_window = getattr(self.image, 'QMainWindow', None)
        if main_window is not None :
            main_window.setWindowTitle(self.image._window_title())
        self.signals.lens_model_changed.emit()
        return self.lens_model

    def open_image_dialog(self) :
        path_str, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(),
            'Open FITS file',
            '',
            'FITS images (*.fits *.fit *.fits.gz *.fit.gz);;All files (*)',
        )
        if not path_str :
            return
        try :
            main_window = getattr(self.image, 'QMainWindow', None) if self.image is not None else None
            self.import_image(path_str, main_window=main_window)
        except Exception as exc :
            QMessageBox.critical(self._dialog_parent(), 'Failed to load FITS', str(exc))

    def open_catalog_dialog(self) :
        path_str, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(),
            'Open catalog (FITS or ASCII)',
            '',
            'Catalog files (*.fits *.cat *.txt *.csv);;All files (*)',
        )
        if not path_str :
            return
        try :
            self.import_catalog(path_str)
            if self.catalog is not None :
                self.catalog.plot()
        except Exception as exc :
            QMessageBox.critical(self._dialog_parent(), 'Failed to import catalog', str(exc))

    def open_lens_model_dialog(self) :
        dir_path = QFileDialog.getExistingDirectory(
            self._dialog_parent(),
            'Select Lenstool model directory',
        )
        if not dir_path :
            return
        try :
            self.import_lens_model(dir_path)
        except Exception as exc :
            QMessageBox.critical(self._dialog_parent(), 'Failed to import model', str(exc))

    def _dialog_parent(self) :
        if self.image is not None and getattr(self.image, 'QMainWindow', None) is not None :
            return self.image.QMainWindow
        if self.side_panel is not None :
            return self.side_panel
        return None

    # ------------------------------------------------------------------
    # UI shell (left side panel + toggle button)
    # ------------------------------------------------------------------
    def _attach_side_panel(self, image=None) :
        """Attach or re-attach the workspace side panel to the current image window."""
        img = image if image is not None else self.image
        if img is None or img.QSplitter is None or img.QWidget is None :
            return

        # Closing the window destroys its splitter and, with it, the panel and toggle button.
        if self.side_panel is not None and sip.isdeleted(self.side_panel) :
            self.side_panel = None
        if self.toggle_panel_btn is not None and sip.isdeleted(self.toggle_panel_btn) :
            self.toggle_panel_btn = None

        panel_is_new = self.side_panel is None
        if panel_is_new :
            self.side_panel = ImageSidePanel(self)
            self.side_panel.hide()

        if self.side_panel.parent() is not None and self.side_panel.parent() is not img.QSplitter :
            self.side_panel.setParent(None)

        moved_to_new_window = self.side_panel.parent() is not img.QSplitter
        if moved_to_new_window :
            img.QSplitter.insertWidget(0, self.side_panel)
            img.QSplitter.setStretchFactor(0, 1)
            img.QSplitter.setStretchFactor(1, 3)
            if not panel_is_new :
                self.signals.window_attached.emit()

        if self.toggle_panel_btn is not None :
            parent = self.toggle_panel_btn.parent()
            if parent is not None and hasattr(parent, 'remove_floating_widget') :
                parent.remove_floating_widget(self.toggle_panel_btn)
            self.toggle_panel_btn.deleteLater()
        self.toggle_panel_btn = self._make_panel_toggle_button(img.QWidget, img.QSplitter, self.side_panel)

    def _make_floating_toggle_button(self, drag_widget, label, corner, callback, margin=8, x_margin=None, y_margin=None) :
        toggle_btn = QPushButton(label)
        toggle_btn.setFixedSize(28, 28)
        toggle_btn.setCursor(Qt.PointingHandCursor)
        toggle_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: rgba(40, 40, 40, 160);"
            "  color: white;"
            "  border: none;"
            "  border-radius: 4px;"
            "  font-size: 14px;"
            "}"
            "QPushButton:hover { background-color: rgba(70, 70, 70, 200); }"
        )
        toggle_btn.clicked.connect(callback)
        drag_widget.add_floating_widget(toggle_btn, corner=corner, margin=margin, x_margin=x_margin, y_margin=y_margin)
        return toggle_btn

    def _toggle_side_panel(self, splitter, side_panel) :
        will_show = not side_panel.isVisible()
        side_panel.setVisible(will_show)
        if will_show :
            total_width = splitter.width()
            if total_width > 0 :
                splitter.setSizes([total_width // 4, 3 * total_width // 4])

    def _make_panel_toggle_button(self, drag_widget, splitter, side_panel) :
        return self._make_floating_toggle_button(
            drag_widget,
            '\u2630',  # hamburger icon
            'top-left',
            lambda : self._toggle_side_panel(splitter, side_panel),
        )
