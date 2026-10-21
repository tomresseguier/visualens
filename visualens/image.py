import os
import re
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Circle, Rectangle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u
import astropy.constants as c
from reproject import reproject_interp
#from astropy.visualization.wcsaxes import *
from astropy.coordinates import SkyCoord
from tqdm import tqdm

import PyQt5
from PyQt5.QtWidgets import QMainWindow, QSplitter, QPushButton
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from astropy.table import Table

from .catalog import Catalog
from .source_extraction.source_extract import source_extract, source_extract_DIM
from .utils.utils_fits.utils_fits_image import open_image, make_default_wcs, create_empty_image
from .utils.utils_general.utils_general import derived_path, hst_wht_fits_path
from .utils.utils_plots.plot_utils_general import *
from .utils.utils_Qt.selectable_classes import *
from .utils.utils_Qt.Special_QWidgets import DragWidget
from .utils.utils_Qt.utils_general import *
from .utils.utils_astro.get_default_cosmology import get_default_cosmology
from .utils.utils_Qt.Special_ImageViews import ImageView_custom_selector




pg.setConfigOption('imageAxisOrder', 'row-major')

DEFAULT_MAIN_WINDOW_WIDTH = 1200
DEFAULT_MAIN_WINDOW_HEIGHT = 900


class Image :
    def __init__(self, image_path=None, main_window=None, plot_image=True, wcs=(0, 0), workspace=None) :
        """
        Parameters
        ----------
        image_path : str or None
            Path to a FITS file. If ``None``, an empty placeholder image is
            created instead (see ``wcs`` below), which can be useful e.g. to
            open a blank canvas for hand-selecting a catalog.
        main_window : QMainWindow, optional
            External window to reuse instead of spawning a new one.
        plot_image : bool
            Whether to display the image right away.
        wcs : astropy.wcs.WCS or (ra, dec) tuple/list
            Only used when ``image_path`` is ``None``. Either a ready-made
            WCS object, or a ``(ra, dec)`` pair (in degrees) used to build a
            default, non-rotated WCS for a 10 arcmin square field centered
            on those coordinates, with the x axis aligned on RA and the y
            axis aligned on Dec.
        workspace : Visualens, optional
            Owning workspace, if any. When set, ``plot_image()`` uses it to
            rebuild the left side panel and its toggle button whenever the
            main window has to be recreated from scratch.
        """
        self.image_path = image_path
        # If an external QMainWindow is provided (e.g. from the GUI), use it
        # instead of spawning a new independent window.
        self.main_window = main_window  # type: ignore[assignment]
        self.workspace = workspace

        if self.image_path is not None :
            weight_path = hst_wht_fits_path(self.image_path)
            if weight_path is not None and os.path.isfile(weight_path) :
                print("Weight file found: " + weight_path)
                self.weight_path = weight_path
            else :
                self.weight_path = None

            self.image_data, self.pix_deg_scale, self.orientation, self.wcs, self.header = open_image(self.image_path)

            self.boosted_image_path = derived_path(self.image_path, '_boosted')
            if os.path.isfile(self.boosted_image_path) :
                print("Boosted image found: " + self.boosted_image_path)
                self.boosted_image, _, _, _, _ = open_image(self.boosted_image_path)
            else :
                self.boosted_image = None
        else :
            # No FITS file provided: build an empty placeholder image instead.
            self.weight_path = None
            wcs_obj = wcs if isinstance(wcs, WCS) else make_default_wcs(wcs[0], wcs[1])
            self.image_data, self.pix_deg_scale, self.orientation, self.wcs, self.header = create_empty_image(wcs_obj)
            self.boosted_image_path = None
            self.boosted_image = None

        self.sources = None
        self.fig = None
        self.ax = None
        self.multiple_images = None
        self.galaxy_selection = None
        self.ImageView = None
        self.toggle_histogram_btn = None
        self.qtItems_dict = {'sources': None,
                             'potfile_cat': None,
                             'imported_cat': None,
                             'multiple_images': None}
        self.ax = None
        self.redshift = None
        self.boosted = False
        self.ImageView_list = []
        self.QMainWindow_list = []
        self._is_split4 = False
        self._split4_grid_widget = None
        self._split4_extra_views = []
        if self.orientation==None :
            self.orientation = 0.
        self.cosmo = get_default_cosmology()

        if plot_image :
            self.plot_image()
        self.filters = None

    def _window_title(self) :
        lens_model = getattr(self.workspace, 'lens_model', None) if self.workspace is not None else None
        if lens_model is not None and getattr(lens_model, 'model_dir', None) :
            return os.path.basename(os.path.normpath(lens_model.model_dir))
        return os.path.basename(self.image_path) if self.image_path is not None else 'Empty image'
    
    
    def set_cosmo(self, cosmo_name) :
        self.cosmo = get_default_cosmology(cosmo_name)
    
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

    def _set_right_side_bar_visible_for(self, image_view, visible) :
        if visible :
            image_view.ui.histogram.show()
            image_view.ui.roiBtn.show()
            image_view.ui.menuBtn.show()
        else :
            image_view.ui.histogram.hide()
            image_view.ui.roiBtn.hide()
            image_view.ui.menuBtn.hide()

    def _toggle_right_side_bar_for(self, image_view) :
        self._set_right_side_bar_visible_for(image_view, not image_view.ui.histogram.isVisible())

    def _make_histogram_toggle_button(self, drag_widget, image_view) :
        return self._make_floating_toggle_button(
            drag_widget,
            '<',
            'top-right',
            lambda : self._toggle_right_side_bar_for(image_view),
            x_margin=40,
            y_margin=8,
        )

    def _create_image_view_and_drag_widget(self) :
        """Build a single (ImageView, DragWidget) pair showing the current image.

        This is the reusable building block behind ``create_QT_instances()``: it
        does not create any window, so it can be used to populate secondary
        windows or extra panes (e.g. ``toggle_split4()``) sharing the main
        window. The returned ``ImageView`` starts out with its own,
        independent hand-selection state; call ``_connect_hand_select_sync()``
        (done automatically by ``plot_image()``/``toggle_split4()``/
        ``plot_secondary_image()``) to link it with the other open views so
        they all feed into the same ``hand_selected_catalog``.
        """
        to_plot = np.flip(self.image_data, axis=0) if not self.boosted else np.flip(self.boosted_image, axis=0)

        image_view = ImageView_custom_selector(self.wcs)
        image_view.setImage(to_plot)

        drag_widget = DragWidget(image_view)

        histogram_toggle_btn = self._make_histogram_toggle_button(drag_widget, image_view)
        image_view.toggle_histogram_btn = histogram_toggle_btn
        drag_widget.toggle_histogram_btn = histogram_toggle_btn

        return image_view, drag_widget, histogram_toggle_btn

    def create_QT_instances(self, force_new_window=False) :
        image_view, drag_widget, _ = self._create_image_view_and_drag_widget()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(drag_widget)

        if not force_new_window and self.main_window is not None :
            # Reuse the provided main window.
            main_window = self.main_window
            # Replace any existing central widget.
            main_window.setCentralWidget(splitter)
            main_window.setWindowTitle(self._window_title())
            # Ensure the window is visible (may already be shown). Note that
            # if the window was already visible, `main_window.show()` is a
            # no-op and will NOT recursively show the just-added central
            # widget (Qt only propagates visibility down on the transition
            # to visible), so the new widget subtree must be shown explicitly.
            main_window.show()
            splitter.show()
        else :
            main_window = QMainWindow()
            main_window.setWindowTitle(self._window_title())
            main_window.resize(DEFAULT_MAIN_WINDOW_WIDTH, DEFAULT_MAIN_WINDOW_HEIGHT)
            main_window.setCentralWidget(splitter)
            main_window.show()

        return image_view, splitter, drag_widget, main_window
    
    def _flush_closed_windows(self) :
        if self.ImageView is not None :
            if not self.ImageView.isVisible() :
                self.QMainWindow.close()
                self.ImageView, self.QSplitter, self.QWidget, self.QMainWindow = None, None, None, None
                self.toggle_histogram_btn = None
        for window in self.QMainWindow_list :
            if not window.isVisible() :
                window.close()
        self.ImageView_list = [ image for window, image in zip(self.QMainWindow_list, self.ImageView_list) if window.isVisible() ]
        self.QMainWindow_list = [ window for window in self.QMainWindow_list if window.isVisible() ]
        return

    def link_zoom(self) :
        self._view_range_sync_state()['zoom'] = True
        self._connect_view_range_sync()

    def unlink_zoom(self) :
        self._view_range_sync_state()['zoom'] = False
        self._disconnect_view_range_sync_if_idle()

    def link_pan(self) :
        self._view_range_sync_state()['pan'] = True
        self._connect_view_range_sync()

    def unlink_pan(self) :
        self._view_range_sync_state()['pan'] = False
        self._disconnect_view_range_sync_if_idle()

    def link_zoom_and_pan(self) :
        sync = self._view_range_sync_state()
        sync['zoom'] = True
        sync['pan'] = True
        self._connect_view_range_sync()

    def unlink_zoom_and_pan(self) :
        sync = self._view_range_sync_state()
        sync['zoom'] = False
        sync['pan'] = False
        self._disconnect_view_range_sync_if_idle()

    def toggle_right_side_bar(self) :
        if self.ImageView is not None :
            self._toggle_right_side_bar_for(self.ImageView)

    def plot_image(self) :
        """Ensure the main image window is open.

        If the main window is still open, this does nothing. Otherwise, the
        whole main viewer is recreated from scratch: the image, its drag
        widget/histogram toggle, and (if this ``Image`` belongs to a
        ``Visualens`` workspace) the left side panel and its toggle button.
        Secondary windows/panes are created via
        ``Visualens.plot_secondary_image()`` and ``Visualens.toggle_split4()``, not
        by calling this again.
        """
        self._flush_closed_windows()
        if self.ImageView is not None :
            return

        print('Creating main window...')
        self.ImageView, self.QSplitter, self.QWidget, self.QMainWindow = self.create_QT_instances()
        self.toggle_histogram_btn = self.ImageView.toggle_histogram_btn
        self.QMainWindow_list = [self.QMainWindow] + self.QMainWindow_list
        self.ImageView_list = [self.ImageView] + self.ImageView_list
        self._is_split4 = False
        self._split4_grid_widget = None
        self._split4_extra_views = []
        print('Done')

        if self.workspace is not None :
            self.workspace._attach_side_panel(self)

        if hasattr(self, '_view_range_sync') and (self._view_range_sync['zoom'] or self._view_range_sync['pan']) :
            self._connect_view_range_sync()
        self._connect_hand_select_sync()
        return
    
    def boost(self, boost=[2,1.5,1]) :
        if self.image_path is None :
            raise ValueError('boost() requires a real FITS image; this Image instance was created without one.')
        if self.boosted_image is None :
            print('Adjusting contrast...')
            adjusted_image = adjust_contrast(self.image_data, boost[0], pivot=boost[1])
            print('Adjusting luminosity...')
            self.boosted_image = adjust_luminosity(adjusted_image, boost[2])
            print('Writing to memory...')
            hdul = fits.HDUList([fits.PrimaryHDU()] + [ fits.ImageHDU(data=self.boosted_image[:,:,i], header=self.wcs.to_header(), name=name) for i, name in enumerate(['RED','GREEN','BLUE']) ])
            hdul.writeto(self.boosted_image_path, overwrite=True)
        if not self.boosted :
            print('Plotting...')
            self.ImageView.setImage(np.flip(self.boosted_image, axis=0))
            #self.ImageView.autoLevels()
            self.boosted = True
            for extra_ImageView in self.ImageView_list :
                extra_ImageView.setImage(np.flip(self.boosted_image, axis=0))
            print('Done')
    
    def unboost(self) :
        if self.boosted :
            print('Plotting...')
            self.ImageView.setImage(np.flip(self.image_data, axis=0))
            #self.ImageView.autoLevels()
            self.boosted = False
            for extra_ImageView in self.ImageView_list :
                extra_ImageView.setImage(np.flip(self.image_data, axis=0))
            print('Done')
    
    def set_weight(self, weight_path) :
        self.weight_path = weight_path
    
    def extract_sources(self, image_path=None, weight_path=None, DIM_ref_path=None, rerun=False, reproject=True) :
        print("Source extraction not yet available. Work in progress.")
        if False :
            if image_path is None :
                if self.image_path is None :
                    raise ValueError('extract_sources() requires a real FITS image; this Image instance was created without one.')
                image_path = self.image_path
                weight_path = self.weight_path
                
            out_dir = os.path.join(os.path.dirname(image_path), 'source_extraction')
            if not os.path.exists(out_dir) :
                os.makedirs(out_dir, exist_ok=True)
            
            outfile_name = 'SExtractor_cat.fits' if DIM_ref_path is None else 'SExtractor_cat_DIM.fits'
            out_path = os.path.join( out_dir, outfile_name )
            if os.path.isfile(out_path) and not rerun :
                print('Previous SExtractor catalog found.')
                with fits.open(out_path) as hdu :
                    self.sources_all = Table(hdu[1].data)
            elif DIM_ref_path is None :
                self.sources_all = source_extract(image_path, weight_path=weight_path, pixel_scale=self.pix_deg_scale*3600, zero_point=None, out_dir=out_dir,
                                                  outfile_name=outfile_name, return_sources=True)
            else :
                if type(DIM_ref_path) is list :
                    if reproject :
                        reprojected_image_path = self.reproject(DIM_ref_path[0], image_path)
                        reprojected_weight_path = self.reproject(DIM_ref_path[0], weight_path)
                        reprojected_image_path = [reprojected_image_path, reprojected_weight_path]
                    else :
                        reprojected_image_path = [image_path, weight_path]
                else :
                    if reproject :
                        reprojected_image_path = self.reproject(DIM_ref_path, image_path)
                    else :
                        reprojected_image_path = image_path
                
                self.sources_all = source_extract_DIM(DIM_ref_path, reprojected_image_path, pixel_scale=self.pix_deg_scale*3600, zero_point=None, out_dir=out_dir,
                                                      outfile_name=outfile_name, return_sources=True)
            
            # This next part doesn't make sense here as the purpose of extract_sources() is to extract sources from the imported image,
            # but the code should be added to import_cat()/make_catalog()
            # THE WAY TO CONVERT ANGLES IS MORE COMPLICATED!!!
            x, y = self.world_to_image(self.sources_all['RA'], self.sources_all['DEC'], unit='deg')
            if x[0] != self.sources_all['X_IMAGE'][0] :
                #print('Catalog sextracted from different image: replacing X_IMAGE, Y_IMAGE and THETA_IMAGE columns with current image coordinates.')
                #self.sources_all['X_IMAGE'], self.sources_all['Y_IMAGE'] = x, y
                print('Catalog SExtracted from different image: keeping X_IMAGE, Y_IMAGE and THETA_IMAGE and adding x, y, theta columns from current image coordinates.')
                self.sources_all.add_column(x, name='x')
                self.sources_all.add_column(y, name='y')
                ref_image_angle = ( np.arctan2(self.wcs.wcs.get_pc()[1, 0], self.wcs.wcs.get_pc()[0, 0]) %np.pi ) * 360/np.pi
                print('Overall angle of imported image: ' + str(ref_image_angle))
                #self.sources_all.add_column(self.sources_all['THETA_WORLD'] + ref_image_angle, name='prout')
            
            #mask_mag = self.sources_all['MAG_AUTO']<-10.
            #mask = self.sources['KRON_RADIUS']
            #mask_galstar = self.sources_all['CLASS_STAR']<0.4
            #mask_size = self.sources_all['A_IMAGE']*self.sources_all['B_IMAGE']*np.pi>1000.
            #mask = mask_mag & mask_galstar & mask_size
            
            #self.sources = self.sources_all #[mask]
            self.make_photometry(self.sources_all)
            self.sources = self.make_catalog(self.sources_all)
            return str(len(self.sources.cat)) + ' sources found.'
    
    def reproject(self, ref_image_path, image_path) :
        reprojected_image_path = derived_path(image_path, '_reprojected')
        if os.path.isfile(reprojected_image_path) :
            print('Previous reprojected image found.')
        else :
            with fits.open(ref_image_path) as hdu :
                reference_header = hdu['SCI'].header
            with fits.open(image_path) as hdu :
                print('Reprojecting image ' + image_path + ' onto reference ' + ref_image_path)
                reprojected_data, footprint = reproject_interp(hdu['SCI'], reference_header)
                fits.writeto(reprojected_image_path, reprojected_data, reference_header)
        return reprojected_image_path

    def rotate(self, angle=None) :
        """Reproject the image onto a new rotated pixel grid.

        The output is the smallest image that still contains all finite pixels
        of the input (``np.nan`` padding is dropped). Extra space required by
        the rotation is filled with ``np.nan``. Pixel scale is conserved and
        the WCS is updated so sky positions are unchanged.

        Parameters
        ----------
        angle : float or None
            Rotation angle in degrees. If ``None``, uses ``self.orientation``
            so the new x/y axes align with RA/Dec (North up, East left).
            The output orientation is ``self.orientation - angle``.
        """
        if self.image_path is None :
            raise ValueError('rotate() requires a real FITS image; this Image instance was created without one.')
        if angle is None :
            angle = self.orientation

        wcs_in = self.wcs.celestial
        data = np.asarray(self.image_data, dtype=float)
        is_rgb = data.ndim == 3

        if is_rgb :
            valid = np.any(np.isfinite(data), axis=2)
        else :
            valid = np.isfinite(data)
        if not np.any(valid) :
            raise ValueError('No finite pixels found to rotate.')

        ys, xs = np.where(valid)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())

        # Corners of the finite-content bounding box (pixel edges).
        xc = np.array([xmin - 0.5, xmax + 0.5, xmax + 0.5, xmin - 0.5])
        yc = np.array([ymin - 0.5, ymin - 0.5, ymax + 0.5, ymax + 0.5])
        corners = wcs_in.pixel_to_world(xc, yc)
        center = wcs_in.pixel_to_world(0.5 * (xmin + xmax), 0.5 * (ymin + ymax))

        scale = float(self.pix_deg_scale)
        theta = np.deg2rad(self.orientation - angle)

        # North-up / East-left CD, then rotated so ORIENTAT = orientation - angle.
        wcs_out = WCS(naxis=2)
        wcs_out.wcs.ctype = list(wcs_in.wcs.ctype)
        wcs_out.wcs.cunit = ['deg', 'deg']
        wcs_out.wcs.crval = [center.ra.deg, center.dec.deg]
        wcs_out.wcs.crpix = [1.0, 1.0]
        wcs_out.wcs.cd = np.array([
            [-scale * np.cos(theta), scale * np.sin(theta)],
            [ scale * np.sin(theta), scale * np.cos(theta)],
        ])

        xp, yp = skycoord_to_pixel(corners, wcs_out, origin=1)
        xmin_o, xmax_o = float(xp.min()), float(xp.max())
        ymin_o, ymax_o = float(yp.min()), float(yp.max())
        wcs_out.wcs.crpix = [(1.0 - xmin_o) + 0.5, (1.0 - ymin_o) + 0.5]
        shape_out = (int(round(ymax_o - ymin_o)), int(round(xmax_o - xmin_o)))

        print(f'Rotating image by {angle} deg (output orientation {self.orientation - angle} deg)...')

        def _reproject_plane(plane) :
            arr, footprint = reproject_interp((plane, wcs_in), wcs_out, shape_out=shape_out)
            arr = np.asarray(arr, dtype=float)
            arr[footprint == 0] = np.nan
            return arr

        if is_rgb :
            rotated = np.stack([_reproject_plane(data[:, :, i]) for i in range(data.shape[2])], axis=2)
            valid_out = np.any(np.isfinite(rotated), axis=2)
        else :
            rotated = _reproject_plane(data)
            valid_out = np.isfinite(rotated)

        # Trim residual all-NaN borders left by interpolation.
        ys_o, xs_o = np.where(valid_out)
        x0, x1 = int(xs_o.min()), int(xs_o.max()) + 1
        y0, y1 = int(ys_o.min()), int(ys_o.max()) + 1
        if is_rgb :
            rotated = rotated[y0:y1, x0:x1, :]
        else :
            rotated = rotated[y0:y1, x0:x1]
        wcs_out.wcs.crpix[0] -= x0
        wcs_out.wcs.crpix[1] -= y0

        header_out = wcs_out.to_header()
        for key in self.header.keys() :
            if key in header_out or key in ('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                                            'EXTEND', 'COMMENT', 'HISTORY', 'BZERO', 'BSCALE') :
                continue
            if key.startswith(('CD', 'PC', 'CRPIX', 'CRVAL', 'CDELT', 'CTYPE', 'CUNIT',
                               'CROTA', 'PV', 'A_', 'B_', 'AP_', 'BP_')) :
                continue
            try :
                header_out[key] = self.header[key]
            except ValueError :
                pass

        out_path = derived_path(self.image_path, '_rotated')
        if is_rgb :
            hdul = fits.HDUList(
                [fits.PrimaryHDU()] +
                [fits.ImageHDU(data=rotated[:, :, i], header=header_out, name=name)
                 for i, name in enumerate(['RED', 'GREEN', 'BLUE'])]
            )
            hdul.writeto(out_path, overwrite=True)
        else :
            fits.writeto(out_path, rotated, header_out, overwrite=True)
        print('Rotated image saved to ' + out_path)

        new_orientation = self.orientation - angle
        self.image_path = out_path
        self.image_data = rotated
        self.wcs = wcs_out
        self.header = header_out
        self.orientation = new_orientation
        self.pix_deg_scale = float(np.sqrt(wcs_out.pixel_scale_matrix[0, 0]**2 +
                                           wcs_out.pixel_scale_matrix[0, 1]**2))
        self.boosted_image = None
        self.boosted = False
        self.boosted_image_path = derived_path(self.image_path, '_boosted')

        to_plot = np.flip(self.image_data, axis=0)
        if self.ImageView is not None :
            self.ImageView.setImage(to_plot)
            for ImageView in self.ImageView_list :
                ImageView.setImage(to_plot)
            if getattr(self, 'QMainWindow', None) is not None :
                self.QMainWindow.setWindowTitle(os.path.basename(self.image_path))
        print('Done')
        return out_path
    
    def make_photometry(self, cat) :
        print("################")
        print("Figuring out the photometry:")
        print("Assuming instrument HST/ACS")
        print("PHOTFLAM = " + str(self.header['PHOTFLAM']))
        print("Pivot wavelength = " + str(self.header['PHOTPLAM']))
        print("################")
        flux_lambda = self.header['PHOTFLAM'] * cat['FLUX_ISO'] * u.erg/u.cm**2/u.s/u.AA #FLUX_AUTO, FLUX_ISO, FLUX_APER
        magST = -2.5*np.log10(flux_lambda.value) - 21.1
        pivot_wavelength = self.header['PHOTPLAM'] * u.AA
        flux_nu = flux_lambda.to(u.erg/u.cm**2/u.s/u.Hz, u.spectral_density(pivot_wavelength))
        magAB = -2.5*np.log10(flux_nu.value) - 48.6
        if 'FILTER2' in self.header.keys() :
            print("filter " + self.header['FILTER2'] + " found")
            cat.add_column(magAB, name='magAB_' + self.header['FILTER2'])
            cat.add_column(magST, name='magST_' + self.header['FILTER2'])
        else :
            cat.add_column(magAB, name='magAB')
            cat.add_column(magST, name='magST')
        return 'Magnitudes calculated'
    
    
    def make_catalog(self, cat, color=[1., 1., 0], units=None, verbose=True) :
        if self.ImageView is None :
            self.plot_image()
        #to_return = Catalog(cat, self.image_data, self.wcs, self.ImageView, QMainWindow=self.QMainWindow, image_path=self.image_path, 
        #                            QWidget = self.QWidget, QSplitter=self.QSplitter, color=color, 
        #                            mag_colnames=mag_colnames, mpl_fig=self.fig, mpl_ax=self.ax, 
        #                            pix_deg_scale=self.pix_deg_scale, units=units)
        to_return = Catalog(cat, self, color=color, units=units, verbose=verbose)
        return to_return
    
    ###########################################################################
    
    
    def world_to_image(self, ra, dec, unit='deg') :
        coord = SkyCoord(ra, dec, unit=unit)
        image_coord = WCS.world_to_pixel(self.wcs, coord)
        if len(image_coord[0].shape)==0 :
            image_coord = (image_coord[0]*1., image_coord[1]*1.)
        return image_coord
    
    def image_to_world(self, x, y, unit='deg') :
        world_coord = WCS.pixel_to_world(self.wcs, x, y)
        return world_coord.ra.deg, world_coord.dec.deg
    
    def clear_Items(self) :
        for key in self.qtItems_dict.keys() :
            if self.qtItems_dict[key] is not None :
                for i in tqdm( range(len(self.qtItems_dict[key])) ) :
                    self.ImageView.removeItem(self.qtItems_dict[key][i])
    

    @property
    def hand_selected_catalog(self) :
        """Catalog of sources hand-selected via double-click, shared across
        every currently linked ``ImageView_custom_selector`` (main window,
        secondary windows and split4 panes all feed into the same catalog).
        """
        if self.ImageView is None :
            return None
        return self.ImageView.catalog

    def export_hand_selection_as_mult_file(self, path=None) :
        if path is None :
            base_dir = os.path.dirname(self.image_path) if self.image_path is not None else os.getcwd()
            path = os.path.join(base_dir, 'hand_selected_catalog.lenstool')
        header = "#REFERENCE 0\n## id   RA      Dec        a         b         theta     z         mag\n"
        with open(path, 'w') as f :
            f.write(header)
            for row in self.hand_selected_catalog :
                f.write(f"{row['id']:<3}  {row['ra']:11.7f}  {row['dec']:11.7f}  0.25  0.25  0.0  0.0  0.0\n")
        print('Hand selected catalog exported to ' + path)

    def clear_hand_selection(self) :
        """Empty the shared hand-selected catalog and clear its plotted
        markers ('+' pending and 'o' confirmed) in every currently open,
        linked view of this image (main window, secondary windows and any
        split4 panes)."""
        self._flush_closed_windows()
        if self.ImageView is not None :
            self.ImageView.clear_selection()


    def start_hand_select(self) :
        cat_dict = {'id': [], 'ra': [], 'dec': [], 'x': [], 'y': [], 'a': [], 'b': [], 'theta': []}
        cat = Table(cat_dict)
        self.hand_made_cat = Catalog(cat, self, units='pixel')
        
        def mouse_clicked(evt):
            if evt.double():
                pos = evt.scenePos()
                if self.ImageView.getView().sceneBoundingRect().contains(pos):
                    mouse_point = self.ImageView.getView().mapSceneToView(pos)
                    x, y_flipped = mouse_point.x(), mouse_point.y()
                    x, y = x, self.image_data.shape[0] - y_flipped
                    ra, dec = self.image_to_world(x, y)
                    self.hand_made_cat.cat.add_row( {'id': [len(self.hand_made_cat.cat) + 1], 'ra': [ra], 'dec': [dec], 'x': [x], 'y': [y], 'a': [10], 'b': [10], 'theta': [0]} )
                    self.hand_made_cat.qtItems.append(PyQt5.QtWidgets.QGraphicsEllipseItem())
                    self.hand_made_cat.plot(color=[1,1,1,0])
                    
        self._doubleclick_connection = self.ImageView.scene.sigMouseClicked.connect(mouse_clicked)
        
        def keyPressEvent(event):
            #print('Hand selection stopped.')
            if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Space :
                if hasattr(self, '_doubleclick_connection'):
                    self.ImageView.scene.sigMouseClicked.disconnect(self._doubleclick_connection)
                    del self._doubleclick_connection
                self.hand_made_cat.clear()
                self.QMainWindow.keyPressEvent = self._original_keyPressEvent
                print('Hand selection stopped.')
        
        self._original_keyPressEvent = self.QMainWindow.keyPressEvent
        self.QMainWindow.keyPressEvent = keyPressEvent
    
    
    
    def plot_image_mpl(self, wcs_projection=True, units='pixel', pos=111, make_axes_labels=True, make_grid=True, crop=None, replace_image=True, extra_pad=None) :
        fig, ax = plot_image_mpl(self.image_data, wcs=self.wcs, wcs_projection=wcs_projection, units=units, pos=pos, \
                                 make_axes_labels=make_axes_labels, make_grid=make_grid, crop=crop, extra_pad=extra_pad)
        plot_NE_arrows(ax, self.wcs)
        if replace_image :
            self.fig, self.ax = fig, ax
        return fig, ax
    
    def plot_sub_region(self, ra, dec, size=3):
        """
        Plots a square region around given RA and Dec coordinates.

        Parameters:
        ra (float): Right Ascension of the center in degrees.
        dec (float): Declination of the center in degrees.
        size (float): Size of the square region in arcseconds (default is 10).
        """
        
        x_center, y_center = self.world_to_image(ra, dec, unit='deg')
                
        size_pix = int( size / (self.pix_deg_scale*3600) / 2 )
        
        fig, axs = plt.subplots(1,3)
        for i in range(len(x_center)) :
            x_min = int(x_center[i]) - size_pix
            x_max = int(x_center[i]) + size_pix
            y_min = int(y_center[i]) - size_pix
            y_max = int(y_center[i]) + size_pix
            region = self.image_data[y_min:y_max, x_min:x_max, :]
            axs[i].imshow(region, origin='lower')
            axs[i].axis('off')
            
        return fig, axs
    
    def load_filters(self, filter_dir=None):
        if filter_dir==None :
            if self.image_path is None :
                raise ValueError('load_filters() requires a real FITS image; this Image instance was created without one.')
            listdir = os.listdir( os.path.dirname(self.image_path) )
            listdir_lower = [ name.lower() for name in os.listdir( os.path.dirname(self.image_path) ) ]
            indices = np.where([ 'filter' in name for name in listdir_lower ])[0]
            filters = []
            if len(indices)>0 :
                for i in indices :
                    filter_dir = os.path.join(os.path.dirname(self.image_path), listdir[i])
                    print('Looking for filters in directory: ' + filter_dir)
                    f = load_filters(filter_dir)
                    filters.append(f)
                l = np.array( [ len(f) for f in filters ] )
                idx = np.argmax(l)
                self.filters = filters[idx]
                print( str(len(self.filters)) + ' filters loaded from directory: ' + listdir[indices[idx]] )
            else :
                raise ValueError('No filter directory found in image directory.')
        else :
            self.filters = load_filters(filter_dir)
    
    def add_scale_bar(self, z=None, unit='arcmin', length=1, color='white', linewidth=4, text_offset=0.01, position=['bottom', 'right']):
        """Add a floating scale bar that stays anchored in the chosen viewport corner.

        The bar is a UIGraphicsItem (pg.ScaleBar) so it lives in screen/viewport
        coordinates: it never pans with the image and its on-screen pixel width
        automatically updates as the user zooms in or out, always representing the
        same angular or physical scale.

        Parameters
        ----------
        z : float, optional
            Source redshift, required when *unit* is a physical length ('pc', 'kpc', 'Mpc').
        unit : str
            One of 'arcsec', 'arcmin', 'pc', 'kpc', 'Mpc'.
        length : float
            Desired scale-bar length in *unit*.
        color : str or tuple
            Bar and label colour accepted by pyqtgraph (e.g. 'white', (255,255,255)).
        linewidth : int
            Thickness of the bar rectangle in screen pixels.
        text_offset : float
            Unused legacy parameter (kept for API compatibility).
        position : list of str
            Two-element list with vertical ('top'/'bottom') and horizontal
            ('left'/'right') placement of the bar, e.g. ['bottom', 'left'].
        """
        print(self.cosmo)

        if unit == 'arcsec':
            length_deg = length / 3600.
            label = f'{length}"'
        elif unit == 'arcmin':
            length_deg = length / 60.
            label = f"{length}'"
        elif unit in ['pc', 'kpc', 'Mpc']:
            if z is None:
                raise ValueError("Redshift z must be provided for physical units.")
            ang_diam_dist = self.cosmo.angular_diameter_distance(z).to(unit).value
            length_rad = length / ang_diam_dist
            length_deg = np.rad2deg(length_rad)
            # Angular equivalent shown as a subtitle under the physical label.
            # Pick arcsec when the value is below 60, arcmin otherwise.
            length_arcsec = length_deg * 3600.
            if length_arcsec < 60.:
                ang_fmt = f'{length_arcsec:.1f}"' if length_arcsec < 10. else f'{length_arcsec:.0f}"'
            else:
                length_arcmin = length_deg * 60.
                ang_fmt = f"{length_arcmin:.1f}'" if length_arcmin < 10. else f"{length_arcmin:.0f}'"
            label = f'{length} {unit}\n{ang_fmt}'
        else:
            raise ValueError(f"Unknown unit: {unit}")

        # Bar length expressed in image pixels (the data coordinate unit of the ViewBox)
        bar_length_pix = length_deg / self.pix_deg_scale

        # Compute anchor parameters before any item manipulation
        # UIGraphicsItem.anchor(itemPos, parentPos, offset):
        #   itemPos  – which point of the bar item to pin (0=top/left, 1=bottom/right)
        #   parentPos – which point of the ViewBox to pin to (same convention)
        #   offset    – additional screen-pixel nudge (positive x → right, positive y → down)
        margin = 20  # screen pixels from the viewport edge
        vertical   = 'bottom' if 'bottom' in position else 'top'
        horizontal = 'left'   if 'left'   in position else 'right'
        vy = 1 if vertical   == 'bottom' else 0
        vx = 0 if horizontal == 'left'   else 1
        ox = margin  if horizontal == 'left'   else -margin
        oy = -margin if vertical   == 'bottom' else  margin

        # If a bar already exists, update it in-place rather than destroying it.
        # Destroying via setParentItem(None) leaves the bar's viewRangeChanged slot
        # still connected to the ViewBox; when the view later updates it fires on the
        # now-parentless item and raises "Cannot anchor; parent is not set."
        if hasattr(self, '_scale_bar') and self._scale_bar is not None:
            self._scale_bar.size   = bar_length_pix
            self._scale_bar._width = linewidth
            self._scale_bar.brush  = pg.mkBrush(color)
            self._scale_bar.pen    = pg.mkPen(None)
            self._scale_bar.text.setText(label)
            self._scale_bar.text.setColor(color)
            self._scale_bar.anchor((vx, vy), (vx, vy), offset=(ox, oy))
            self._scale_bar.update()
            return self._scale_bar

        view = self.ImageView.getView()

        # pg.ScaleBar is a UIGraphicsItem: it renders at a fixed screen position and
        # redraws its pixel width automatically whenever the ViewBox zoom changes so
        # that it always represents bar_length_pix data-space units.
        bar = pg.ScaleBar(
            size=bar_length_pix,
            width=linewidth,
            brush=pg.mkBrush(color),
            pen=pg.mkPen(None),   # no outline around the filled rectangle
            suffix='',
        )
        bar.text.setText(label)
        bar.text.setColor(color)
        bar.setParentItem(view)
        bar.anchor((vx, vy), (vx, vy), offset=(ox, oy))

        self._scale_bar = bar
        return bar
    




    ########## Hand-selection synchronization ##########
    def _connect_hand_select_sync(self) :
        """(Re-)link every currently open ``ImageView_custom_selector`` of this
        image (main window, secondary windows, split4 panes) so they share a
        single hand-selected catalog and marker set. Should be called
        whenever ``self.ImageView_list`` changes.
        """
        self._flush_closed_windows()
        if self.ImageView_list :
            ImageView_custom_selector.link(self.ImageView_list)

    ########## View range synchronization functions##########
    def _all_viewboxes(self) :
        self._flush_closed_windows()
        return [iv.getView() for iv in self.ImageView_list]

    def _view_range_sync_state(self) :
        if not hasattr(self, '_view_range_sync') :
            self._view_range_sync = {'zoom': False, 'pan': False, 'connections': {}, 'guard': False}
        return self._view_range_sync

    def _connect_view_range_sync(self) :
        sync = self._view_range_sync_state()
        vbs = self._all_viewboxes()
        for vb in list(sync['connections'].keys()) :
            if vb not in vbs :
                try :
                    vb.sigRangeChanged.disconnect(sync['connections'][vb])
                except (TypeError, RuntimeError) :
                    pass
                del sync['connections'][vb]
        for vb in vbs :
            if vb not in sync['connections'] :
                sync['connections'][vb] = vb.sigRangeChanged.connect(
                    lambda _vb=vb: self._sync_linked_view_range(_vb)
                )

    def _sync_linked_view_range(self, source_vb) :
        sync = self._view_range_sync_state()
        if sync['guard'] or (not sync['zoom'] and not sync['pan']) :
            return
        vbs = self._all_viewboxes()
        if len(vbs) < 2 :
            return
        sync['guard'] = True
        try :
            xrange, yrange = source_vb.viewRange()
            cx = 0.5 * (xrange[0] + xrange[1])
            cy = 0.5 * (yrange[0] + yrange[1])
            wx = xrange[1] - xrange[0]
            wy = yrange[1] - yrange[0]
            for vb in vbs :
                if vb is source_vb :
                    continue
                ox0, ox1 = vb.viewRange()[0]
                oy0, oy1 = vb.viewRange()[1]
                if sync['zoom'] and sync['pan'] :
                    new_x, new_y = xrange, yrange
                elif sync['zoom'] :
                    occx = 0.5 * (ox0 + ox1)
                    occy = 0.5 * (oy0 + oy1)
                    new_x = (occx - 0.5 * wx, occx + 0.5 * wx)
                    new_y = (occy - 0.5 * wy, occy + 0.5 * wy)
                else :
                    owx = ox1 - ox0
                    owy = oy1 - oy0
                    new_x = (cx - 0.5 * owx, cx + 0.5 * owx)
                    new_y = (cy - 0.5 * owy, cy + 0.5 * owy)
                vb.setRange(xRange=new_x, yRange=new_y, padding=0)
        finally :
            sync['guard'] = False

    def _disconnect_view_range_sync_if_idle(self) :
        sync = self._view_range_sync_state()
        if sync['zoom'] or sync['pan'] :
            return
        for vb, conn in list(sync['connections'].items()) :
            try :
                vb.sigRangeChanged.disconnect(conn)
            except (TypeError, RuntimeError) :
                pass
        sync['connections'].clear()






class FilterImage(Image) :
    def __init__(self, image_path, main_window=None, plot_image=False) :
        super().__init__(image_path, main_window=main_window, plot_image=plot_image)
        self.filter = self._get_filter_name()
        self.wavelength = self._get_pivot_wavelength()
        psf_path = self._get_psf_path()
        if psf_path is not None :
            self.psf = PSF(psf_path)
        else :
            self.psf = None
            
    def _get_filter_name(self):
        """Extract filter name from header or filename."""
        # Try header first
        for key in ['FILTER', 'FILTER1', 'FILTER2']:
            if key in self.header and isinstance(self.header[key], str):
                val = self.header[key].strip().upper()
                if re.match(r'F\d{3,4}[WMN]*', val):
                    return val

        # Try filename
        m = re.search(r'F\d{3,4}[WMN]*', os.path.basename(self.image_path).upper())
        if m:
            return m.group(0)

        print(f"Could not determine filter name for {os.path.basename(self.image_path)}.")
        return "UNKNOWN"
    
    def _get_pivot_wavelength(self):
        lam = None
        f_lower = self.filter.lower()
        
        for key in ['PHOTPLAM', 'PIVOTWL', 'WAVELENGTH']:
            if key in self.header:
                lam = self.header[key]
                break

        if lam is None:
            lam = np.nan
            print(f"Could not determine wavelength for {self.filter} ({self.image_path}).")

        return lam
    
    def _get_psf_path(self):
        # Search first in parent of image dir, then in image dir
        search_dirs = [
            os.path.dirname(os.path.dirname(self.image_path)),
            os.path.dirname(self.image_path)
        ]

        for base in search_dirs:
            if not os.path.isdir(base):
                continue
            try:
                listdir = os.listdir(base)
            except Exception:
                continue

            listdir_lower = [name.lower() for name in listdir]
            indices = [i for i, name in enumerate(listdir_lower) if 'psf' in name]
            if len(indices) == 0:
                continue

            i = indices[0]
            psf_dir = os.path.join(base, listdir[i])
            print('Looking for psf in directory: ' + psf_dir)
            possible_psf_files = []
            for f in glob.glob(os.path.join(psf_dir, '*.fits')):
                if self.filter.lower() in os.path.basename(f).lower():
                    possible_psf_files.append(f)
            if len(possible_psf_files) > 0:
                print(f"PSF file found for filter {self.filter}: {os.path.basename(possible_psf_files[0])}")
                return possible_psf_files[0]
            else:
                print(f"No PSF file found for filter {self.filter} in {psf_dir}.")

        return None


class PSF :
    def __init__(self, psf_path) :
        with fits.open(psf_path) as hdu :
            self.data = hdu[0].data
            self.wcs = WCS(hdu[0].header)
    
    def plot(self) :
        fig, ax = plt.subplots()
        ax.imshow( np.log(self.data) , origin='lower')
        

def load_filters(path):
    """
    Load all FITS files in `path` and return a dict {filter_name: FilterImage},
    ordered by increasing pivot wavelength.
    If multiple files correspond to the same filter, selects the one most likely
    to be the science image (e.g., containing 'SCI' in header or filename).
    """
    fits_files = [f for f in os.listdir(path) if f.lower().endswith('.fits')]
    filter_groups = {}

    # --- Group files by detected filter name ---
    for fname in fits_files:
        full_path = os.path.join(path, fname)
        try:
            # Just read header to find filter (faster)
            with fits.open(full_path) as hdul:
                hdr = hdul[0].header
                filt = None
                for key in ['FILTER', 'FILTER1', 'FILTER2']:
                    if key in hdr and isinstance(hdr[key], str):
                        val = hdr[key].strip().upper()
                        if re.match(r'F\d{3,4}[WMN]*', val):
                            filt = val
                            break
                if filt is None:
                    m = re.search(r'F\d{3,4}[WMN]*', fname.upper())
                    if m:
                        filt = m.group(0)
        except Exception:
            print(f"Could not read header for {fname}. Skipping.")
            continue

        if filt is None:
            continue

        filter_groups.setdefault(filt, []).append(full_path)

    # --- For each filter, pick best file if multiple ---
    selected_files = {}
    for filt, files in filter_groups.items():
        if len(files) == 1:
            selected_files[filt] = files[0]
        else:
            sci_candidates = []
            for f in files:
                try:
                    with fits.open(f) as hdul:
                        hdr = hdul[0].header
                        for key in ['EXTNAME', 'FILETYPE', 'IMAGETYP']:
                            if key in hdr and 'SCI' in str(hdr[key]).upper():
                                sci_candidates.append(f)
                                break
                    if 'SCI' in os.path.basename(f).upper():
                        sci_candidates.append(f)
                except Exception:
                    continue

            if sci_candidates:
                selected_files[filt] = sci_candidates[0]
                print(f"Selected {os.path.basename(sci_candidates[0])} for {filt}")
            else:
                print(f"Multiple files for {filt}, picking first arbitrarily.")
                selected_files[filt] = files[0]

    # --- Create FilterImage instances ---
    filters_list = [FilterImage(fpath, plot_image=False) for fpath in selected_files.values()]
    filters_list.sort(key=lambda f: np.inf if np.isnan(f.wavelength) else f.wavelength)

    return {f.filter: f for f in filters_list}



