import copy
import numpy as np
import os
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from pyqtgraph.Qt import QtWidgets
from tqdm import tqdm
import matplotlib.pyplot as plt

from lenstronomy.Data.pixel_grid import PixelGrid     
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.psf import PSF
#from lenstronomy.Plots import lens_plot
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util import util
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

from ...utils.utils_astro.utils_general import world_to_relative, relative_to_world
from ...utils.utils_general.sort_points import break_curves
from ...utils.utils_Qt.Special_QWidgets import DragPlotWidget_special, DragImagePlotWidget_special
from ...utils.utils_Qt.utils_general import transform_rectangle
from .event_filters import SourceFilter, ImageFilter
from .lenstronomy_model import lenstronomy_model
from .make_lm_dict import make_lm_dict_opt
from .simulate import simulate, _update_curves
from .optimize import optimize
from .lm_initialize_functions import get_ROI_param_dict_light_model, get_ROI_slider_init_dict_light_model, get_ROI_param_dict_lens_model, get_ROI_slider_init_dict_lens_model
from .make_samples_dict_utils import _load_or_init_samples_dict, _sample_is_computed, _merge_and_dump_samples_dict



_DEFAULT_LENSING_COLUMNS = [
    'magnification',
    'convergence',
    'shear',
    'gamma1',
    'gamma2',
    'time',
    'tangential_magnification',
    'radial_magnification',
    'ra_source',
    'dec_source',
]


class image_simulator :
    def __init__(self, image, lens_model, which_filter=None, throttle_mode=0, use_linear_solver=False, dpl_resolution=200) :
        self.image = image
        self.lensmodel = lens_model
        self._DEFAULT_LENSING_COLUMNS = _DEFAULT_LENSING_COLUMNS
        self.z_source = lens_model.z_source
        self.ROI = image.QWidget.current_ROI
        #self.image.ImageView.addItem(ROI)
        self.models=None
        self.use_linear_solver = use_linear_solver
        
        if not hasattr(self, 'previous_state_current_ROI') :
            self.previous_state_current_ROI = None
        
        ######### Get the ROI's coordinates #########
        x0 = self.ROI.getState()['pos'][0]
        y0 = self.ROI.getState()['pos'][1]
        a = self.ROI.getState()['size'][0]
        b = self.ROI.getState()['size'][1]
        angle = self.ROI.getState()['angle'] *np.pi/180
        
        x0, y0, a, b, angle = transform_rectangle(x0, y0, a, b, angle) #x0, y0 at the top left, a and b > 0
        y0 = image.image_data.shape[0] - y0 # Counting pixels from bottom instead of top

        # Here we transform the ROI's parameters into a square area with corners corresponding to integer pixel coordinates.
        x_center, y_center = x0 + a/2, y0 - b/2
        square_size = max(a, b)
        square_size_floor = int(square_size)
        _x, _y = abs(x_center - round(x_center)), abs(y_center - round(y_center))
        if _x + _y > 0.5 : # The center will be the center of a pixel --> odd number of pixels
            rule = 'odd'
            x_center_pix, y_center_pix = int(x_center) + 0.5, int(y_center) + 0.5
            square_size_pix = square_size_floor + 1 if square_size_floor%2 == 0 else square_size_floor # odd number
        else : # The center will be the corner of a pixel --> even number of pixels
            rule = 'even'
            x_center_pix, y_center_pix = round(x_center), round(y_center)
            square_size_pix = square_size_floor if square_size_floor%2 == 0 else square_size_floor + 1 # even number
        
        # Store the exact pixel-crop origin used for image_data so interactive clicks
        # can be mapped back to full-image coordinates without sub-pixel offsets.
        self._crop_npix = square_size_pix
        self._crop_x0 = int( x_center_pix - square_size_pix/2 ) # x_center_pix - square_size_pix/2 is float with 0 decimal
        self._crop_y0 = int( y_center_pix + square_size_pix/2 )
        self.anchor = (self._crop_x0, self._crop_y0)
        self.center_world = image.image_to_world(x_center_pix, y_center_pix)

        xr_center, yr_center = lens_model.world_to_relative(self.center_world[0], self.center_world[1])
        self._square_size_arcsec = square_size_pix * image.pix_deg_scale*3600
        self._SquareOfInterest_side_arcsec = self._square_size_arcsec
        self._SquareOfInterest_xr_bottomleft = xr_center - self._square_size_arcsec / 2
        self._SquareOfInterest_yr_bottomleft = yr_center - self._square_size_arcsec / 2
        new_field = [self._SquareOfInterest_xr_bottomleft, self._SquareOfInterest_xr_bottomleft + self._square_size_arcsec, 
                     self._SquareOfInterest_yr_bottomleft, self._SquareOfInterest_yr_bottomleft + self._square_size_arcsec]
        
        ######### Set Lenstool field corresponding to ROI's position and calculate the local displacement map #########
        if lens_model.lt is not None :
            self._initial_field = lens_model.lt.get_field([])
            lens_model.lt.set_field(new_field)
            if self.previous_state_current_ROI is None or self.previous_state_current_ROI != self.ROI.getState() or lens_model.dpl_maps[lens_model.z_source][0].shape[0] != dpl_resolution :
                lens_model.compute_lt_dpl(npix=dpl_resolution)
                lens_model.compute_lt_convergence(npix=dpl_resolution)
        else :
            if lens_model.z_source not in lens_model.dpl_maps :
                raise KeyError('Displacement maps for redshift {} have not been calculated yet.'.format(lens_model.z_source))
            else :
                print('Lenstool instance not initialized. Using saved displacement maps instead. \
                      Careful: saved maps only cover the area you selected if you previously computed them for this specific area!')
        
        #-------------- Lens model Lenstronomy definitions --------------#
        
        deltaPix = abs( lens_model.dpl_maps[lens_model.z_source][2].wcs.cdelt[0]*3600 )
        x_grid_interp, y_grid_interp = util.make_grid(dpl_resolution, deltaPix)
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        
        self.LensModel_kwargs = [{'grid_interp_x': x_axes,
                                  'grid_interp_y': y_axes,
                                  'f_': lens_model.convergence_maps[lens_model.z_source][0],
                                  'f_x': lens_model.dpl_maps[lens_model.z_source][0],
                                  'f_y': lens_model.dpl_maps[lens_model.z_source][1]}]
        # Preserved for ``make_lm_dict`` / ``_assemble_lens_model_kwargs``: INTERPOL dict only,
        # so ellipse ROIs on the image plane can append ``PJAFFE_ELLIPSE_POTENTIAL_Q_PHI`` components.
        self.LensModel_base_kwargs = copy.deepcopy(self.LensModel_kwargs[0])
        
        self.LensModel_list = ['INTERPOL']
        self.LensModel = LensModel(lens_model_list=self.LensModel_list, z_lens=self.lensmodel.z_lens, z_source=self.z_source)
        
        #m=4
        #f, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=False, sharey=False)
        #lens_plot.lens_model_plot(ax, lensModel=self.lens_model, kwargs_lens=kwargs_lens,
        #                          with_caustics=True, fast_caustic=True, coord_inverse=False, numPix=round(dpl_resolution/m), deltaPix=deltaPix*m)
        
        #-------------- Qt graphics definitions --------------#
        
        #This version of the code tends to freeze unfortunately
        self.source_plane_widget = DragPlotWidget_special(ROI_param_dict=get_ROI_param_dict_light_model(), slider_init_dict=get_ROI_slider_init_dict_light_model(), throttle_mode=throttle_mode)
        self.source_plane_widget.setTitle('Source plane')
        #source_plane_widget.setAspectLocked(lock=True, ratio=1)
        
        self._imsim_srcplane_layout = QSplitter(Qt.Vertical)
        self._imsim_implane_layout = QWidget()
        
        self._imsim_srcplane_layout.addWidget(self.source_plane_widget)
        
        self._imsim_layout = QSplitter(Qt.Horizontal)
        self._imsim_layout.addWidget(self._imsim_srcplane_layout)
        self._imsim_layout.addWidget(self._imsim_implane_layout)
        
        self.window = QMainWindow()
        self.window.setWindowTitle('Lenstronomy image simulator')
        self.window.setCentralWidget(self._imsim_layout)
        
        
        #-------------- Plot critical & caustic curves --------------#

        self.source_center_coordinates = self.LensModel.ray_shooting(0, 0, self.LensModel_kwargs)
        
        if lens_model.lt is not None and (self.previous_state_current_ROI is None or self.previous_state_current_ROI != self.ROI.getState()) :
            lens_model.compute_lt_curve()
        lens_model.plot_lt_curve(color=[0, 255, 0], which='caustic') #to create self.lt_curve_coords_image_sorted
        
        self.caustic_plot = pg.PlotDataItem()
        self.caustic_plot.setPen( color=[0,255,0,255], width=4.0001 )
        #coords = break_curves(lens_model.lt_caustic_coords_relative, distance_threshold=8.*image.pix_deg_scale*3600) #sort_points(self.lt_caustic_coords_relative, distance_threshold=1.0, angle_threshold=np.pi)
        #x = coords[0] - xr_center - self.source_center_coordinates[0]
        #y = coords[1] - yr_center - self.source_center_coordinates[1]

        self.source_plane_widget.addItem(self.caustic_plot)
        self.source_plane_widget.setXRange(-3, 3)
        self.source_plane_widget.setYRange(-3, 3)
        
        self._lt_curve_coords_relative_broken = break_curves(lens_model.lt_curve_coords_relative, distance_threshold=8.*image.pix_deg_scale*3600) #sort_points(self.lt_curve_coords_relative, distance_threshold=1.0, angle_threshold=np.pi)
        
        self.filter_source = SourceFilter(self)
        self.source_plane_widget.installEventFilter(self.filter_source)
        

        self.source_coord_plot = pg.ScatterPlotItem(size=10, symbol='crosshair', brush=None, pen=(0,255,255))
        self.source_plane_widget.addItem(self.source_coord_plot)
        
        
        #-------------- Set the PixelGrid for lenstronomy --------------#
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * image.pix_deg_scale*3600
        
        npix = self._crop_npix
        self.PixelGrid_kwargs = {'nx': npix,
                                    'ny': npix,  # number of pixels per axis
                                    'ra_at_xy_0': - self._square_size_arcsec / 2,
                                    'dec_at_xy_0': - self._square_size_arcsec / 2,
                                    'transform_pix2angle': transform_pix2angle} 
        self.PixelGrid = PixelGrid(**self.PixelGrid_kwargs)
        
        
        #-------------- DATA --------------#
        if which_filter is not None :
            if which_filter in image.filters.keys() :
                print('Using filter ' + which_filter)
                self.individual_filter = image.filters[which_filter]
            elif type(which_filter) is int :
                if which_filter>2 or which_filter<0 :
                    raise('Filter index must belong to [0,1,2].')
                c = ['red', 'green', 'blue']
                which_filter_str = str(c[which_filter])
                print('Using ' + which_filter_str + ' image.')
                if image.filters is None :
                    image.filters = {}
                if which_filter_str not in image.filters :
                    exptime = image.header['EXPTIME'] if 'EXPTIME' in image.header else 1000.
                    self.individual_filter = filter_lite(image.image_data[:,:,which_filter], exptime=exptime)
                    image.filters[which_filter_str] = self.individual_filter
                else :
                    self.individual_filter = image.filters[which_filter_str]
            else :
                print('Filter ' + which_filter + " has not been imported. Import individual filters with image.load_filters(). Filter fits files have to be in a folder named 'filters' placed in the same directory as your main RGB image.")
                raise ValueError('Filter ' + which_filter + " has not been imported. Import individual filters with image.load_filters(). Filter fits files have to be in a folder named 'filters' placed in the same directory as your main RGB image.")
        elif image.filters is not None and not all(i in [0, 'red', 'green', 'blue'] for i in image.filters) :
            filter_list = list(image.filters.keys())
            filter_list[0] = [filter_list[0]]
            filter_list_str = str(filter_list)
            filter_list_str = filter_list_str.replace(", ", "/")
            filter_list_str = filter_list_str.replace("'", "")
            which_filter = input('Select filter: ' + filter_list_str)
            which_filter = filter_list[0][0] if which_filter=='' else which_filter
            print('Using filter ' + which_filter)
            self.individual_filter = image.filters[which_filter]
        else :
            print('\n------------------')
            print('\nUsing main image as individual filter. \nCareful: image simulator needs flux units when calculating the residuals. Make sure the main image has such units when running the optimization, or import individual filters separately with image.load_filters()')
            which_filter = 'red' if len(image.image_data.shape)==3 else 0
            if image.filters is not None :
                if which_filter in image.filters :
                    self.individual_filter = image.filters[which_filter]
            else :
                exptime = image.header['EXPTIME'] if 'EXPTIME' in image.header else 1000.
                self.individual_filter = filter_lite(image.image_data[:,:,0], exptime=exptime) if which_filter=='red'\
                                         else filter_lite(image.image_data, exptime=exptime)
                image.filters = {which_filter: self.individual_filter}
        
        #-------------- ImageData --------------#
        self.image_data = self.individual_filter.image_data[ self._crop_y0 - self._crop_npix : self._crop_y0, self._crop_x0 : self._crop_x0 + self._crop_npix ]
        
        #if not hasattr(self.individual_filter, 'rms') :
        #    print('Calculating RMS...')
        #    self.individual_filter.rms = np.std(self.individual_filter.image_data)
        #    print('done: ' + str(self.individual_filter.rms))
        #    if not hasattr(self.individual_filter, 'wcs') : # test if is instance of filter_lite instead of full filter class
        #        image.rms = self.individual_filter.rms
        if 'EXPTIME' not in self.individual_filter.header :
            print('\nCareful: EXPTIME not found in header. Using arbitrary value EXPTIME=1ks.')
            exptime = 1000.
        else :
            exptime = self.individual_filter.header['EXPTIME']
        
        self.rms = np.std(self.image_data)
        
        self.ImageData_kwargs = {'image_data': self.image_data,
                                'background_rms': self.rms, #self.individual_filter.rms
                                'exposure_time': exptime,
                                'transform_pix2angle': transform_pix2angle,
                                'ra_at_xy_0': - self._square_size_arcsec / 2,
                                'dec_at_xy_0': - self._square_size_arcsec / 2}
        self.ImageData = ImageData(**self.ImageData_kwargs)
        
        #-------------- PSF --------------#
        if self.individual_filter.psf is not None :
            self.PSF_kwargs = {'psf_type': 'PIXEL',
                                            'kernel_point_source': self.individual_filter.psf.data,
                                            #'truncation': 35,
                                            #'point_source_supersampling_factor': 1,
                                            'pixel_size': image.pix_deg_scale*3600}
        else :
            self.PSF_kwargs = {'psf_type': 'GAUSSIAN',
                                            'fwhm': image.pix_deg_scale*3600*2,
                                            'truncation': 5,
                                            'pixel_size': image.pix_deg_scale*3600}
        self.PSF = PSF(**self.PSF_kwargs)
        
        self.kwargs_numerics = {'supersampling_factor': 8, #ideally, supersampling_factor=16 for light source model, but 8 is ok. Doesn't matter for point source model.
                                'supersampling_convolution': True}
        
        self.sigma = 0.00036

        #-------------- Image plane: observed / simulated / residual / RGB (linked zoom & pan) --------------#
        self.image_plane_observed = pg.ImageView()
        self.image_plane_simulated = pg.ImageView()
        self.image_plane_residual = pg.ImageView()
        self.image_plane_rgb = DragImagePlotWidget_special(ROI_param_dict=get_ROI_param_dict_lens_model(), slider_init_dict=get_ROI_slider_init_dict_lens_model(), size_sliders_scaling=image.pix_deg_scale*3600, throttle_mode=throttle_mode)
        self.image_plane_plot = self.image_plane_simulated  # backward compatibility

        sim0 = np.zeros_like(self.image_data)
        rgb_data_full = image.boosted_image if image.boosted else image.image_data
        rgb_data = rgb_data_full[ self._crop_y0 - self._crop_npix : self._crop_y0, self._crop_x0 : self._crop_x0 + self._crop_npix ]
        self.simulated_image = sim0
        self.residual_image = self.image_data - sim0
        self.rgb_image_data = rgb_data
        self.image_plane_observed.setImage(self.image_data)
        self.image_plane_observed.getView().invertY(False)
        self.image_plane_simulated.setImage(sim0)
        self.image_plane_simulated.getView().invertY(False)
        self.image_plane_residual.setImage(self.residual_image)
        self.image_plane_residual.getView().invertY(False)
        self.image_plane_rgb.setImage(rgb_data)
        #self.image_plane_rgb.getAxis('bottom').setHeight(0)

        XR = []
        YR = []
        for mult in lens_model.mult.cat :
            xr, yr = world_to_relative( mult['ra'], mult['dec'], self.center_world )
            XR.append(xr)
            YR.append(yr)

        #x_cc = (self._lt_curve_coords_relative_broken[0] - self._SquareOfInterest_xr_bottomleft) / (image.pix_deg_scale * 3600)
        #y_cc = self.image_data.shape[0] - (self._lt_curve_coords_relative_broken[1] - self._SquareOfInterest_yr_bottomleft) / (image.pix_deg_scale * 3600)
        #xr_crit, yr_crit = world_to_relative(ra_crit, dec_crit, self.center_world)
        pen_cc = pg.mkPen(color=[255, 0, 255, 255], width=4.0001)
        self.critical_curve_plots = []
        for iv in (self.image_plane_observed, self.image_plane_simulated, self.image_plane_rgb) : #, self.image_plane_residual
            cc = pg.PlotDataItem()
            cc.setPen(pen_cc)
            iv.addItem(cc)
            self.critical_curve_plots.append(cc)

            mult_plot = pg.ScatterPlotItem(size=5, symbol='+', brush='r', pen=None)
            mult_plot.setData(np.array(XR)/image.pix_deg_scale/3600 + self._crop_npix/2, self._crop_npix/2 + np.array(YR)/image.pix_deg_scale/3600)
            iv.addItem(mult_plot)
        #self.critical_curve_plot = self.critical_curve_plots[1]
        self._curve_grid_scale = self._SquareOfInterest_side_arcsec / 100.0
        _update_curves(self)
        self._last_curve_lens_kwargs = copy.deepcopy(self.LensModel_kwargs)

        vb0 = self.image_plane_observed.getView()
        vb1 = self.image_plane_simulated.getView()
        vb2 = self.image_plane_residual.getView()
        vb3 = self.image_plane_rgb.getViewBox()
        vb1.setXLink(vb0)
        vb1.setYLink(vb0)
        vb2.setXLink(vb0)
        vb2.setYLink(vb0)
        vb3.setXLink(vb0)
        vb3.setYLink(vb0)

        # Fixed proportions: top half = Simulated | Observed (each half width); bottom half = Residual (full width).
        # Outer horizontal split (source | image plane) remains adjustable via QSplitter.
        implane_v = QVBoxLayout(self._imsim_implane_layout)
        implane_v.setContentsMargins(0, 0, 0, 0)
        implane_v.setSpacing(2)

        top_row = QWidget()
        top_h = QHBoxLayout(top_row)
        top_h.setContentsMargins(0, 0, 0, 0)
        top_h.setSpacing(2)

        sim_col = QWidget()
        sim_v = QVBoxLayout(sim_col)
        sim_v.setContentsMargins(0, 0, 0, 0)
        sim_v.setSpacing(2)
        sim_v.addWidget(QtWidgets.QLabel("Simulated"))
        sim_v.addWidget(self.image_plane_simulated, stretch=1)

        obs_col = QWidget()
        obs_v = QVBoxLayout(obs_col)
        obs_v.setContentsMargins(0, 0, 0, 0)
        obs_v.setSpacing(2)
        obs_v.addWidget(QtWidgets.QLabel("Observed"))
        obs_v.addWidget(self.image_plane_observed, stretch=1)

        top_h.addWidget(sim_col, stretch=1)
        top_h.addWidget(obs_col, stretch=1)

        bottom_row = QWidget()
        bottom_h = QHBoxLayout(bottom_row)
        bottom_h.setContentsMargins(0, 0, 0, 0)
        bottom_h.setSpacing(2)

        res_col = QWidget()
        res_v = QVBoxLayout(res_col)
        res_v.setContentsMargins(0, 0, 0, 0)
        res_v.setSpacing(2)
        res_v.addWidget(QtWidgets.QLabel("Residual (obs − sim)"))
        res_v.addWidget(self.image_plane_residual, stretch=1)

        rgb_col = QWidget()
        rgb_v = QVBoxLayout(rgb_col)
        rgb_v.setContentsMargins(0, 0, 0, 0)
        rgb_v.setSpacing(2)
        rgb_v.addWidget(QtWidgets.QLabel("RGB"))
        rgb_v.addWidget(self.image_plane_rgb, stretch=1)

        bottom_h.addWidget(res_col, stretch=1)
        bottom_h.addWidget(rgb_col, stretch=1)

        implane_v.addWidget(top_row, stretch=1)
        implane_v.addWidget(bottom_row, stretch=1)

        self.image_plane_views = {
            'observed': self.image_plane_observed,
            'simulated': self.image_plane_simulated,
            'residual': self.image_plane_residual,
            'rgb': self.image_plane_rgb,
        }

        self.filter_image = ImageFilter(self)
        for _iv in (self.image_plane_observed, self.image_plane_simulated, self.image_plane_residual, self.image_plane_rgb):
            _iv.installEventFilter(self.filter_image)

        self.window.show()
        
        # Initial width ratio source:image = 3:4
        x_left, x_right = self._imsim_layout.sizes()
        total_size = x_left + x_right
        ratio = 3/7
        x_left_new = round(total_size * ratio)
        x_right_new = total_size - x_left_new
        self._imsim_layout.setSizes([x_left_new, x_right_new])

        # Initialize image-plane y-range to full image height for all linked views.
        vb0.setYRange(0, self.image_data.shape[0], padding=0)
        
        #lens_model.lt.set_field(self._initial_field)
        self.previous_state_current_ROI = self.ROI.getState()
    

    def clear(self) :
        self.image_plane_rgb.remove_all_ROIs()
        self.source_plane_widget.remove_all_ROIs()
    
    def world_to_relative(self, ra, dec) :
        return world_to_relative(ra, dec, self.center_world)

    def relative_to_world(self, x, y) :
        return relative_to_world(x, y, self.center_world)
    
    def simulate(self, N_sigma=40., verbose=True) :
        self.lm_dict_opt = make_lm_dict_opt(self, N_sigma=N_sigma)
        self.lm_current = lenstronomy_model(self.lm_dict_opt, self)
        simulate(self, verbose=verbose)
    
    def optimize(self, N_sigma=40., fitting_kwargs_list=None, lm_dict_opt=None) :
        fitting_kwargs_list = getattr(self, 'fitting_kwargs_list', None) if fitting_kwargs_list is None else fitting_kwargs_list
        if lm_dict_opt is None : # This will initialize self.lm_dict_opt
            self.simulate(N_sigma=40.)
        else :
            self.lm_dict_opt = lm_dict_opt
        return optimize(self, fitting_kwargs_list=fitting_kwargs_list)
    
    def set_fitting_kwargs_list(self, fitting_kwargs_list) :
        self.fitting_kwargs_list = fitting_kwargs_list
    
    #def save(self) :
    #    self.model_local = format_lm_local(self.models, self.result_kwargs)
    #    self.source_model.save( path=os.path.join(self.lensmodel.model_dir, "lenstronomy_model.pkl") )
        
    def load(self, path=None) :
        if path is None :
            self.lm_imported = lenstronomy_model( os.path.join(self.lensmodel.model_dir, "lenstronomy_model.pkl"), self )
        else :
            self.lm_imported = lenstronomy_model( path, self )
        
        #self.imported_models = copy.deepcopy(self.models)
        #self.imported_models = {'lens_model_list': self.LensModel_list}
        #self.imported_models['source_light_model_list'] = self.imported_source_model['models']['source_light_model_list']
        
        #self.imported_kwargs = {'kwargs_lens': self.LensModel_kwargs}
        #self.imported_kwargs['kwargs_source'] = self.imported_source_model['results']['kwargs_source']
    
    def add_lensing_columns(self, cat=None, which_cat='mult', index=None, overwrite=None, disable_tqdm=False) :
        if cat is None :
            if index is not None :
                if self.lensmodel.workspace is not None :
                    cat = self.lensmodel.workspace.catalogs[index].cat
                else :
                    cat = None
            else :
                if which_cat == 'mult' :
                    cat = self.lensmodel.mult.cat
                elif which_cat == 'imported_cat' :
                    if self.lensmodel.workspace is not None :
                        cat = self.lensmodel.workspace.catalog.cat
                    else :
                        cat = None
                else :
                    if getattr(self.image, which_cat, None) is not None :
                        cat = getattr(self.image, which_cat, None).cat
                    elif getattr(self.lensmodel, which_cat, None) is not None :
                        cat = getattr(self.lensmodel, which_cat, None).cat
                    else :
                        raise ValueError('Catalog ' + which_cat + ' not found.')
        
        check = False
        existing_cols = []
        for name in self._DEFAULT_LENSING_COLUMNS :
            if name in cat.colnames :
                check = True
                existing_cols.append(name)
        yesno = 'y' if overwrite is None else 'y' if overwrite else 'n'
        if check and overwrite is None :
            yesno = input(f"Following lensing columns already exist: {existing_cols}. Overwrite? [Y]/n")
        
        if yesno.lower() in ['y', ''] :
            mu_col = np.full(len(cat), np.nan)
            gamma_col = np.full(len(cat), np.nan)
            kappa_col = np.full(len(cat), np.nan)
            tmu_col = np.full(len(cat), np.nan)
            rmu_col = np.full(len(cat), np.nan)
            time_col = np.full(len(cat), np.nan)
            gamma1_col = np.full(len(cat), np.nan)
            gamma2_col = np.full(len(cat), np.nan)
            
            ra_source_col = np.full(len(cat), np.nan)
            dec_source_col = np.full(len(cat), np.nan)
            
            z_colname = None
            for name in ['z', 'z_spec', 'zspec', 'z_phot', 'zphot'] :
                if name in cat.colnames :
                    z_colname = name
                    break
            if z_colname is not None :
                if np.unique(cat[z_colname]).shape[0] > 1 or np.unique(cat[z_colname])[0] != self.z_source :
                    print('Warning: catalog contains sources at multiple redshifts different from current source plane redshift. Lensing quantities will be calculated for sources at redshift ' + str(self.z_source) + '.')
            else :
                print('Calculating lensing quantities for sources at redshift ' + str(self.z_source) + '.')
                
            for i in tqdm(range(len(cat)), disable=disable_tqdm) :
                ra, dec = cat['ra'][i], cat['dec'][i]
                xr, yr = self.world_to_relative(ra, dec)
                delta = self.image.pix_deg_scale * 3600 / 2

                x_grid = np.array([xr])
                y_grid = np.array([yr])
                
                # Magnification
                mag_map = util.array2image(self.LensModel.magnification(x_grid, y_grid, self.LensModel_kwargs))
                mu_col[i] = mag_map[0][0]

                # Shear
                g1, g2 = self.LensModel.gamma(x_grid, y_grid, self.LensModel_kwargs)
                gamma1_col[i] = g1[0]
                gamma2_col[i] = g2[0]
                gamma_col[i] = np.sqrt(gamma1_col[i]**2 + gamma2_col[i]**2)

                # Convergence
                kappa_map = self.LensModel.kappa(x_grid, y_grid, self.LensModel_kwargs)
                kappa_col[i] = kappa_map[0]

                # Tangential and radial magnification
                tmu_col[i] = 1 / (1 - kappa_col[i] - gamma_col[i] )
                rmu_col[i] = 1 / (1 - kappa_col[i] + gamma_col[i] )

                # Time delays
                time_map = self.LensModel.arrival_time(x_grid, y_grid, self.LensModel_kwargs)
                time_col[i] = time_map[0]

                # Displacement
                dx_map, dy_map = self.LensModel.alpha(x_grid, y_grid, self.LensModel_kwargs)
                dx = dx_map[0]
                dy = dy_map[0]
                
                # Source coordinates
                ra_source_col[i] = ra + dx /3600 /np.cos( dec * np.pi/180 )
                dec_source_col[i] = dec - dy /3600
            
            columns_to_add = [mu_col, kappa_col, gamma_col, gamma1_col, gamma2_col, time_col, tmu_col, rmu_col, ra_source_col, dec_source_col]
            for i, name in enumerate(self._DEFAULT_LENSING_COLUMNS) :
                if name in cat.colnames :
                    cat.replace_column(name, columns_to_add[i])
                else :
                    cat.add_column(columns_to_add[i], name=name)

    def make_samples_dict(self, nsamples, recompute=False, start_sample_index=None, stop_sample_index=None) :
        lensing_columns = self._DEFAULT_LENSING_COLUMNS
        if start_sample_index is None :
            start_sample_index = 0
        if stop_sample_index is None :
            stop_sample_index = nsamples

        self.samples_dir = os.path.join(self.lensmodel.model_dir, 'samples_lenstronomy')
        if not os.path.exists(self.samples_dir) :
            print('Creating samples directory')
            os.makedirs(self.samples_dir)

        samples_dict_path = os.path.join(self.samples_dir, 'samples_dict_' + str(nsamples) + '.pkl')
        im_ids = self.lensmodel.mult.cat['id']

        self.samples_dict = _load_or_init_samples_dict(
            samples_dict_path,
            nsamples,
            lensing_columns,
            im_ids,
            start_sample_index,
            stop_sample_index,
            recompute,
        )

        for i in tqdm(range(start_sample_index, stop_sample_index), desc=f'samples [{start_sample_index}:{stop_sample_index})') :
            if not recompute and _sample_is_computed(self.samples_dict, i, im_ids, lensing_columns) :
                continue

            sample_kwargs_name = self.lm_imported._make_sample_kwargs('MCMC', i)
            self.LensModel_list = self.lm_imported.local['models']['lens_model_list']
            self.LensModel = LensModel(lens_model_list=self.LensModel_list, z_lens=self.lensmodel.z_lens, z_source=self.z_source)
            self.LensModel_kwargs = self.lm_imported.local[sample_kwargs_name]['kwargs_lens']
            del self.lm_imported.local[sample_kwargs_name]
            del self.lm_imported.world[sample_kwargs_name]

            #self.clear()
            #self.lm_imported.send_to_imsim(step='MCMC', sample_index=i)
            #self.simulate(verbose=False)
            self.add_lensing_columns(cat=self.lensmodel.mult.cat, overwrite=True, disable_tqdm=True)
            for im in self.lensmodel.mult.cat :
                for col in lensing_columns :
                    self.samples_dict[im['id']][col][i] = im[col]

            if i % 10 == 0 or i == stop_sample_index - 1 :
                self.samples_dict = _merge_and_dump_samples_dict(samples_dict_path, 
                                                                 self.samples_dict,
                                                                 start_sample_index,
                                                                 stop_sample_index,
                                                                 nsamples,
                                                                 lensing_columns,
                                                                 im_ids)

    def compute_uncertainties(self, nsamples=None, recompute=False, start_sample_index=None, stop_sample_index=None) :
        if nsamples is None :
            nsamples_list = []
            for i, kwargs_lens_sampled in enumerate(self.lm_imported.local['kwargs_MCMC']['kwargs_lens']) :
                for param_name, param_value in kwargs_lens_sampled.items() :
                    nsamples_list.append(len(param_value))
            nsamples = np.min(nsamples_list)
            print(str(nsamples) + ' full samples found in imported lenstronomy model.')

        self.make_samples_dict(nsamples, recompute=recompute, start_sample_index=start_sample_index, stop_sample_index=stop_sample_index)
        
        for col in self._DEFAULT_LENSING_COLUMNS :
            col_16_percentile = np.full(len(self.lensmodel.mult.cat), np.nan)
            col_84_percentile = np.full(len(self.lensmodel.mult.cat), np.nan)
            col_50_percentile = np.full(len(self.lensmodel.mult.cat), np.nan)

            name = f'{col}_16_percentile'
            if name in self.lensmodel.mult.cat.colnames :
                self.lensmodel.mult.cat.replace_column(name, col_16_percentile)
            else :
                self.lensmodel.mult.cat.add_column(col_16_percentile, name=name)

            name = f'{col}_84_percentile'
            if name in self.lensmodel.mult.cat.colnames :
                self.lensmodel.mult.cat.replace_column(name, col_84_percentile)
            else :
                self.lensmodel.mult.cat.add_column(col_84_percentile, name=name)

            name = f'{col}_50_percentile'
            if name in self.lensmodel.mult.cat.colnames :
                self.lensmodel.mult.cat.replace_column(name, col_50_percentile)
            else :
                self.lensmodel.mult.cat.add_column(col_50_percentile, name=name)

            for im in self.lensmodel.mult.cat :
                im[f'{col}_16_percentile'] = np.percentile(self.samples_dict[im['id']][col], 16)
                im[f'{col}_84_percentile'] = np.percentile(self.samples_dict[im['id']][col], 84)
                im[f'{col}_50_percentile'] = np.percentile(self.samples_dict[im['id']][col], 50)

    def compute_magnification_map(self, numPix=1000, plot=True) :
        self.simulate(verbose=False)
        deltapix = self._square_size_arcsec / numPix
        x_grid, y_grid = util.make_grid(numPix, deltapix)
        self.magnification_map = util.array2image(self.LensModel.magnification(x_grid, y_grid, self.LensModel_kwargs))
        if plot :
            fig, ax = plt.subplots()
            ax.imshow(self.magnification_map, origin='lower')
        return self.magnification_map



class filter_lite() :
    def __init__(self, image_data, exptime=1000.) :
        self.image_data = image_data
        self.header = {'EXPTIME': exptime}
        self.psf = None


def compute_curves(imsim, grid_scale=0.01) :
    lens_model_extensions = LensModelExtensions(imsim.LensModel)
    ra_crit, dec_crit, ra_caustic, dec_caustic = lens_model_extensions.critical_curve_caustics(
            imsim.LensModel_kwargs,
            compute_window=imsim._SquareOfInterest_side_arcsec,
            grid_scale=grid_scale,
            center_x=0.,
            center_y=0.
            )
    return ra_crit, dec_crit, ra_caustic, dec_caustic



#class SpecialWindow(QMainWindow):
#    def __init__(self, lt, parent=None):
#        super().__init__(parent)
#        #self.lt = lt
#        
#    def closeEvent(self, event):
#        print("lenstool instance not re-initialized")
#        #self.lt.lt = lenstool.Lenstool( self.lt._FileToUse )
#        #self.lt.lt.set_field(self.lt.imsim._initial_field)
#        #event.accept()



