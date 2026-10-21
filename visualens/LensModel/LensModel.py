import os
import glob
import shutil
from webbrowser import get
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import constants as astro_constants
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
import warnings
import gc
import PyQt5
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5.QtCore import Qt
import pickle
try:
    import lenstool
    _HAS_LENSTOOL = True
except ImportError :
    _HAS_LENSTOOL = False
    warnings.warn(
        "It looks like Lenstool is not installed. It is required to enable lens modeling functions.\n"
        "You can install it with conda:\n\n"
        "    conda install conda-forge::lenstool\n",
        ImportWarning,
        stacklevel=2)

###############################################################################
from ..utils.utils_astro.utils_general import world_to_relative, relative_to_world
from ..utils.utils_astro.utils_WCS import get_ImageItem_projected_onto_reference
from ..utils.utils_plots.plot_utils_general import plot_corner
from ..utils.utils_general.utils_general import extract_line
from ..utils.utils_general.sort_points import break_curves
from .simulate_image.simulate_image import image_simulator
from .im2source import start_im2source, stop_im2source
from .utils.operations import MakeFunctionFromMap
from .utils.utils_general import export_thumbnails, get_lenstool_file_path, import_lenstool_files
from .utils.utils_multiple_images import import_multiple_images
from .utils.param_extractors import read_potfile, make_param_latex_table, read_bayes_file, parse_lenstool_parameter_file, write_single_sample_best_file
from ..utils.utils_Qt.utils_general import transform_rectangle
from ..utils.utils_Qt.custom_colormaps import get_colormap
from ..utils.utils_astro.get_default_cosmology import get_default_cosmology



class LensModel :
    def __init__(self, model_path, image=None, workspace=None, compute_predictions=True, verbose=True, use_best=False) :
        self.image = image
        self.workspace = workspace
        self.reference = None
        self.saturation = 1.
        self.lt = None
        self.z_lens = None
        self.mult = None
        self.source = None
        self.images = None
        self.images_filtered = None
        self.arclets = None
        self.potfile = None
        self.critical_curve_plot = None
        self.caustic_curve_plot = None
        self.curves = None
        self.current_map_ImageItem = None
        self.map_histogram = None
        self._map_histogram_toggle_btn = None
        self.map_cmap = 'viridis'
        self.map_opacity = 0.6
        self.magnification_abs = False
        self._compute_predictions = compute_predictions
        self.verbose = verbose
        self.has_run = False
        
        # Get model directory
        if not os.path.exists(model_path) :
            raise FileNotFoundError(f"Model directory/file {model_path} does not exist")
        self.model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        
        # Get parameter file
        self.param_file_path = None
        self.best_file_path = None
        if os.path.isfile(model_path) :
            name = os.path.basename(model_path)
            if name.startswith('best') and name.endswith('.par') :
                self.best_file_path = model_path
                self.has_run = True
            else :
                self.param_file_path = model_path
        
        all_par_file_paths = glob.glob(os.path.join(self.model_dir, "*.par"))
        all_par_file_names = [ os.path.basename(file_path) for file_path in all_par_file_paths ]
        # Get the parameter file path if it is not already set
        if self.param_file_path is None :
            for file_path in all_par_file_paths :
                if not os.path.basename(file_path).startswith('best') :
                    with open(file_path, 'r') as file :
                        for line in file :
                            stripped_line = line.strip()
                            if stripped_line and not stripped_line.startswith('#') :  # Skip empty and comment lines
                                if stripped_line.startswith('runmode') :
                                    self.param_file_path = file_path
                                    
        self.param = parse_lenstool_parameter_file(self.param_file_path) if self.param_file_path is not None else None
        
        # Get best file path if it is not already set
        if self.best_file_path is None :
            if 'best.par' in all_par_file_names :
                self.best_file_path = os.path.join(self.model_dir, 'best.par')
                self.has_run = True
            else :
                # Check if a best file exist with a different name extension
                l = [ file_name.startswith('best') and file_name.endswith('.par') for file_name in all_par_file_names ]
                if True in l :
                    i = np.where(l)[0][0]
                    self.has_run = True
                    self.best_file_path = all_par_file_paths[i]
        
        self.param_best = parse_lenstool_parameter_file(self.best_file_path) if self.best_file_path is not None else None

        # If no Image was provided, build an empty placeholder one centered on the
        # model's reference coordinates (read from the parameter/best file), before any
        # of the catalog-loading code below needs a valid self.image.
        if self.image is None :
            from ..image import Image   # local import to avoid a circular import with image.py
            ref = None
            for param_dict in (self.param, self.param_best) :
                if param_dict is not None and 'runmode' in param_dict and 'reference' in param_dict['runmode'] :
                    ref = tuple(param_dict['runmode']['reference'][1:])
                    break
            if ref is None :
                self._vprint("No reference coordinates found in parameter file; centering the auto-created Image on (0, 0).")
                ref = (0., 0.)
            self.image = Image(wcs=ref, workspace=self.workspace)

        # Get bayes file path if it exists
        self.bayes_file_path = os.path.join(self.model_dir, 'bayes.dat') if 'bayes.dat' in os.listdir(self.model_dir) else None
                
        if self.param is not None :
            # Get the multiple images and potfile galaxies from file names in parameter file 
            self.mult_path = os.path.join(self.model_dir, self.param['image']['multfile'][1]) if 'image' in self.param else None
            self.potfile_path = os.path.join(self.model_dir, self.param['potfile']['filein'][1]) if 'potfile' in self.param else None
            self.load_potfile(self.potfile_path)
        else :
            # Look for multiple image file and potfile without getting their file names from parameter file
            self.mult_path = get_lenstool_file_path(self.model_dir, 'mult')
            self.potfile_path = get_lenstool_file_path(self.model_dir, 'potfile')
            self.load_potfile(self.potfile_path)
            if self.potfile is not None :
                self._vprint("Potfile was found. Current instance now has potfile plotting capabilities.")
            if self.mult is None and self.potfile is None :
                self._vprint("No valid multiple image or potfile files found")
        
        if self.mult_path is not None :
            self._vprint("Multiple image file found.")
            self.families = []
            self.broad_families = []
            self.which = []
            import_multiple_images(self, self.mult_path, self.image, units='pixel', marker='o', alpha=[1,0], saturation=0.8)
        
        # Get the lens redshift if unique
        if self.param is not None  or self.param_best is not None :
            param_dict = self.param if self.param is not None else self.param_best
            redshifts = []
            for name in param_dict :
                if 'potential' in name or 'potentiel' in name :
                    redshifts.append(param_dict[name]['z_lens'])
            if len(np.unique(redshifts))==1 :
                self.z_lens = redshifts[0]
                self._vprint(f"Lens redshift fixed at z={self.z_lens}")
            else :
                self._vprint("Several different redshift values found: " + str(np.unique(redshifts)))
                self.z_lens = np.max(redshifts)
                self._vprint("Setting LensModel.z_lens to the furthest lens' redshift: " + str(self.z_lens))
        
        # Checks if Lenstool files were found and if so use Lenstool's wrapper
        # Moves to the model's directory (required by the Lenstool wrapper)
        # Check which file to use: best.par or parameter file
        self._FileToUse = None
        if self.param_file_path is not None and not (use_best and self.best_file_path is not None) :
            self._FileToUse = os.path.basename(self.param_file_path)
        elif self.best_file_path is not None :
            self._vprint("Parameter file not found, using best.par file instead. Limited statistics capabilities available.")
            self._FileToUse = os.path.basename(self.best_file_path)
        else :
            self._vprint("No best file or parameter file found."
                         "Make sure parameter file has extension '.par' or best file has name 'best.par'")
            import_lenstool_files(self)
        
        # Load the bayes samples if the bayes file exists
        if self.bayes_file_path is not None :
            self.samples_df_full = read_bayes_file(self.bayes_file_path, z=self.z_lens)
            self.bayes_table = Table.from_pandas(self.samples_df_full)

            # Extract the numeric columns (skip non-numeric, zero-range etc.)
            self.samples_df = self.samples_df_full.copy()
            for col in self.samples_df.columns :
                if col=='Chi2' or col=='Nsample' or col=='ln(Lhood)':
                    del self.samples_df[col]
        

        # Automatically use Lenstool's wrapper if a valid lenstool file was found
        if self._FileToUse is not None :
            if not _HAS_LENSTOOL :
                self._vprint("\n------\n"
                             "It looks like Lenstool is not installed. Only limited lens modeling functions will be available.\n"
                             "You can install Lenstool with conda:\n\n"
                             "    conda install conda-forge::lenstool\n"
                             "------\n")
                self._vprint("Lens model loaded with limited capabilities.")
                self.lt = None
                # Functionalities without Lenstool's wrapper
                if param_dict is not None :
                    self.reference = tuple(param_dict['runmode']['reference'][1:]) if 'runmode' in param_dict else None
                import_lenstool_files(self)
            else :
                self._vprint(f"\nA valid lenstool file was found: {self._FileToUse}")
                self._vprint('--------------------')
                self._vprint("Moving to " + self.model_dir)
                self._vprint('--------------------')
                os.chdir(self.model_dir)
                self._vprint(f"Loading {self._FileToUse}")
                self.lt = lenstool.Lenstool( self._FileToUse )
                if self.param_file_path is not None and self.bayes_file_path is not None :
                    try :
                        self.lt.readBayesModels()
                        self.lt.readConstraints()
                        chains, colnames = self.lt.get_chains()
                        self.samples_table = Table(chains, names=colnames)
                        # the get_chains() function flips the log likelihood for some reason, so we flip it back
                        self.samples_table['ln(Lhood)'] = -self.samples_table['ln(Lhood)']
                        self.lt.setBayesModel(method=-4)
                    except RuntimeError :
                        self._vprint("Error reading Bayes models. Try restarting your kernel before importing new lens model.")
                self.reference = (self.lt.M.ref_ra, self.lt.M.ref_dec)
        
        
        # some useful initializations
        self.critical_curve_plot = None
        self.caustic_curve_plot = None
        self.magnification_res = 1000
        self.magnification_line_ax = None
        self.previous_state_current_ROI = None
        self.LENSTRONOMY_fixed_source_kwargs = []
        
        # compute sources and images from multiple images
        if self.mult is not None and self.lt is not None and compute_predictions :
            self.add_lensing_columns(cat=self.mult.cat)
            self.compute_sources_and_images()

        # read burnin and chires files
        self.read_burnin()
        self.read_chires()

        # get cosmology from Lenstool instance if available, otherwise use default cosmology
        self.cosmo = self._get_cosmo()

        # Load the caustic and critical curves if they exist
        self._get_curves()
        
        # set initial default source redshift
        if getattr(self, 'z_lens', None) is not None :
            self.Dl = self.cosmo.angular_diameter_distance_z1z2(0., self.z_lens)
            self.z_source = np.ceil(self.z_lens) + 2.
            self._get_maps_zinf()
            self._update_z_source()
            







            
    def compute_sources_and_images(self, ngrid=256, restrict_to_mult_field=False) :
        if self.source is not None :
            self.source.clear()
        if self.images is not None :
            self.images.clear()
        
        self._vprint("Computing sources")
        source = self.mult.cat.copy()
        source.rename_column('ra', 'ra_image')
        source.rename_column('dec', 'dec_image')
        source.rename_column('ra_source', 'ra')
        source.rename_column('dec_source', 'dec')
        
        import_multiple_images(self, source, self.image, AttrName='source', units='pixel', marker='x', alpha=[1,1], saturation=1., scale=0.5)
        
        # Format source catalog in the Lenstool format
        source.rename_column('id', 'n')
        source['x'], source['y'] = source['ra'], source['dec'] #self.world_to_relative(source['ra'], source['dec'])
        
        for colname in source.colnames :
            if colname not in ['n','x','y','a','b','z','theta','mag'] :
                source.remove_column(colname)
        
        to_remove = []
        for i, s in enumerate(source) :
            if np.isnan(s['z']) or s['z'] <= self.z_lens :
                to_remove.append(i)

        source.remove_rows(to_remove)
        self._vprint('done')
        
        self._vprint("Computing images")
        _initial_ngrid_value = self.lt.G.ngrid
        self.lt.set_grid(ngrid, 0)
        
        if restrict_to_mult_field :
            xr, yr = self.world_to_relative(self.mult.cat['ra'], self.mult.cat['dec'])
            _initial_field = self.lt.get_field([])
            self.lt.set_field([np.min(xr)-1., np.max(xr)+1., np.min(yr)-1., np.max(yr)+1.])
        
        self.lt.set_sources(source)
        self.lt.e_lensing()
        image = self.lt.get_images()
        
        image['ra'], image['dec'] = self.relative_to_world(image['x'], image['y'])
        image['x'], image['y'] = self.image.world_to_image(image['ra'], image['dec'])
        
        image.rename_column('n', 'id')
        
        cols_to_add = [[], [], []]
        
        for row in image :
            i = np.where(self.mult.cat['id']==row['id'])[0][0]
            for j, colname in enumerate(['family','broad_family', 'confidence']) :
                cols_to_add[j].append(self.mult.cat[colname][i])
        for j, colname in enumerate(['family','broad_family', 'confidence']) :
            image.add_column(cols_to_add[j], name=colname)
        
        import_multiple_images(self, image, self.image, AttrName='images', units='pixel', marker='+', alpha=[1,1], saturation=1., scale=0.75)
        import_multiple_images(self, image, self.image, AttrName='images_filtered', units='pixel', marker='+', alpha=[1,1], saturation=1., scale=0.75)
        self.filter_image()
        
        self.lt.set_grid(_initial_ngrid_value, 0)
        if restrict_to_mult_field :
            self.lt.set_field(_initial_field)
        self._vprint('done')
    
    def _vprint(self, *args, **kwargs) :
        if self.verbose :
            print(*args, **kwargs)

    def world_to_relative(self, ra, dec) :
        return world_to_relative(ra, dec, self.reference)
    
    def relative_to_world(self, xr, yr) :
        return relative_to_world(xr, yr, self.reference)
    
    def load_potfile(self, path) :
        if self.potfile is not None :
            self.potfile.clear()
        if path is not None :        
            potfile_Table = read_potfile(path)
            self.potfile = self.image.make_catalog(potfile_Table, color=[1.,0.,0.], units='arcsec', verbose=self.verbose)
            self.potfile.workspace = self.workspace
    
    def _get_maps_zinf(self) :
        self._displacements_maps_zinf_path = os.path.join(self.model_dir, 'dpl_zinf.fits')
        if os.path.exists(self._displacements_maps_zinf_path) :
            self._vprint('dpl_zinf.fits found')
            with fits.open(self._displacements_maps_zinf_path) as hdul :
                dx_map_zinf, dy_map_zinf, dmap_wcs = hdul['dx'].data, hdul['dy'].data, WCS(hdul[0].header)
        else :
            self._update_z_source()
            self._vprint(f"Computing displacement maps... (can take a little while)")
            dx_map, dy_map, dmap_wcs = self.compute_displacement_maps()
            self._vprint('done')
            dx_map_zinf = dx_map / self.DlsDs
            dy_map_zinf = dy_map / self.DlsDs

            header = dmap_wcs.to_header()
            header['UNITS'] = 'arcsec'
            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU(header=header))
            hdul.append(fits.ImageHDU(data=dx_map_zinf, header=header, name='dx'))
            hdul.append(fits.ImageHDU(data=dy_map_zinf, header=header, name='dy'))
            hdul.writeto(self._displacements_maps_zinf_path, overwrite=True)

        self._dpl_zinf = {'dx': dx_map_zinf, 'dy': dy_map_zinf, 'wcs': dmap_wcs}
        self._maps_wcs = dmap_wcs

        # Get the x and y pixel scales (in degrees per pixel) from the WCS
        self._maps_xpix2arcsec, self._maps_ypix2arcsec = np.abs(dmap_wcs.wcs.cdelt[0]) * 3600., np.abs(dmap_wcs.wcs.cdelt[1]) * 3600.
 
        self.alpha_inf = np.array([dx_map_zinf, dy_map_zinf])
        dx_map_zinf_pix_units = dx_map_zinf / self._maps_xpix2arcsec
        dy_map_zinf_pix_units = dy_map_zinf / self._maps_ypix2arcsec
        self.kappa_inf = (np.gradient(dx_map_zinf_pix_units)[1] + np.gradient(dy_map_zinf_pix_units)[0]) / 2
        gamma_1_inf = (np.gradient(dx_map_zinf_pix_units)[1] - np.gradient(dy_map_zinf_pix_units)[0]) / 2
        gamma_2_inf = (np.gradient(dx_map_zinf_pix_units)[0] + np.gradient(dy_map_zinf_pix_units)[1]) / 2
        self.gamma_inf = np.array([gamma_1_inf, gamma_2_inf])
        self.shear_inf = np.sqrt(gamma_1_inf**2 + gamma_2_inf**2)

    def _get_curves(self) :
        self.lt_curves_path = os.path.join(self.model_dir, 'lt_curves.pkl')
        if os.path.exists(self.lt_curves_path) :
            self._vprint('lt_curves.pkl found')
            with open(self.lt_curves_path, 'rb') as f :
                self.lt_curves = pickle.load(f)
        else :
            self.lt_curves = {}

        self.lt_caustics_path = os.path.join(self.model_dir, 'lt_caustics.pkl')
        if os.path.exists(self.lt_caustics_path) :
            self._vprint('lt_caustics.pkl found')
            with open(self.lt_caustics_path, 'rb') as f :
                self.lt_caustics = pickle.load(f)
        else :
            self.lt_caustics = {}
    
    def plot(self, which=None) :
        if which is not None :
            self.set_which(which)
        if self.mult is not None :
            self.mult.plot() #marker='o', scale=1.25)#size=1.5, linewidth=2)
            self.mult.plot_column('id')
        if self.source is not None :
            self.source.plot() #marker='x', alpha=[1,1], saturation=1., scale=0.5)
            #self.source.plot_column('id')
        #if self.images is not None :
        #    #self.images.plot(marker='x', alpha=[1,1], saturation=1., scale=1)
        #    #self.images.plot_column('id')
        if self.images_filtered is not None :
            self.images_filtered.plot() #marker='+', alpha=[1,1], saturation=1., scale=0.75)
            #self.images_filtered.plot_column('id')
        if self.curves is not None :
            self.curves.plot()
    
    def clear(self) :
        if self.mult is not None :
            self.mult.clear()
        if self.images is not None :
            self.images.clear()
        if self.images_filtered is not None :
            self.images_filtered.clear()
        if self.arclets is not None :
            self.arclets.clear()
        if self.curves is not None :
            self.curves.clear()
        if self.critical_curve_plot is not None :
            self.image.ImageView.removeItem(self.critical_curve_plot)
            self.critical_curve_plot = None
        if self.caustic_curve_plot is not None :
            self.image.ImageView.removeItem(self.caustic_curve_plot)
            self.caustic_curve_plot = None
            
    def set_which(self, *names) :
        if names[0]=='all' :
            self.which = self.broad_families if type(self.broad_families) is list else self.broad_families.tolist()
        elif isinstance(names[0], list) :
            self.which = names[0]
        else :
            self.which = list(names)
        self._vprint("Images to plot are now ", self.which)
        #self.clear()
        #self.plot()
    
    
    def export_thumbnails(self, group_images=True, square_thumbnails=True, square_size=150, margin=50, distance=200, export_dir=None, boost=True, make_broad_view=True, broad_view_params=None) :
        export_thumbnails(self.mult, group_images=group_images, square_thumbnails=square_thumbnails, square_size=square_size, margin=margin, \
                          distance=distance, export_dir=export_dir, boost=boost, make_broad_view=make_broad_view, broad_view_params=broad_view_params)
    
    #def make_webpage(self) :
    #    print('function to be implemented in the future')
        
    def make_latex(self) :
        latex_str = make_param_latex_table(self.param_file_path, convert_to_kpc=True, z=self.z_lens)
        return latex_str
    
    def read_burnin(self) :
        """
        Reads the burnin.dat file created by Lenstool and saves the data as an
        astropy Table (LensModel.burnin_table), similar to samples_table.
        
        burnin.dat has no header, but its columns are identical to those of
        bayes.dat (['Nsample', 'ln(Lhood)', <one column per free parameter>, 'Chi2']).
        The 'Nsample' column is skipped. Other column names are taken from the
        bayes.dat header when available, or from Lenstool's bayesHeader() otherwise.
        
        Args:
            burnin_file_path: path to the burnin.dat file. Defaults to
                              'burnin.dat' in the model directory.
        Returns:
            burnin_table: astropy Table with one row per burn-in sample
        """
        burnin_file_path = os.path.join(self.model_dir, 'burnin.dat')
        if not os.path.isfile(burnin_file_path) :
            self._vprint("No burnin file found at " + burnin_file_path)
            self.burnin_table = None
            return None
        
        burnin = np.loadtxt(burnin_file_path)
        if burnin.ndim == 1 :
            burnin = burnin[None, :]
        
        # Get the column names from the bayes.dat header (or from Lenstool's wrapper)
        colnames = None
        if self.bayes_file_path is not None :
            colnames = []
            with open(self.bayes_file_path, 'r') as file :
                for line in file :
                    if line.startswith('#') :
                        colnames.append(line[1:].strip())
                    else :
                        break
        elif self.lt is not None :
            colnames = self.lt.bayesHeader()
            colnames[-1] = 'Evidence' # Evidence is the last column in the burnin.dat file, instead of the chi2
        
        if colnames is None or len(colnames) != burnin.shape[1] :
            self._vprint("Could not match burnin columns to bayes.dat header, using generic column names.")
            colnames = ['col' + str(i) for i in range(burnin.shape[1])]
        
        burnin = burnin[:, 1:]
        colnames = colnames[1:]
        
        self.burnin_table = Table(burnin, names=colnames)
        self._vprint(f"Loaded {len(self.burnin_table)} burn-in samples from {burnin_file_path}")
        return self.burnin_table
    
    def plot_chi2(self, show_matplotlib=True) :
        self._chi2_dict = {"burnin": None, "sampling": None}

        cst = -2 * self.bayes_table['ln(Lhood)'][0] - self.bayes_table['Chi2'][0] if getattr(self, "bayes_table", None) is not None else 0.
        if getattr(self, "burnin_table", None) is not None :
            chi2_burnin = -2 * self.burnin_table['ln(Lhood)'] - cst
            x_burnin = np.arange(len(chi2_burnin))
            self._chi2_dict["burnin"] = {"chi2": chi2_burnin, "steps": x_burnin}

        if getattr(self, "bayes_table", None) is not None :
            chi2_sampling = self.bayes_table['Chi2']
            n_burnin = len(self._chi2_dict["burnin"]["steps"]) if self._chi2_dict["burnin"] is not None else 0
            x_sampling = np.arange(n_burnin, n_burnin + len(chi2_sampling))
            self._chi2_dict["sampling"] = {"chi2": chi2_sampling, "steps": x_sampling}
    
        if show_matplotlib :
            fig, ax = plt.subplots()
            for key, label in (("burnin", 'Burn-in'), ("sampling", 'Sampling')) :
                if self._chi2_dict[key] is not None :
                    ax.plot(self._chi2_dict[key]["steps"], self._chi2_dict[key]["chi2"], label=label)
            ax.legend()
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Chi2')
            ax.grid(True)
            fig.show()
            return fig, ax
        return self._chi2_dict

    def read_chires(self) :
        """
        Reads the chires.dat file created by Lenstool and saves the data as an
        astropy Table (LensModel.chires_table).
        
        Only the per-image rows matching the header columns
        (N, ID, z, Narcs, chip, ...) are stored in the table; 'N/A' entries
        (e.g. dx/dy on family summary rows) are converted to NaN. The summary
        lines at the end of the file are not stored, except for the chitot and
        log(Likelihood) values, which are saved as LensModel.chitot and
        LensModel.log_likelihood and printed.
        
        Args:
            chires_file_path: path to the chires.dat file. Defaults to
                              'chires.dat' in the model directory.
        Returns:
            chires_table: astropy Table with one row per (image, Narcs) entry
        """
        chires_file_path = os.path.join(self.model_dir, 'chires.dat')
        if not os.path.isfile(chires_file_path) :
            self._vprint("No chires file found at " + chires_file_path)
            self.chires_table = None
            return None
        
        colnames = None
        rows = []
        self.chitot = None
        self.log_likelihood = None
        with open(chires_file_path, 'r') as file :
            for line in file :
                tokens = line.split()
                if len(tokens) == 0 :
                    continue
                if colnames is None :
                    # The header is the first line containing the 'ID' column
                    if 'ID' in tokens :
                        colnames = tokens
                    continue
                if len(tokens) == len(colnames) and tokens[0].isdigit() :
                    rows.append(tokens)
                elif tokens[0] == 'chitot' :
                    self.chitot = float(tokens[1])
                elif tokens[0] == 'log(Likelihood)' :
                    self.log_likelihood = float(tokens[1])
        
        if colnames is None or len(rows) == 0 :
            self._vprint("Could not parse any data rows from " + chires_file_path)
            self.chires_table = None
            return None
        
        self.chires_table = Table()
        for name, column in zip(colnames, zip(*rows)) :
            try :
                self.chires_table[name] = np.array(column, dtype=int)
            except ValueError :
                try :
                    self.chires_table[name] = np.array([np.nan if value == 'N/A' else float(value) \
                                                        for value in column])
                except ValueError :
                    self.chires_table[name] = np.array(column)
        
        self._vprint(f"Loaded {len(self.chires_table)} rows from {chires_file_path}")
        print(f"chitot          : {self.chitot}")
        print(f"log(Likelihood) : {self.log_likelihood}")

        mask = self.chires_table['Narcs']==1
        N = len(self.chires_table[mask])
        self.RMS = np.sqrt( np.sum( self.chires_table[mask]['rmsi']**2 ) / N)
        print('\n--------------------------------')
        print(f"RMS : {self.RMS}")
        print('--------------------------------\n')

        return self.chires_table

    def write_optimized_param_file(self, output_path=None, prior_mode=None, nsigma=None, z_prior_mode=None) :
        """
        Creates a new Lenstool input parameter file identical to the one at
        self.param_file_path, but with the initial values of the potentials
        replaced by the optimized values from self.param_best.
        Comments, formatting and all other sections (potfile, cosmology, etc.) are preserved.
        Gaussian priors in the limit sections can either be kept unchanged
        (they are then centered on the new, optimized initial values) or replaced
        with uniform priors (flag 1) spanning the optimized value +/- nsigma*sigma.
        Optimized image redshifts (z_m_limit lines in the image section, with the
        optimized values taken from the 'z_opt' column of LensModel.mult.cat)
        receive a similar treatment: gaussian redshift priors ('z_m_limit 1 <ids> 3
        mean sigma precision') are either re-centered on the optimized redshift
        (same sigma), replaced with uniform priors around it, or copied unchanged.
        Optimized potfile parameters (e.g. 'sigma 3 mean stddev', 'cut 3 mean stddev'),
        whose best values are not written in the best file, are taken from the maximum
        likelihood sample of the bayes chains (LensModel.samples_table); their
        gaussian priors follow prior_mode like the limit sections.
        Args:
            output_path: path of the new parameter file. Defaults to the original
                         file name with an '_optimized' suffix.
            prior_mode: what to do with gaussian priors in the limit sections:
                        'gaussian' to keep them (same sigma, now centered on the
                        optimized value), 'uniform' to replace them with uniform
                        priors. If None and gaussian priors are present, the
                        initial and optimized values are printed and the user is
                        asked which option to use.
            nsigma: half-width of the replacement uniform priors, in units of the
                    gaussian sigma. Only used with the 'uniform' option.
                    If None, the user is asked.
            z_prior_mode: what to do with gaussian redshift priors (z_m_limit):
                          'gaussian' to re-center them on the optimized redshifts,
                          'uniform' to replace them with uniform priors, 'keep' to
                          copy the lines unchanged. If None, defaults to prior_mode
                          when the latter was specified; otherwise the user is asked.
        Returns:
            output_path: path of the written parameter file
        """
        if self.param_file_path is None :
            raise ValueError("No parameter file found (LensModel.param_file_path is None).")
        if self.param_best is None :
            raise ValueError("No best file found (LensModel.param_best is None). Run the optimization first.")
        if prior_mode not in (None, 'gaussian', 'uniform') :
            raise ValueError("prior_mode must be 'gaussian' or 'uniform'")
        if z_prior_mode not in (None, 'gaussian', 'uniform', 'keep') :
            raise ValueError("z_prior_mode must be 'gaussian', 'uniform' or 'keep'")
        prior_mode_specified = prior_mode is not None
        
        if output_path is None :
            root, ext = os.path.splitext(self.param_file_path)
            output_path = root + '_optimized' + ext
        
        def normalize_key(key) :
            return 'ellipticity' if key=='ellipticite' else key
        
        def format_value(value) :
            if isinstance(value, list) :
                return ' '.join(str(v) for v in value)
            return str(value)
        
        # Ordered potential sections from the parameter file and the best file
        param_pot_names = [ name for name in self.param if name.startswith('potential') ]
        best_pot_items = [ (name, self.param_best[name]) for name in self.param_best if name.startswith('potential') ]
        
        def find_best_pot(param_pot_name) :
            # Match by position
            order_index = param_pot_names.index(param_pot_name)
            if order_index < len(best_pot_items) :
                return best_pot_items[order_index][1]
            return None
        
        def find_pot_for_limit(limit_section_name) :
            # Match 'limit <name>' to 'potential <name>'
            for pot_name in param_pot_names :
                if pot_name.split()[1:]==limit_section_name.split()[1:] :
                    return pot_name
            return None
        
        def split_z_m_limit(values) :
            # values: tokens following the 'z_m_limit' keyword (enable flag, image ids, prior type, prior params)
            # Returns the image ids and the index of the prior type token (the first integer after the enable flag)
            for i, tok in enumerate(values[1:], start=1) :
                if str(tok).lstrip('+-').isdigit() :
                    return [str(v) for v in values[1:i]], i
            return None, None
        
        def z_opt_for_ids(image_ids) :
            # Optimized redshift of a system from the multiple image catalog
            if self.mult is None or 'z_opt' not in self.mult.cat.colnames :
                return None
            for image_id in image_ids :
                for row in self.mult.cat :
                    if str(row['id'])==image_id and not np.isnan(row['z_opt']) :
                        return float(row['z_opt'])
            return None
        
        def potfile_best_value(keyword) :
            # Optimized potfile values are not written in the best file:
            # take them from the maximum likelihood sample of the bayes chains
            samples_table = getattr(self, 'samples_table', None)
            if samples_table is None :
                return None
            # Potfile keywords vs parameter names in the chains columns ('Pot0 rcut (arcsec)' etc.)
            name_map = {'sigma': 'sigma', 'cut': 'rcut', 'rcut': 'rcut', 'core': 'rcore',
                        'slope': 'slope', 'vdslope': 'vdslope'}
            if keyword not in name_map :
                return None
            lhood_col = None
            for col in samples_table.colnames :
                if 'lhood' in col.lower() :
                    lhood_col = col
                    break
            if lhood_col is None :
                return None
            best_index = np.argmax(samples_table[lhood_col])
            for col in samples_table.colnames :
                if col.split()[:2]==['Pot0', name_map[keyword]] :
                    return float(samples_table[col][best_index])
            return None
        
        def print_priors(priors) :
            for prior in priors :
                print(f"    {prior['section']}  {prior['key']}: initial = {prior['initial']}, "
                      f"optimized = {prior['optimized']}, sigma = {prior['sigma']}")
        
        # Collect the gaussian priors (flag 3) from the limit sections
        pot_gaussian_priors = []
        for section in self.param :
            if section.startswith('limit') :
                pot_name = find_pot_for_limit(section)
                if pot_name is None :
                    continue
                best_pot = find_best_pot(pot_name)
                for key, values in self.param[section].items() :
                    if isinstance(values, list) and values[0]==3 :
                        pot_gaussian_priors.append({ 'section': section,
                                                     'key': key,
                                                     'sigma': values[1],
                                                     'initial': self.param[pot_name].get(key),
                                                     'optimized': best_pot.get(key) if best_pot is not None else None })
        
        # Collect the gaussian priors (flag 3, 'keyword 3 mean stddev') from the potfile section(s)
        for section in self.param :
            if section.startswith('potfile') :
                for key, values in self.param[section].items() :
                    if isinstance(values, list) and values[0]==3 and key != 'filein':
                        pot_gaussian_priors.append({ 'section': section,
                                                     'key': key,
                                                     'sigma': values[2],
                                                     'initial': values[1],
                                                     'optimized': potfile_best_value(key) })
        
        # Collect the gaussian redshift priors (flag 3) from the z_m_limit lines of the image section
        z_gaussian_priors = []
        if 'image' in self.param and 'z_m_limit' in self.param['image'] :
            z_m_limit_entries = self.param['image']['z_m_limit']
            if not isinstance(z_m_limit_entries[0], list) :
                z_m_limit_entries = [z_m_limit_entries]
            for values in z_m_limit_entries :
                image_ids, i = split_z_m_limit(values)
                if image_ids is not None and values[i]==3 :
                    z_gaussian_priors.append({ 'section': 'image',
                                               'key': 'z_m_limit ' + ' '.join(image_ids),
                                               'sigma': values[i+2],
                                               'initial': values[i+1],
                                               'optimized': z_opt_for_ids(image_ids) })
        
        if pot_gaussian_priors and prior_mode is None :
            print("Gaussian priors (flag 3) found in the limit/potfile sections:")
            print_priors(pot_gaussian_priors)
            answer = ''
            while answer not in ['g', 'u'] :
                answer = input("Keep gaussian priors, now centered on the optimized values [g], "
                               "or replace them with uniform priors of half-width n*sigma [u]? ").strip().lower()
            prior_mode = 'gaussian' if answer=='g' else 'uniform'
        
        if z_gaussian_priors and z_prior_mode in (None, 'uniform') :
            print("Gaussian redshift priors (z_m_limit, flag 3) found in the image section:")
            print_priors(z_gaussian_priors)
            if prior_mode_specified :
                z_prior_mode = prior_mode
            else :
                answer = ''
                while answer not in ['g', 'u', 'k'] :
                    answer = input("Re-center gaussian priors on the optimized redshifts [g], "
                                   "replace them with uniform priors of half-width n*sigma [u], "
                                   "or keep the lines unchanged [k]? ").strip().lower()
                z_prior_mode = {'g': 'gaussian', 'u': 'uniform', 'k': 'keep'}[answer]
        
        if nsigma is None and ((pot_gaussian_priors and prior_mode=='uniform')
                               or (z_gaussian_priors and z_prior_mode=='uniform')) :
            nsigma = float(input("Half-width of the uniform priors in units of sigma (n): "))
        
        with open(self.param_file_path, 'r') as f :
            lines = f.readlines()
        
        new_lines = []
        current_section = None
        current_best_pot = None
        current_limit_best_pot = None
        for line in lines :
            stripped = line.split('#')[0].strip()
            tokens = stripped.split()
            
            if not tokens :
                new_lines.append(line)
                continue
            
            if current_section is None :
                if tokens[0].lower() in ('fini', 'finish') :
                    new_lines.append(line)
                    continue
                # New section begins
                current_section = stripped.replace('potentiel', 'potential', 1)
                current_best_pot = None
                current_limit_best_pot = None
                if current_section.startswith('potential') :
                    current_best_pot = find_best_pot(current_section)
                    if current_best_pot is None :
                        self._vprint(f"No optimized values found for '{stripped}', keeping initial values.")
                elif current_section.startswith('limit') and prior_mode=='uniform' :
                    pot_name = find_pot_for_limit(current_section)
                    if pot_name is not None :
                        current_limit_best_pot = find_best_pot(pot_name)
                new_lines.append(line)
                continue
            
            if tokens[0].lower()=='end' :
                current_section = None
                current_best_pot = None
                current_limit_best_pot = None
                new_lines.append(line)
                continue
            
            if current_best_pot is not None :
                key = normalize_key(tokens[0])
                if key not in ('identity', 'profile', 'z_lens') :
                    best_value = current_best_pot.get(key)
                    if best_value is not None :
                        indent = line[:len(line) - len(line.lstrip())]
                        comment = '  ' + line[line.index('#'):].rstrip('\n') if '#' in line else ''
                        new_lines.append(indent + tokens[0] + '  ' + format_value(best_value) + comment + '\n')
                        continue
            
            # Replace gaussian priors (flag 3) with uniform priors around the optimized value
            if current_limit_best_pot is not None and len(tokens)>=3 and tokens[1]=='3' :
                best_value = current_limit_best_pot.get(normalize_key(tokens[0]))
                if isinstance(best_value, (int, float)) :
                    sigma = float(tokens[2])
                    indent = line[:len(line) - len(line.lstrip())]
                    comment = '  ' + line[line.index('#'):].rstrip('\n') if '#' in line else ''
                    new_lines.append(indent + tokens[0] + f'  1 {best_value - nsigma*sigma} {best_value + nsigma*sigma}' + comment + '\n')
                    continue
            
            # Update gaussian redshift priors (z_m_limit, flag 3) with the optimized redshifts
            # ('keep' or None leaves the lines unchanged)
            if current_section=='image' and tokens[0]=='z_m_limit' and z_prior_mode in ('gaussian', 'uniform') :
                image_ids, i = split_z_m_limit(tokens[1:])
                if image_ids is not None and tokens[1+i]=='3' :
                    z_opt = z_opt_for_ids(image_ids)
                    if z_opt is None :
                        self._vprint(f"No optimized redshift found for {image_ids}, keeping line unchanged.")
                    else :
                        k = 1 + i  # index of the prior type in `tokens`
                        sigma = float(tokens[k+2])
                        indent = line[:len(line) - len(line.lstrip())]
                        comment = '  ' + line[line.index('#'):].rstrip('\n') if '#' in line else ''
                        if z_prior_mode=='gaussian' :
                            # Same sigma (and precision), re-centered on the optimized redshift
                            new_tokens = tokens[:k+1] + [str(z_opt)] + tokens[k+2:]
                        else :
                            new_tokens = tokens[:k] + ['1', str(z_opt - nsigma*sigma), str(z_opt + nsigma*sigma)] + tokens[k+3:]
                        new_lines.append(indent + ' '.join(new_tokens) + comment + '\n')
                        continue
            
            # Update gaussian potfile priors (flag 3) with the maximum likelihood values from the bayes chains
            if current_section.startswith('potfile') and len(tokens)>=4 and tokens[1]=='3' and prior_mode in ('gaussian', 'uniform') :
                best_value = potfile_best_value(tokens[0])
                if best_value is None :
                    self._vprint(f"No optimized value found for potfile parameter '{tokens[0]}', keeping line unchanged.")
                else :
                    sigma = float(tokens[3])
                    indent = line[:len(line) - len(line.lstrip())]
                    comment = '  ' + line[line.index('#'):].rstrip('\n') if '#' in line else ''
                    if prior_mode=='gaussian' :
                        # Same sigma, re-centered on the maximum likelihood value
                        new_tokens = tokens[:2] + [str(best_value)] + tokens[3:]
                    else :
                        new_tokens = [tokens[0], '1', str(best_value - nsigma*sigma), str(best_value + nsigma*sigma)] + tokens[4:]
                    new_lines.append(indent + ' '.join(new_tokens) + comment + '\n')
                    continue
            
            new_lines.append(line)
        
        with open(output_path, 'w') as f :
            f.writelines(new_lines)
        self._vprint(f"Optimized parameter file written to {output_path}")
        return output_path

    def set_z_source(self, z, color=[255,100,255], recompute=False, plot_critical=True, plot_caustic=False) :
        self.z_source = z
        self._vprint(self.best_file_path)
        self._vprint(os.getcwd())
        #self.lt.set_grid(50, 0)

        ######## Curves ########
        if z not in self.lt_curves.keys() or recompute :
            self.compute_lt_curve(z)
        self.lt_curve_coords_relative = self.lt_curves[z]
        self.lt_caustic_coords_relative = self.lt_caustics[z]
        self._curves_add_all_coords()
        if plot_critical :
            self.plot_lt_curve(color=color, which='critical')
        if plot_caustic :
            self.plot_lt_curve(color=[0, 255, 255], which='caustic')
        
        self._update_z_source()

        mmap, wcs = self.mu, self._maps_wcs
        self.get_magnification = MakeFunctionFromMap(mmap, wcs)
    
    def start_im2source(self) :
        start_im2source(self)
        
    def stop_im2source(self) :
        stop_im2source(self)
    
    def start_magnification(self) :
        return None
    
    def add_lensing_columns(self, cat=None, which_cat='imported_cat', index=None, z_source=None, overwrite=None) :
        if cat is None :
            if index is not None :
                if self.workspace is not None :
                    cat = self.workspace.catalogs[index].cat
                else :
                    cat = None
            elif which_cat == 'imported_cat' :
                if self.workspace is not None :
                    cat = self.workspace.catalog.cat
                else :
                    cat = None
            else :
                cat = getattr(self.image, which_cat, None).cat
        
        lensing_columns = ['magnification', 'convergence', 'shear', 'gamma1', 'gamma2', 'time', 'tangential_magnification', 'radial_magnification', 'ra_source', 'dec_source']
        check = False
        existing_cols = []
        for name in lensing_columns :
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
            
            initial_field = self.lt.get_field([])
            
            z_colname = None
            for name in ['z', 'z_spec', 'zspec', 'z_phot', 'zphot'] :
                if name in cat.colnames :
                    z_colname = name
                    break
            if z_colname is None and z_source is None :
                self._vprint('Redshift column not found in catalog. Using current source redshift = ' + str(self.z_source))
                z_source = self.z_source
                
            for i in tqdm(range(len(cat))) :
                z = cat[z_colname][i] if z_source is None else z_source
                if z>self.z_lens : # Also returns False if z is np.nan
                    #print(str(cat['id'][i]) + ': computing lensing maps at redshift ' + str(z))
                    ra, dec = cat['ra'][i], cat['dec'][i]
                    world_coord = SkyCoord(ra, dec, unit='deg')
                    xr, yr = self.world_to_relative(ra, dec)
                    #delta = 1.
                    delta = self.image.pix_deg_scale * 3600 / 2
                    self.lt.set_field([xr-delta, xr+delta, yr-delta, yr+delta])
                    
                    #npix = 11
                    npix = 2
                    ampli, wcs = self.lt.g_ampli(1, npix, z)
                    mu_col[i] = np.mean(ampli)
                    
                    #if False : # this crashes Lenstool if 
                    try :
                        kappa, wcs = self.lt.g_mass(1, npix, self.z_lens, z)
                    except ZeroDivisionError :
                        self.lt.set_field([xr-1., xr+1., yr-1., yr+1.])
                        kappa, wcs = self.lt.g_mass(1, 3, self.z_lens, z)
                        self.lt.set_field([xr-delta, xr+delta, yr-delta, yr+delta])
                    
                    kappa_col[i] = np.mean(kappa)
                    #gamma_col[i] = ( (1-kappa_col[i])**2 - 1/mu_col[i] )**0.5
                    
                    dx, dy, wcs = self.lt.g_dpl(npix, z)
                    dx = np.mean(dx)
                    dy = np.mean(dy)                    
                    ra_source_col[i] = ra + dx /3600 /np.cos( dec * np.pi/180 )
                    dec_source_col[i] = dec - dy /3600

                    time, wcs = self.lt.g_time(1, npix, z)
                    time_col[i] = np.mean(time)

                    shear, wcs = self.lt.g_shear(1, npix, z)
                    gamma_col[i] = np.mean(shear)

                    shear1, wcs = self.lt.g_shear(3, npix, z)
                    gamma1_col[i] = np.mean(shear1)

                    shear2, wcs = self.lt.g_shear(4, npix, z)
                    gamma2_col[i] = np.mean(shear2)

                #else :
                    #print(str(cat['id'][i]) + ': redshift ' + str(z) + ' lower than lens redshift --> NaN')
                
                tmu_col[i] = 1 / (1 - kappa_col[i] - gamma_col[i] )
                rmu_col[i] = 1 / (1 - kappa_col[i] + gamma_col[i] )
            
            columns_to_add = [mu_col, kappa_col, gamma_col, gamma1_col, gamma2_col, time_col, tmu_col, rmu_col, ra_source_col, dec_source_col]
            for i, name in enumerate(lensing_columns) :
                if name in cat.colnames :
                    cat.replace_column(name, columns_to_add[i])
                else :
                    cat.add_column(columns_to_add[i], name=name)
            self.lt.set_field(initial_field)
    
    def compute_lt_curve(self, z=None, limitHigh=0.5, limitLow=0.1) :
        if z==None :
            z = self.z_source
        self._vprint('Computing critical curve (can take a little while)...')
        self.lt_curve = self.lt.criticnew(zs=z, limitHigh=limitHigh, limitLow=limitLow) #limitHigh=1., limitLow=0.05
        
        ni = len(self.lt_curve[0])
        ne = len(self.lt_curve[1])
        lt_curve_xr = np.zeros(ni + ne)
        lt_curve_yr = np.zeros(ni + ne)
        for i in range(ni) :
            lt_curve_xr[i] = self.lt_curve[0][i].I.x
            lt_curve_yr[i] = self.lt_curve[0][i].I.y
        for i in range(ne) :
            lt_curve_xr[ni+i] = self.lt_curve[1][i].I.x
            lt_curve_yr[ni+i] = self.lt_curve[1][i].I.y
            
        lt_curve_ra, lt_curve_dec = self.relative_to_world(lt_curve_xr, lt_curve_yr)
        lt_curve_x, lt_curve_y = self.image.world_to_image(lt_curve_ra, lt_curve_dec)
        self.lt_curve_coords_image = [lt_curve_x, self.image.image_data.shape[0] - lt_curve_y]
        self.lt_curve_coords_world = [lt_curve_ra, lt_curve_dec]
        self.lt_curve_coords_relative = [lt_curve_xr, lt_curve_yr]
        self._vprint('done')
        
        self.lt_curves[z] = self.lt_curve_coords_relative
        with open(self.lt_curves_path, 'wb') as f:
            pickle.dump(self.lt_curves, f)
        
        
        ###### Caustics ######
        ni = len(self.lt_curve[0])
        ne = len(self.lt_curve[1])
        lt_caustic_xr = np.zeros(ni + ne)
        lt_caustic_yr = np.zeros(ni + ne)
        for i in range(ni) :
            lt_caustic_xr[i] = self.lt_curve[0][i].S.x
            lt_caustic_yr[i] = self.lt_curve[0][i].S.y
        for i in range(ne) :
            lt_caustic_xr[ni+i] = self.lt_curve[1][i].S.x
            lt_caustic_yr[ni+i] = self.lt_curve[1][i].S.y
            
        lt_caustic_ra, lt_caustic_dec = self.relative_to_world(lt_caustic_xr, lt_caustic_yr)
        lt_caustic_x, lt_caustic_y = self.image.world_to_image(lt_caustic_ra, lt_caustic_dec)
        self.lt_caustic_coords_image = [lt_caustic_x, self.image.image_data.shape[0] - lt_caustic_y]
        self.lt_caustic_coords_world = [lt_caustic_ra, lt_caustic_dec]
        self.lt_caustic_coords_relative = [lt_caustic_xr, lt_caustic_yr]
        self._vprint('done')
        
        self.lt_caustics[z] = self.lt_caustic_coords_relative
        with open(self.lt_caustics_path, 'wb') as f:
            pickle.dump(self.lt_caustics, f)
        
        self.plot_lt_curve()
    
    def _curves_add_all_coords(self) :
        lt_curve_xr, lt_curve_yr = self.lt_curve_coords_relative
        lt_curve_ra, lt_curve_dec = self.relative_to_world(lt_curve_xr, lt_curve_yr)
        lt_curve_x, lt_curve_y = self.image.world_to_image(lt_curve_ra, lt_curve_dec)
        
        self.lt_curve_coords_world = [lt_curve_ra, lt_curve_dec]
        self.lt_curve_coords_image = [lt_curve_x, self.image.image_data.shape[0] - lt_curve_y]
        
        lt_caustic_xr, lt_caustic_yr = self.lt_caustic_coords_relative
        lt_caustic_ra, lt_caustic_dec = self.relative_to_world(lt_caustic_xr, lt_caustic_yr)
        lt_caustic_x, lt_caustic_y = self.image.world_to_image(lt_caustic_ra, lt_caustic_dec)
        
        self.lt_caustic_coords_world = [lt_caustic_ra, lt_caustic_dec]
        self.lt_caustic_coords_image = [lt_caustic_x, self.image.image_data.shape[0] - lt_caustic_y]
    
    def plot_lt_curve(self, color=[255, 0, 255], which='critical') :
        attr_name = 'critical_curve_plot' if which=='critical' else 'caustic_curve_plot'
        existing_plot = getattr(self, attr_name)
        if existing_plot is not None :
            self.image.ImageView.removeItem(existing_plot)
        
        if which=='critical' :
            coords = self.lt_curve_coords_image
        elif which=='caustic' :
            coords = self.lt_caustic_coords_image
        
        curve_coords_image_sorted = break_curves(coords)
        #curve_coords_image_sorted = sort_points(coords, distance_threshold=1.0/(self.image.pix_deg_scale*3600), angle_threshold=np.pi)
        if which=='critical' :
            self.lt_curve_coords_image_sorted = curve_coords_image_sorted
        
        new_plot = pg.PlotDataItem()
        new_plot.setPen( color=color+[255], width=4.0001 )
        new_plot.setData(curve_coords_image_sorted[0], curve_coords_image_sorted[1])
        self.image.ImageView.addItem(new_plot)
        setattr(self, attr_name, new_plot)

    def clear_lt_curve(self, which='critical') :
        """Remove the currently plotted critical or caustic curve, without recomputing anything."""
        attr_name = 'critical_curve_plot' if which=='critical' else 'caustic_curve_plot'
        existing_plot = getattr(self, attr_name)
        if existing_plot is not None :
            self.image.ImageView.removeItem(existing_plot)
            setattr(self, attr_name, None)

    def plot_convergence(self) :
        self._plot_map(self.kappa, self._maps_wcs)
    def plot_magnification(self) :
        data = np.abs(self.mu) if self.magnification_abs else self.mu
        self._plot_map(data, self._maps_wcs)
    def plot_tangential_magnification(self) :
        self._plot_map(self.rmu, self._maps_wcs)
    def plot_radial_magnification(self) :
        self._plot_map(self.tmu, self._maps_wcs)
    def plot_shear(self) :
        self._plot_map(self.shear, self._maps_wcs)
        
    def _plot_map(self, map_data, wcs, cmap=None, opacity=None, levels=None, show_histogram=True) :
        """
        Overlay a 2D map (e.g. convergence, magnification, shear) on top of the
        image currently displayed, positioned according to its WCS. The map gets
        its own interactive histogram/LUT, placed to the left of the image's
        histogram, which can be shown/hidden with the '<' button of the viewer.
        Any previously plotted map is replaced.
        Args:
            map_data: 2D array, indexed as map_data[y, x] (FITS convention).
                      NaN and infinite values are displayed as transparent.
            wcs: astropy WCS (or FITS header) of the map.
            cmap: name of a pyqtgraph/matplotlib colormap, or a pg.ColorMap.
            opacity: opacity of the map, between 0 and 1.
            levels: (min, max) color levels. Defaults to the 1st and 99th
                    percentiles of the finite values.
            show_histogram: whether the map histogram is initially visible.
        Returns:
            map_plot: the pg.ImageItem displaying the map
        """
        if cmap is None :
            cmap = self.map_cmap
        if opacity is None :
            opacity = self.map_opacity
        data = np.array(map_data, dtype=float)
        if data.ndim != 2 :
            raise ValueError(f"map_data must be a 2D array, got shape {data.shape}")
        data[~np.isfinite(data)] = np.nan
        if not np.isfinite(data).any() :
            raise ValueError("map_data contains no finite values")
        map_wcs = (wcs if isinstance(wcs, WCS) else WCS(wcs)).celestial

        self.image.plot_image()
        self._clear_map()
        image_view = self.image.ImageView

        self.current_map_ImageItem = get_ImageItem_projected_onto_reference(self.image.image_data, self.image.wcs, data, map_wcs)
        
        self.current_map_ImageItem.setOpacity(opacity)
        image_view.addItem(self.current_map_ImageItem)
        # Stack the map right above the image, below catalogs/curves added earlier
        siblings = image_view.imageItem.parentItem().childItems()
        i = siblings.index(image_view.imageItem)
        if i + 1 < len(siblings) and siblings[i+1] is not self.current_map_ImageItem :
            self.current_map_ImageItem.stackBefore(siblings[i+1])

        self.map_histogram = pg.HistogramLUTWidget(image=self.current_map_ImageItem)
        colormap = get_colormap(cmap)
        self.map_histogram.gradient.setColorMap(colormap)
        if levels is None :
            levels = np.nanpercentile(data, [1., 99.])
        self.map_histogram.setLevels(*levels)
        # Shift the image histogram and its buttons one column to the right and
        # insert the map histogram in column 1, i.e. to the left of the image histogram
        self._set_image_histogram_shift(image_view, True)
        image_view.ui.gridLayout.addWidget(self.map_histogram, 0, 1, 2, 1)
        self.map_histogram.setVisible(show_histogram)

        self._map_histogram_toggle_btn = self.image._make_floating_toggle_button(
            self.image.QWidget,
            '<',
            'top-right',
            lambda : self.map_histogram.setVisible(not self.map_histogram.isVisible()),
            x_margin=70,
            y_margin=8,
        )
        return self.current_map_ImageItem

    def _apply_map_style(self) :
        """Update colormap and opacity on the current map overlay, if any."""
        if self.current_map_ImageItem is None :
            return
        self.current_map_ImageItem.setOpacity(self.map_opacity)
        if self.map_histogram is None :
            return
        colormap = get_colormap(self.map_cmap)
        self.map_histogram.gradient.setColorMap(colormap)

    @staticmethod
    def _set_image_histogram_shift(image_view, shifted) :
        """
        Move the image histogram (and the ROI/menu buttons below it) of a pyqtgraph
        ImageView one column to the right (shifted=True) to make room for the map
        histogram in column 1, or put them back to their default position (shifted=False).
        """
        ui = image_view.ui
        layout = ui.gridLayout
        offset = 1 if shifted else 0
        for widget, row, col, row_span, col_span in [(ui.histogram, 0, 1, 1, 2),
                                                     (ui.roiBtn, 1, 1, 1, 1),
                                                     (ui.menuBtn, 1, 2, 1, 1)] :
            layout.removeWidget(widget)
            layout.addWidget(widget, row, col + offset, row_span, col_span)

    def _clear_map(self) :
        """Remove the map plotted with plot_map(), along with its histogram and toggle button."""
        try :
            if self.current_map_ImageItem is not None and self.image.ImageView is not None :
                self.image.ImageView.removeItem(self.current_map_ImageItem)
            if self.map_histogram is not None :
                self.map_histogram.setParent(None)
                self.map_histogram.deleteLater()
                if self.image.ImageView is not None :
                    self._set_image_histogram_shift(self.image.ImageView, False)
            if self._map_histogram_toggle_btn is not None :
                if self.image.QWidget is not None :
                    self.image.QWidget.remove_floating_widget(self._map_histogram_toggle_btn)
                self._map_histogram_toggle_btn.deleteLater()
        except RuntimeError :
            # The underlying Qt objects were already deleted (e.g. the window was closed)
            pass
        self.current_map_ImageItem = None
        self.map_histogram = None
        self._map_histogram_toggle_btn = None
        
    def plot_bayes(self) :
        plot_corner(self.samples_df)
        corr_matrix = self.samples_df.corr()
        self.fig_cov, self.ax_cov = plt.subplots()
        cax = self.ax_cov.imshow(corr_matrix, cmap='PuOr')
        cbar = self.fig_cov.colorbar(cax, ax=self.ax_cov)
        self.ax_cov.set_xticks(np.arange(len(corr_matrix.columns)))
        self.ax_cov.set_yticks(np.arange(len(corr_matrix.index)))
        self.ax_cov.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        self.ax_cov.set_yticklabels(corr_matrix.index)
    
    def filter_image(self, threshold_arcsec=0.1) :
        if self.mult is not None :
            threshold_pix = threshold_arcsec / 3600 / self.image.pix_deg_scale
            to_remove = []
            for i, image in enumerate(self.images_filtered.cat) :
                ref_mask = self.mult.cat['id']==image['id']
                #if not ref_mask.any() :
                #    d = 0
                #else :
                ref = self.mult.cat[ np.where(ref_mask)[0][0] ]
                d = ( (ref['x'] - image['x'])**2 + (ref['y'] - image['y'])**2 )**0.5
                if d<threshold_pix :
                    to_remove.append(i)
            self.images_filtered.cat.remove_rows(to_remove)
        
        ### Grouping of images with similar positions, not used anymore ###
        if False :
            threshold_pix = threshold_arcsec / 3600 / self.image.pix_deg_scale
            
            N = len(self.images_filtered.cat)
            distance_matrix = np.zeros((N, N))
            for i in range(N) :
                for j in range(N) :
                    im_i = self.images_filtered.cat[i]
                    im_j = self.images_filtered.cat[j]
                    distance_matrix[i, j] = ( (im_i['x'] - im_j['x'])**2 + (im_i['y'] - im_j['y'])**2 )**0.5
                    #if i==j :
                    #    distance_matrix[i, j] = np.nan
            to_group_matrix = np.zeros((N, N))
            #to_groug_matrix[ np.logical_and(distance_matrix<threshold_pix, distance_matrix!=0.) ] = 1
            to_group_matrix[ distance_matrix<threshold_pix ] = 1.
            
            def find_related_groups(matrix):
                N = len(matrix)
                visited = [False] * N
                groups = []
                def dfs(node, group):
                    visited[node] = True
                    group.append(node)
                    for neighbor in range(N):
                        if matrix[node][neighbor] == 1 and not visited[neighbor]:
                            dfs(neighbor, group)
                for i in range(N):
                    if not visited[i]:
                        group = []
                        dfs(i, group)
                        groups.append(group)
                return groups
            groups = find_related_groups(to_group_matrix)
            
            to_remove = []
            for i, group in enumerate(groups) :
                x_mean = np.mean(self.images_filtered.cat['x'][group])
                y_mean = np.mean(self.images_filtered.cat['y'][group])
                self.images_filtered.cat[group[0]]['x'] = x_mean
                self.images_filtered.cat[group[0]]['y'] = y_mean
                to_remove += list(np.array(group)[1:])
            self.images_filtered.cat.remove_rows(to_remove)
    
    
    def start_extract_magnification_line(self) :
        self.doubleclick_magnification_marker = pg.ScatterPlotItem(size=12, symbol='x', brush='b', pen='b')
        self.source_magnification_marker = pg.ScatterPlotItem(size=8, symbol='o', brush='y', pen='y')
        self.image.ImageView.addItem(self.doubleclick_magnification_marker)
        self.image.ImageView.addItem(self.source_magnification_marker)
        self.magnification_markers_x = []
        self.magnification_markers_y = []
        self.magnification_source_markers_x = []
        self.magnification_source_markers_y = []
        self.magnification_temp_SkyCoords = []
        
        def mouse_clicked(evt):
            if evt.double():
                pos = evt.scenePos()
                if self.image.ImageView.getView().sceneBoundingRect().contains(pos):
                    if len(self.magnification_temp_SkyCoords)==2 :
                        self.magnification_markers_x = []
                        self.magnification_markers_y = []
                        self.magnification_source_markers_x = []
                        self.magnification_source_markers_y = []
                        self.magnification_temp_SkyCoords = []
                        self.doubleclick_magnification_marker.setData([], [])
                        self.source_magnification_marker.setData([], [])
                    
                    mouse_point = self.image.ImageView.getView().mapSceneToView(pos)
                    x, y_flipped = mouse_point.x(), mouse_point.y()
                    x, y = x, self.image.image_data.shape[0] - y_flipped
                    ra, dec = self.image.image_to_world(x, y)
                    
                    self.magnification_markers_x.append(x)
                    self.magnification_markers_y.append(self.image.image_data.shape[0] - y)
                    self.magnification_temp_SkyCoords.append(SkyCoord(ra, dec, unit='deg'))
                    self.doubleclick_magnification_marker.setData(self.magnification_markers_x, self.magnification_markers_y)
                    
                    start = WCS.world_to_pixel(self.magnification_wcs, self.magnification_temp_SkyCoords[0])
                    self.magnification_line_start = (start[0]*1., start[1]*1.)
                    if len(self.magnification_temp_SkyCoords)==2 :
                        end = WCS.world_to_pixel(self.magnification_wcs, self.magnification_temp_SkyCoords[1])
                        self.magnification_line_end = (end[0]*1., end[1]*1.)
                        self.magnification_line = extract_line( self.magnification_line_start, self.magnification_line_end, self.magnification_map )
                        magnification_wcs = self.magnification_wcs
                        cd = magnification_wcs.wcs.cdelt[np.newaxis, :] * magnification_wcs.wcs.pc
                        deg_per_pix = np.sqrt((cd**2).sum(axis=0))[0]
                        self.magnification_line[0] = np.array(self.magnification_line[0]) * deg_per_pix * 3600 #x axis in arcsec
                        if True : #self.magnification_line_ax==None :
                            self.magnification_line_distances = []
                            
                            plt.close()
                            self._vprint('Creating new magnification plot')
                            self.magnification_line_fig, self.magnification_line_ax = plt.subplots()
                            self.magnification_line_ax.set_yscale('log')
                            
                            #self.magnification_line_ax_xlim = self.magnification_line_ax.get_xlim()
                            #self.magnification_line_ax_ylim = self.magnification_line_ax.get_ylim()
                            
                        self.magnification_line_ax.clear()
                        self.magnification_line_ax.plot(self.magnification_line[0], np.abs(self.magnification_line[1]))
                        #self.magnification_line_fig.show()
            elif evt.button()==PyQt5.QtCore.Qt.MiddleButton :
                pos = evt.scenePos()
                self._vprint(pos)
                mouse_point = self.image.ImageView.getView().mapSceneToView(pos)
                x, y_flipped = mouse_point.x(), mouse_point.y()
                x, y = x, self.image.image_data.shape[0] - y_flipped
                self.magnification_source_markers_x.append(x)
                self.magnification_source_markers_y.append(self.image.image_data.shape[0] - y)
                self.source_magnification_marker.setData(self.magnification_source_markers_x, self.magnification_source_markers_y)
                
                distance = ( (self.magnification_markers_x[0] - x)**2 + (self.magnification_markers_y[0] - y_flipped)**2 )**0.5 * self.image.pix_deg_scale*3600 #in arcsec
                self.magnification_line_distances.append(distance)
                
                if True : # remove this when plot update available
                    plt.close()
                    self._vprint('Creating new magnification plot')
                    self.magnification_line_fig, self.magnification_line_ax = plt.subplots()
                    self.magnification_line_ax.set_yscale('log')
                    self.magnification_line_ax.plot(self.magnification_line[0], np.abs(self.magnification_line[1]))
                    
                    #self.magnification_line_ax_xlim = self.magnification_line_ax.get_xlim()
                    #self.magnification_line_ax_ylim = self.magnification_line_ax.get_ylim()
                    self.magnification_line_ax.set_xlim(self.magnification_line_ax.get_xlim())
                    self.magnification_line_ax.set_ylim(self.magnification_line_ax.get_ylim())
                
                for distance in self.magnification_line_distances :
                    self.magnification_line_ax.plot(np.full(10, distance), np.linspace(0, np.max(self.magnification_line[1]), 10), ls='--', c='grey')
                
                
        
        self._doubleclick_connection = self.image.ImageView.scene.sigMouseClicked.connect(mouse_clicked)
        
        def keyPressEvent(event):
            #print('Hand selection stopped.')
            if event.key() == Qt.Key_Escape :
                if hasattr(self, 'doubleclick_magnification_marker'):
                    self.image.ImageView.removeItem(self.doubleclick_magnification_marker)
                    del self.doubleclick_magnification_marker
                if hasattr(self, 'source_magnification_marker'):
                    self.image.ImageView.removeItem(self.source_magnification_marker)
                    del self.source_magnification_marker
                if hasattr(self, '_doubleclick_connection'):
                    self.image.ImageView.scene.sigMouseClicked.disconnect(self._doubleclick_connection)
                    del self._doubleclick_connection
                self.magnification_temp_SkyCoords = []
                self.image.ImageView.keyPressEvent = self._original_keyPressEvent
                self._vprint('Magnification line extraction stopped.')
        
        self._original_keyPressEvent = self.image.ImageView.keyPressEvent
        self.image.ImageView.keyPressEvent = keyPressEvent
    
    
    def send_to_source_plane(self) :
        if self.workspace is None or self.workspace.catalog is None :
            return
        for row in self.workspace.catalog.cat :
            row['ra'], row['dec'] = self.transform_coords_radec(row['ra'], row['dec'])
            row['x'], row['y'] = self.image.world_to_image(row['ra'], row['dec'])
    
    
    def start_simulate_image(self, which_filter=None, throttle_mode=0) :
        self.imsim = image_simulator(self.image, lens_model=self, which_filter=which_filter, throttle_mode=throttle_mode)
        
    
    ######### DEPRECATED: check what to keep #########
    def compute_convergence(self, z=None, npix=2000) :
        z = self.z_source if z is None else z
        self.convergence_map, self.convergence_map_wcs = self.lt.g_mass(1, npix, self.z_lens, z)
        fig, ax = plt.subplots()
        ax.imshow(self.convergence_map, origin='lower')
        return self.convergence_map, self.convergence_map_wcs
    
    def compute_displacement_maps(self, z=None, npix=2000) :
        z = self.z_source if z is None else z
        self.dx_map, self.dy_map, self.dmap_wcs = self.lt.g_dpl(npix, z)
        return self.dx_map, self.dy_map, self.dmap_wcs

    def compute_mass_map(self, z=None, npix=1000) :
        z = self.z_source if z is None else z
        self.mass_map, self.mass_map_wcs = self.lt.g_mass(1, npix, self.z_lens, z)
        fig, ax = plt.subplots()
        ax.imshow(np.arctan(self.mass_map), origin='lower')
        return self.mass_map, self.mass_map_wcs
    
    def compute_magnification_map(self, z=None, npix=1000) :
        z = self.z_source if z is None else z
        self.magnification_map, self.magnification_wcs = self.lt.g_ampli(1, npix, z)
        fig, ax = plt.subplots()
        ax.imshow(np.arctan(np.abs(self.magnification_map)), origin='lower')
        return self.magnification_map, self.magnification_wcs
    
    def compute_time_map(self, z=None, npix=1000) :
        z = self.z_source if z is None else z
        self.time_map, self.time_wcs = self.lt.g_time(1, npix, z)
        fig, ax = plt.subplots()
        ax.imshow(self.time_map, origin='lower')
        return self.time_map, self.time_wcs
    
    def compute_shear_map(self, z=None, npix=1000, which='gamma') :
        z = self.z_source if z is None else z
        ishear = 1 if which=='gamma' else 3 if which=='gamma1' else 4 if which=='gamma2' else None
        if ishear is None :
            raise ValueError('Invalid shear type: ' + which)
        self.shear_map, self.shear_wcs = self.lt.g_shear(ishear, npix, z)
        fig, ax = plt.subplots()
        ax.imshow(self.shear_map, origin='lower')
        return self.shear_map, self.shear_wcs

    def compute_mass(self, ra, dec, z=None) :
        initial_field = self.lt.get_field([])
        z = self.z_source if z is None else z
        xr, yr = self.world_to_relative(ra, dec)
        delta = 0.0001 # 0.1 milli arcsec
        self.lt.set_field([xr-delta, xr+delta, yr-delta, yr+delta])
        npix = 2
        kappa, _ = self.lt.g_mass(1, npix, self.z_lens, z)
        self.lt.set_field(initial_field)
        return np.mean(kappa)
    
    def compute_magnification(self, ra, dec, z=None) :
        initial_field = self.lt.get_field([])
        z = self.z_source if z is None else z
        xr, yr = self.world_to_relative(ra, dec)
        delta = 0.0001 # 0.1 milli arcsec
        self.lt.set_field([xr-delta, xr+delta, yr-delta, yr+delta])
        npix = 2
        ampli, _ = self.lt.g_ampli(1, npix, z)
        self.lt.set_field(initial_field)
        return np.mean(ampli)
    
    def compute_time(self, ra, dec, z=None) :
        initial_field = self.lt.get_field([])
        z = self.z_source if z is None else z
        xr, yr = self.world_to_relative(ra, dec)
        delta = 0.0001 # 0.1 milli arcsec
        self.lt.set_field([xr-delta, xr+delta, yr-delta, yr+delta])
        npix = 2
        time, _ = self.lt.g_time(1, npix, z)
        self.lt.set_field(initial_field)
        return np.mean(time)
    
    def compute_shear(self, ra, dec, z=None, which='gamma') :
        initial_field = self.lt.get_field([])
        z = self.z_source if z is None else z
        ishear = 1 if which=='gamma' else 3 if which=='gamma1' else 4 if which=='gamma2' else None
        xr, yr = self.world_to_relative(ra, dec)
        delta = 0.0001 # 0.1 milli arcsec
        self.lt.set_field([xr-delta, xr+delta, yr-delta, yr+delta])
        npix = 2
        shear, _ = self.lt.g_shear(ishear, npix, z)
        self.lt.set_field(initial_field)
        return np.mean(shear)
        
    def compute_displacement(self, ra, dec, z=None) :
        initial_field = self.lt.get_field([])
        z = self.z_source if z is None else z
        xr, yr = self.world_to_relative(ra, dec)
        delta = 0.0001 # 0.1 milli arcsec
        self.lt.set_field([xr-delta, xr+delta, yr-delta, yr+delta])
        npix = 2
        dx, dy, _ = self.lt.g_dpl(npix, z)
        self.lt.set_field(initial_field)
        return np.mean(dx), np.mean(dy)
    
    ######### END OF DEPRECATED #########
    

    def set_field(self) :
        # Careful, this function only works when the ROI is flat and image in world frame
        
        self.ROI = self.image.QWidget.current_ROI
        x0 = self.ROI.getState()['pos'][0]
        y0 = self.ROI.getState()['pos'][1]
        a = self.ROI.getState()['size'][0]
        b = self.ROI.getState()['size'][1]
        angle = self.ROI.getState()['angle'] *np.pi/180
        
        x0, y0, a, b, angle = transform_rectangle(x0, y0, a, b, angle) #x0, y0 at the top left
        size_y = self.image.image_data.shape[0]
        y0 = size_y-y0 #Counting pixels from bottom instead of top
        
        ra_left, dec_top = self.image.image_to_world(x0, y0)
        ra_right, dec_bottom = self.image.image_to_world(x0 + a, y0 - b)
        
        xr_left, yr_top = self.world_to_relative(ra_left, dec_top)
        xr_right, yr_bottom = self.world_to_relative(ra_right, dec_bottom)
        
        self._previous_field = self.lt.get_field([])
        self.lt.set_field([xr_left, xr_right, yr_bottom, yr_top])
        
    
    def compute_uncertainties(self, nsamples=None, recompute_samples=False) :
        self.samples_dir = os.path.join(self.model_dir, 'samples')
        if not os.path.exists(self.samples_dir) :
            self._vprint('Creating samples directory')
            os.makedirs(self.samples_dir)
        
        lensing_columns = [ 'magnification', 
                            'convergence', 
                            'shear', 
                            'gamma1', 
                            'gamma2', 
                            'time', 
                            'tangential_magnification', 
                            'radial_magnification', 
                            'ra_source', 
                            'dec_source',
                            'z_opt' ]
        
        samples_dict_path = os.path.join(self.samples_dir, 'samples_dict.pkl')
        if os.path.exists(samples_dict_path) and not recompute_samples :
            self._vprint('Loading samples dictionary from ' + samples_dict_path)
            with open(samples_dict_path, 'rb') as f :
                self.samples_dict = pickle.load(f)

        else :
            if os.path.exists(samples_dict_path) :
                # Computation of lensing properties for all samples takes time so we save previous dictionary just in case
                self._vprint('Renaming previous samples dictionary to ' + samples_dict_path.replace('.pkl', '_previous.pkl'))
                shutil.move(samples_dict_path, samples_dict_path.replace('.pkl', '_previous.pkl'))

            # if os.path.exists(os.path.join(self.samples_dir, os.path.basename(self.mult_path))) :
            #     self._vprint('Overwriting mult file in samples directory')
            # else :
            #     self._vprint('Copying mult file to samples directory')
            # shutil.copy2(self.mult_path, self.samples_dir)
            
            nsamples = self.lt._nvals # len(self.samples_df.index) if nsamples is None else nsamples
            
            self.samples_dict = {}
            for imID in self.mult.cat['id'] :
                self.samples_dict[imID] = {col: np.full(nsamples, np.nan) for col in lensing_columns}
                
            for i in tqdm(range(nsamples)) :
                self.lt.setBayesModel(i)


                # sample_file_path = write_single_sample_best_file(self, i)
                # sample_lt = import_lenstool(sample_file_path, self.image, compute_predictions=False, verbose=False)

                """ Sample the redshift as well for those that were optimized """
                for im in self.mult.cat :
                    #if not np.isnan(im['z_opt']) :
                    broad_family_members_mask = np.array([ member['broad_family']==im['broad_family'] for member in self.mult.cat ])
                    for member in self.mult.cat[broad_family_members_mask] :
                        for col in self.samples_table.colnames :
                            if member['id'] == col[len('Redshift of '):] :
                                self._vprint('Sampling redshift of ' + im['id'] + ': ' + str(im['z']) + ' --> ' + str(self.samples_table[col][i]))
                                im['z'] = self.samples_table[col][i]
                                self.samples_dict[im['id']]['z_opt'][i] = self.samples_table[col][i]
                                
                """ Compute magnification etc. for the sample """
                self.lt.add_lensing_columns(cat=self.mult.cat)

                for im in self.mult.cat :
                    for col in lensing_columns :
                        if col != 'z_opt' :
                            self.samples_dict[im['id']][col][i] = im[col]
                    
                # del sample_lt
                # gc.collect()
                # os.remove(sample_file_path)

                if i % 10 == 0 :
                    pickle.dump(self.samples_dict, open(samples_dict_path, 'wb'))
            
            pickle.dump(self.samples_dict, open(samples_dict_path, 'wb'))
        
        for col in lensing_columns :
            col_16_percentile = np.full(len(self.mult.cat), np.nan)
            col_84_percentile = np.full(len(self.mult.cat), np.nan)
            col_50_percentile = np.full(len(self.mult.cat), np.nan)

            name = f'{col}_16_percentile'
            if name in self.mult.cat.colnames :
                self.mult.cat.replace_column(name, col_16_percentile)
            else :
                self.mult.cat.add_column(col_16_percentile, name=name)

            name = f'{col}_84_percentile'
            if name in self.mult.cat.colnames :
                self.mult.cat.replace_column(name, col_84_percentile)
            else :
                self.mult.cat.add_column(col_84_percentile, name=name)

            name = f'{col}_50_percentile'
            if name in self.mult.cat.colnames :
                self.mult.cat.replace_column(name, col_50_percentile)
            else :
                self.mult.cat.add_column(col_50_percentile, name=name)

            for im in self.mult.cat :
                im[f'{col}_16_percentile'] = np.percentile(self.samples_dict[im['id']][col], 16)
                im[f'{col}_84_percentile'] = np.percentile(self.samples_dict[im['id']][col], 84)
                im[f'{col}_50_percentile'] = np.percentile(self.samples_dict[im['id']][col], 50)

        self.lt.setBayesModel(method=-4)
        

    def compute_map_as(self, fits_file_path, which='dpl') :
        with fits.open(fits_file_path) as hdul :
            data_hdu = None
            for hdu in hdul :
                if getattr(hdu, 'data', None) is not None :
                    data_hdu = hdu
                    break
            if data_hdu is None :
                raise ValueError(f"No image data found in FITS file: {fits_file_path}")
            
            target_header = data_hdu.header
            target_data = data_hdu.data
            target_wcs = WCS(target_header)
        
        ndim = target_data.ndim
        if ndim > 2 :
            ny, nx = target_data.shape[-2], target_data.shape[-1]
        else :
            ny, nx = target_data.shape
        
        which_norm = which.lower().strip()
        if which_norm in ['displacement', 'disp'] :
            which_norm = 'dpl'
        elif which_norm in ['magnification', 'mu', 'ampli'] :
            which_norm = 'magnification'
        elif which_norm in ['convergence', 'kappa', 'mass'] :
            which_norm = 'convergence'
        elif which_norm in ['shear', 'gamma'] :
            which_norm = 'shear'
        
        valid_which = ['dpl', 'magnification', 'shear', 'convergence']
        if which_norm not in valid_which :
            raise ValueError(
                f"Invalid 'which' value: {which}. "
                f"Allowed values are: {valid_which}"
            )
        
        if which_norm == 'dpl' :
            map_data = np.full((2, ny, nx), np.nan, dtype=float)
        else :
            map_data = np.full((ny, nx), np.nan, dtype=float)
        self._vprint(f"Computing '{which_norm}' map on input pixel grid ({ny} x {nx})...")
        for iy in tqdm(range(ny), disable=not self.verbose) :
            for ix in range(nx) :
                if ndim == 3 :
                    ra, dec, _ = target_wcs.pixel_to_world_values(ix, iy, 0.)
                else :
                    ra, dec = target_wcs.pixel_to_world_values(ix, iy)
                    
                if which_norm == 'dpl' :
                    dx, dy = self.compute_displacement(ra, dec)
                    map_data[0, iy, ix] = dx
                    map_data[1, iy, ix] = dy
                elif which_norm == 'magnification' :
                    map_data[iy, ix] = self.compute_magnification(ra, dec)
                elif which_norm == 'shear' :
                    map_data[iy, ix] = self.compute_shear(ra, dec, which='gamma')
                elif which_norm == 'convergence' :
                    map_data[iy, ix] = self.compute_mass(ra, dec)
        
        out_header = target_wcs.to_header()
        out_header['BUNIT'] = 'arcsec' if which_norm == 'dpl' else ''
        out_header['MAPTYPE'] = which_norm
                
        input_dir = os.path.dirname(fits_file_path)
        input_name = os.path.basename(fits_file_path)
        root_name, _ = os.path.splitext(input_name)
        out_path = os.path.join(input_dir, f"{root_name}_{which_norm}_map.fits")
        
        if which_norm == 'dpl' :
            fits.PrimaryHDU(data=map_data[0], header=out_header).writeto(out_path.replace('.fits', '_dx.fits'), overwrite=True)
            fits.PrimaryHDU(data=map_data[1], header=out_header).writeto(out_path.replace('.fits', '_dy.fits'), overwrite=True)
        else :
            fits.PrimaryHDU(data=map_data, header=out_header).writeto(out_path, overwrite=True)
        self._vprint(f"Saved map to {out_path}")
        
        arcsec_per_pixel = target_wcs.wcs.cdelt[0] * 3600
        def plot_source_plane_grid(dpl_maps) :
            # Careful: this function only works when x, y match ra, dec axes
            fig, ax = plt.subplots()
            x, y = np.meshgrid(np.arange(0, dpl_maps[0].shape[1], 1), np.arange(0, dpl_maps[0].shape[0], 1))
            x_im = x * arcsec_per_pixel
            y_im = y * arcsec_per_pixel
            x_src = x_im - dpl_maps[0]
            y_src = y_im - dpl_maps[1]
    
            ax.scatter(x_src, y_src, marker = '.', sizes=[0.5], alpha=0.5)
            ax.set_xlabel('x (arcsec)')
            ax.set_ylabel('y (arcsec)')
            ax.set_title('Data grid in the source plane')
            plt.show()
        
        if which_norm == 'dpl' :
            plot_source_plane_grid(map_data)
        
        return map_data, target_wcs

    def _get_cosmo(self) :
        if getattr(self, 'lt', None) is not None :
            return FlatLambdaCDM(H0=self.lt.C.H0, Om0=self.lt.C.omegaM)
        else :
            return get_default_cosmology()
    
    def get_DlsDs(self, z=None) :
        if getattr(self, 'cosmo', None) is not None and getattr(self, 'z_lens', None) is not None :
            z = self.z_source if z is None else z
            Dls = self.cosmo.angular_diameter_distance_z1z2(self.z_lens, z)
            Ds = self.cosmo.angular_diameter_distance_z1z2(0., z)
            return Dls, Ds
        else :
            raise ValueError('Cosmology or lens redshift not set')
    
    def _update_z_source(self) :
        self.Dls, self.Ds = self.get_DlsDs()
        self.DlsDs = (self.Dls / self.Ds).value
        self.Sigma_crit = astro_constants.c**2 / (4 * np.pi * astro_constants.G) * self.Ds / (self.Dl * self.Dls)

        if getattr(self, 'alpha_inf', None) is not None :
            self.alpha = self.alpha_inf * self.DlsDs
            self.kappa = self.kappa_inf * self.DlsDs
            self.gamma = self.gamma_inf * self.DlsDs
            self.shear = self.shear_inf * self.DlsDs

            self.tmu = 1 / (1 - self.kappa - self.shear)
            self.rmu = 1 / (1 - self.kappa + self.shear)
            self.mu = 1 / ( (1 - self.kappa)**2 - self.shear**2 )



def import_lenstool(model_dir, image=None, compute_predictions=False, verbose=True) :
    return LensModel(model_dir, image, compute_predictions=compute_predictions, verbose=verbose)











