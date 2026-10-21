import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import copy
import corner

from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Data.pixel_grid import PixelGrid  
from lenstronomy.Util import param_util
        
from ...utils.utils_astro.utils_general import world_to_relative, relative_to_world
from ...utils.utils_Qt.Special_QWidgets import (
    SelectableCircleROI,
    SelectableEllipseROI,
    attach_sliders,
    _update_dpos_square_overlay,
    _roi_link_box_slider,
)
from ...utils.utils_Qt.utils_general import transform_ROI_params_inverse, make_handles
from .extract_samples import make_samples_dict
from .utils_LaTeX import make_lm_LaTeX, make_lm_table, table_to_LaTeX_str


def _apply_kwargs_bounds_to_roi_sliders(roi, lo, hi, eps=1e-8, plot_widget=None):
    """
    Apply ``kwargs_lower`` / ``kwargs_upper`` entries to the ROI's TripleSliders.

    Lenstronomy names are mapped to slider keys (e.g. ``source_amp`` -> ``amp``).
    Expands each slider's ``min_range`` / ``max_range`` if the requested bounds
    lie outside the defaults from ``get_light_model_ranges``.
    """
    if not isinstance(lo, dict) or not isinstance(hi, dict):
        return
    sliders = getattr(roi, 'sliders', None)
    if not sliders:
        return

    def set_triple_bounds(s, lo_v, hi_v):
        lo_v, hi_v = float(lo_v), float(hi_v)
        if lo_v >= hi_v:
            return
        s.min_range = min(s.min_range, lo_v)
        s.max_range = max(s.max_range, hi_v)
        s.vmin = lo_v
        s.vmax = hi_v
        s.vmid = max(min(s.vmid, s.vmax - eps), s.vmin + eps)
        s.update()

    for k_kwargs, k_slider in (
        ('amp', 'amp'),
        ('n_sersic', 'n_sersic'),
        ('sigma', 'sigma'),
        ('R_sersic', 'R_sersic'),
        ('sigma0', 'sigma0'),
        ('Ra', 'Ra'),
        ('Rs', 'Rs'),
        ('q', 'q'),
        ('phi', 'phi'),
        ('source_amp', 'amp'),
    ):
        if k_kwargs in lo and k_kwargs in hi and k_slider in sliders:
            set_triple_bounds(sliders[k_slider], lo[k_kwargs], hi[k_kwargs])

    _, s_link_box = _roi_link_box_slider(roi)
    if s_link_box is not None:
        s = s_link_box
        dpos_val = None
        if all(k in lo and k in hi for k in ('center_x', 'center_y')):
            dx = (float(hi['center_x']) - float(lo['center_x'])) / 2.0
            dy = (float(hi['center_y']) - float(lo['center_y'])) / 2.0
            dpos_val = 0.5 * (dx + dy)
        elif all(k in lo and k in hi for k in ('ra_source', 'dec_source')):
            dx = (float(hi['ra_source']) - float(lo['ra_source'])) / 2.0
            dy = (float(hi['dec_source']) - float(lo['dec_source'])) / 2.0
            dpos_val = 0.5 * (dx + dy)
        if dpos_val is not None and dpos_val > 0:
            s.vmid = max(min(dpos_val, s.max_range - eps), s.min_range + eps)
            s.update()

    # Programmatic dpos changes do not emit TripleSlider.valuesChanged; refresh overlay.
    if plot_widget is not None:
        _update_dpos_square_overlay(plot_widget, roi)


class lenstronomy_model :
    def __init__(self, lm_dict_or_str, imsim, chain_list=None) :
        # lm is a dictionary containing the parameters of a source model, either in local or world coordinates.
        self.imsim = imsim
        self.chain_list = chain_list
        if type(lm_dict_or_str)==str :
            lm_dict_or_str = self.load_lm(path=lm_dict_or_str)
        if 'chain_list' in lm_dict_or_str :
            self.chain_list = lm_dict_or_str['chain_list']
            del lm_dict_or_str['chain_list']
        if self._is_local_model_dict(lm_dict_or_str) :
            self.local = lm_dict_or_str
            #self._fix_ps_coord_names()
            self.to_world()
            #self._remove_interpol(self.world)
        else :
            self.world = lm_dict_or_str
            self.to_local()
        
        if self.chain_list is not None :
            steps = [step[0] for step in self.chain_list]
            if 'kwargs_PSO' not in self.local and 'PSO' in steps :
                i = steps.index('PSO')  
                self.local['kwargs_PSO'] = make_samples_dict(self.local, self.chain_list[i])
                self.to_world()
            if 'kwargs_MCMC' not in self.local and 'emcee' in steps :
                i = steps.index('emcee')
                self.local['kwargs_MCMC'] = make_samples_dict(self.local, self.chain_list[i])
                self.to_world()
                
    
    def _fix_ps_coord_names(self) :
        for block in self._iter_kwargs_blocks(self.local) :
            for src in block['kwargs_source_ps'] :
                if 'ra_source' in src and 'dec_source' in src :
                    src['center_x'] = src['ra_source']
                    src['center_y'] = src['dec_source']
                    del src['ra_source']
                    del src['dec_source']
                    
    def _remove_interpol(self, model_dict) :
        if 'INTERPOL' in model_dict['models']['lens_model_list'] :
            index_interpol = model_dict['models']['lens_model_list'].index('INTERPOL')
            model_dict['models']['lens_model_list'].remove('INTERPOL')
            for block in self._iter_kwargs_blocks(model_dict) :
                del block['kwargs_lens'][index_interpol]

    def _add_interpol(self, model_dict) :
        model_dict['models']['lens_model_list'].insert(0, 'INTERPOL')
        for block in self._iter_kwargs_blocks(model_dict) :
            ### To remove, backward compatibility ###
            if not 'kwargs_lens' in block :
                block['kwargs_lens'] = []
            ### End of backward compatibility block ###
            block['kwargs_lens'].insert(0, {})
        model_dict['kwargs']['kwargs_lens'][0] = self.imsim.LensModel_kwargs[0]
        model_dict['kwargs_fixed']['kwargs_lens'][0] = self.imsim.LensModel_kwargs[0]

    def _iter_kwargs_blocks(self, model_dict):
        """
        Yield all sub-dicts that can contain kwargs_source / kwargs_source_ps blocks 
        (e.g. kwargs_init/sigma/lower/upper/fixed, kwargs_opt, kwargs_MCMC, ...).
        """
        for k, v in model_dict.items():
            if k == 'models':
                continue
            if isinstance(v, dict) and ('kwargs_source' in v or 'kwargs_source_ps' in v or 'kwargs_lens' in v):
                yield v

    def _local_to_world_coords(self, d):
        """Convert a single kwargs dict from local-relative to world coordinates."""
        # visualens naming convention: center_x/center_y -> center_ra/center_dec
        if 'center_x' in d and 'center_y' in d:
            ra, dec = relative_to_world(d['center_x'], d['center_y'], self.imsim.center_world)
            d['center_ra'] = ra
            d['center_dec'] = dec
            del d['center_x']
            del d['center_y']

        # lenstronomy naming convention: ra_source/dec_source are local-relative in Lenstronomy
        if 'ra_source' in d and 'dec_source' in d:
            ra, dec = relative_to_world(d['ra_source'], d['dec_source'], self.imsim.center_world)
            d['center_ra'] = ra
            d['center_dec'] = dec
            del d['ra_source']
            del d['dec_source']

    def _world_to_local_coords(self, d):
        """Convert a single kwargs dict from world to local-relative coordinates."""
        if 'center_ra' in d and 'center_dec' in d:
            xr, yr = world_to_relative(d['center_ra'], d['center_dec'], self.imsim.center_world)
            d['center_x'] = xr
            d['center_y'] = yr
            del d['center_ra']
            del d['center_dec']

        #if 'center_ra' in d and 'center_dec' in d:
        #    xr, yr = world_to_relative(d['center_ra'], d['center_dec'], self.imsim.center_world)
        #    d['ra_source'] = xr
        #    d['dec_source'] = yr
        #    del d['center_ra']
        #    del d['center_dec']

    def _is_local_model_dict(self, lm_dict):
        """Local models include INTERPOL by construction."""
        ### To remove, backward compatibility ###
        if 'lens_model_list' in lm_dict['models'] :
            ### End of backward compatibility block ###
            if 'INTERPOL' in lm_dict['models']['lens_model_list']:
                return True
        return False
    
    def plot(self, pix_scale=0.001, size_arcsec=0.4, cmap='grey', with_cbar=True, axes=None, title=None, cutoff=None) :
        if axes is None :
            fig, axes = plt.subplots(4, 1, figsize=(4, 16), sharex=False, sharey=False)
        else :
            fig = axes[0].figure

        if title is not None :
            axes[0].set_title(title)

        """ Observed image """
        observed_image = self.imsim.image_data.copy()
        self.observed_image = observed_image

        """ Image simulated """
        simulated_image = self._make_image_array()
        self.simulated_image = simulated_image

        vmin = min(observed_image.min(), simulated_image.min())
        vmax = max(observed_image.max(), simulated_image.max()) if cutoff is None else cutoff
        im0 = axes[0].imshow(observed_image, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        im1 = axes[1].imshow(simulated_image, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

        """ Residuals """
        residuals = (observed_image - simulated_image) / np.std(self.imsim.image_data)
        self.residuals = residuals
        res_abs_max = 6.#np.max(np.abs(residuals))
        im2 = axes[2].imshow(residuals, origin='lower', cmap='bwr', vmin=-res_abs_max, vmax=res_abs_max)

        """ Source plane """
        source_array = self._make_source_array(pix_scale=pix_scale, size_arcsec=size_arcsec)
        self.source_array = source_array
        #axes[3].imshow(source_array, origin='lower', cmap='inferno')
        #source_array_crop = source_array.copy()
        if with_cbar :
            im3 = axes[3].imshow(np.log10(source_array), origin='lower', cmap=cmap)

        if with_cbar :
            for ax, im in zip(axes, [im0, im1, im2, im3]) :
                ax.axis('off')
                fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.005, location='bottom') #, shrink=0.98)
        else :
            for ax in axes :
                ax.axis('off')
        #fig.set_constrained_layout(True)
        #fig.tight_layout(pad=0.02)


        if False :
            kwargs_name = 'kwargs_opt' if 'kwargs_opt' in self.local else 'kwargs'
            kwargs = self.local[kwargs_name].copy()
            kwargs['kwargs_lens'] = self.local['kwargs']['kwargs_lens']
            kwargs['kwargs_lens_light'] = []
            if 'kwargs_source_ps' in kwargs :
                kwargs['kwargs_ps'] = kwargs['kwargs_source_ps']
                del kwargs['kwargs_source_ps']
            else :
                kwargs['kwargs_ps'] = []
                
            ps = []
            kwargs_name = 'kwargs_opt' if 'kwargs_opt' in self.local else 'kwargs'
            for i, src in enumerate(self.local[kwargs_name]['kwargs_source_ps']) :
                ps.append({})
                for param in src :
                    if param=='center_x' :
                        ps[i]['ra_source'] = src[param] - self.imsim.source_center_coordinates[0]
                    elif param=='center_y' :
                        ps[i]['dec_source'] = src[param] - self.imsim.source_center_coordinates[1]
                    else :
                        ps[i][param] = src[param]
            
            kwargs['kwargs_ps'] = ps

            self.modelPlot = ModelPlot([[self.imsim.ImageData_kwargs, self.imsim.PSF_kwargs, self.imsim.kwargs_numerics]], 
                            self.local['models'], kwargs, arrow_size=0.02, cmap_string="gist_heat",
                            linear_solver=False)
                            
            f, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=False, sharey=False)
            self.modelPlot.data_plot(ax=axes[0])
            self.modelPlot.model_plot(ax=axes[1])
            self.modelPlot.normalized_residual_plot(ax=axes[2], v_min=-6, v_max=6)
            #self.modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.001, numPix=1000)
            #self.modelPlot.convergence_plot(ax=axes[1, 1], v_max=1)
            #self.modelPlot.magnification_plot(ax=axes[1, 2])
            f.tight_layout()
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
            plt.show()
        
            f2, a2 = plt.subplots()
            a2 = self.modelPlot.source_plot(ax=a2, deltaPix_source=0.0005, numPix=1500, cmap='gray') #0.0005, 1500
            source_array = a2.get_children()[0].get_array()
            
            fig, ax = plt.subplots()
            #ax.imshow( source_array-np.min(source_array), origin="lower", cmap='gray' )
            ax.imshow( source_array, vmax=np.log(0.06), origin='lower', cmap='gray' )
            
            f3, a3 = plt.subplots()
            a3.imshow( np.exp(source_array), vmax=0.06, origin="lower", cmap='gray' )
        
        
        if self.chain_list is not None :
            for i in range(len(self.chain_list)):
                chain_plot.plot_chain_list(self.chain_list, 0)
        
        #return ax, source_array
        #return source_array
        
    
    def to_world(self) :
        model_world = copy.deepcopy(self.local)
        if 'models' not in model_world :
            model_world['models'] = {}
        if 'kwargs' not in model_world :
            model_world['kwargs'] = {}

        lens_models_local = model_world.get('models', {}).get('lens_model_list', [])
        lens_kwargs_local = model_world.get('kwargs', {}).get('kwargs_lens', [])
        lens_models_world = []
        lens_kwargs_world = []
        for i, (name, kw) in enumerate( zip(lens_models_local, lens_kwargs_local) ) :
            kw_world = copy.deepcopy(kw)
            self._local_to_world_coords(kw_world)
            lens_models_world.append(name)
            lens_kwargs_world.append(kw_world)
        model_world['models']['lens_model_list'] = lens_models_world
        model_world['kwargs']['kwargs_lens'] = lens_kwargs_world

        for block in self._iter_kwargs_blocks(model_world):
            if 'kwargs_source' in block :
                for src in block['kwargs_source'] :
                    self._local_to_world_coords(src)
            if 'kwargs_source_ps' in block :
                for ps in block['kwargs_source_ps'] :
                    self._local_to_world_coords(ps)
            if 'kwargs_lens' in block :
                for lens in block['kwargs_lens'] :
                    self._local_to_world_coords(lens)
        
        self._remove_interpol(model_world)
        self.world = model_world
        #return model_world


    def to_local(self) :        
        model_local = copy.deepcopy(self.world)
        lens_models_world = model_local.get('models', {}).get('lens_model_list', [])
        lens_kwargs_world = model_local.get('kwargs', {}).get('kwargs_lens', [])
        lens_models_local = []
        lens_kwargs_local = []
        for name, kw in zip(lens_models_world, lens_kwargs_world) :
            kw_local = copy.deepcopy(kw)
            self._world_to_local_coords(kw_local)
            lens_models_local.append(name)
            lens_kwargs_local.append(kw_local)
        model_local['models']['lens_model_list'] = lens_models_local
        if 'kwargs' not in model_local :
            model_local['kwargs'] = {}
        model_local['kwargs']['kwargs_lens'] = lens_kwargs_local

        for block in self._iter_kwargs_blocks(model_local) :
            if 'kwargs_source' in block :
                for src in block['kwargs_source'] :
                    self._world_to_local_coords(src)
            if 'kwargs_source_ps' in block :
                for ps in block['kwargs_source_ps'] :
                    self._world_to_local_coords(ps)
                    if 'center_x' in ps and 'center_y' in ps :
                        ps['ra_source'] = ps['center_x']
                        ps['dec_source'] = ps['center_y']
                        del ps['center_x']
                        del ps['center_y']
            if 'kwargs_lens' in block :
                for lens in block['kwargs_lens'] :
                    self._world_to_local_coords(lens)

        self._add_interpol(model_local)
        self.local = model_local
        #return model_local


    def add(self, model_2) :
        if type(model_2)==str :
            model_2 = self.load_lm(path=model_2)
        if 'lens_model_list' in model_2['models'] :
            if 'INTERPOL' in model_2['models']['lens_model_list'] :
                """ The model to add is in local format """        
                self._remove_interpol(model_2)
                for name in ['kwargs', 'kwargs_fixed', 'kwargs_lower', 'kwargs_upper', 'kwargs_sigma', 'kwargs_init', 'kwargs_opt', 'kwargs_MCMC'] : #, 'kwargs_PSO'
                    if name in self.local and name in model_2 :
                        self.local[name]['kwargs_lens'] += model_2[name]['kwargs_lens']
                        self.local[name]['kwargs_source'] += model_2[name]['kwargs_source']
                        self.local[name]['kwargs_source_ps'] += model_2[name]['kwargs_source_ps']
                    elif name in self.local and not name in model_2 :
                        self.local[name]['kwargs_lens'] += [{} for lens in model_2['models']['lens_model_list']]
                        self.local[name]['kwargs_source'] += [{} for src in model_2['models']['source_light_model_list']]
                        self.local[name]['kwargs_source_ps'] += [{} for ps in model_2['models']['point_source_model_list']]
                    elif name in model_2 and not name in self.local :
                        self.local[name] = {}
                        self.local[name]['kwargs_lens'] = [{} for lens in self.local['models']['lens_model_list']] + model_2[name]['kwargs_lens']
                        self.local[name]['kwargs_source'] = [{} for src in self.local['models']['source_light_model_list']] + model_2[name]['kwargs_source']
                        self.local[name]['kwargs_source_ps'] = [{} for ps in self.local['models']['point_source_model_list']] + model_2[name]['kwargs_source_ps']
                self.local['models']['lens_model_list'] += model_2['models']['lens_model_list']
                self.local['models']['source_light_model_list'] += model_2['models']['source_light_model_list']
                self.local['models']['point_source_model_list'] += model_2['models']['point_source_model_list']
                self.to_world()
                return None
        else :
            """ Here we add a placeholder for the lens models """
            model_2['models']['lens_model_list'] = []
            for name in ['kwargs', 'kwargs_fixed', 'kwargs_lower', 'kwargs_upper', 'kwargs_sigma', 'kwargs_init', 'kwargs_opt', 'kwargs_MCMC'] :
                if name in model_2 :
                    model_2[name]['kwargs_lens'] = []

        """ The model to add is in world format """        
        for name in ['kwargs', 'kwargs_fixed', 'kwargs_lower', 'kwargs_upper', 'kwargs_sigma', 'kwargs_init', 'kwargs_opt', 'kwargs_MCMC'] : #, 'kwargs_PSO'
            if name in self.world and name in model_2 :
                self.world[name]['kwargs_lens'] += model_2[name]['kwargs_lens']
                self.world[name]['kwargs_source'] += model_2[name]['kwargs_source']
                self.world[name]['kwargs_source_ps'] += model_2[name]['kwargs_source_ps']
            elif name in self.world and not name in model_2 :
                self.world[name]['kwargs_lens'] += [{} for lens in model_2['models']['lens_model_list']]
                self.world[name]['kwargs_source'] += [{} for src in model_2['models']['source_light_model_list']]
                self.world[name]['kwargs_source_ps'] += [{} for ps in model_2['models']['point_source_model_list']]
            elif name in model_2 and not name in self.world :
                self.world[name] = {}
                self.world[name]['kwargs_lens'] = [{} for lens in self.world['models']['lens_model_list']] + model_2[name]['kwargs_lens']
                self.world[name]['kwargs_source'] = [{} for src in self.world['models']['source_light_model_list']] + model_2[name]['kwargs_source']
                self.world[name]['kwargs_source_ps'] = [{} for ps in self.world['models']['point_source_model_list']] + model_2[name]['kwargs_source_ps']
        self.world['models']['lens_model_list'] += model_2['models']['lens_model_list']
        self.world['models']['source_light_model_list'] += model_2['models']['source_light_model_list']
        self.world['models']['point_source_model_list'] += model_2['models']['point_source_model_list']
        self.to_local()


    def save(self, path='lenstronomy_model.pkl') :
        #if os.path.exists(path) :
        #    with open(path, 'rb') as file :
        #        existing_model = pickle.load(file)
        #else :
        #    existing_model = {'models': {'source_light_model_list': []}, 
        #                      'results': {'kwargs_source': []}}
        
        #full_model = join_lenstronomy_model(existing_model, formatted_model)
        
        path = make_model_path(path)
        
        to_write = self.world.copy()
        
        if self.chain_list is not None :
            to_write['chain_list'] = self.chain_list
        
        with open(path, 'wb') as file :
            pickle.dump(to_write, file)
        print('Model saved at ' + path)


    def load_lm(self, path='lenstronomy_model.pkl') :
        path = make_model_path(path)
        if os.path.exists(path) :
            print('Source model found at: ' + path)
        
        with open(path, 'rb') as file :
            imported_model_world = pickle.load(file)
        
        # For backward compatibility
        if 'results' in imported_model_world :
            imported_model_world['kwargs'] = copy.deepcopy(imported_model_world['results'])
            del imported_model_world['results']
        
        #self.__init__(self.world, self.imsim)
        #self.to_local()
        return imported_model_world
    
    
    def plot_mcmc(self, figsize=None, **corner_kwargs):
        """Corner plot of the MCMC parameter samples stored in ``kwargs_MCMC``."""

        if 'kwargs_MCMC' not in self.local:
            raise KeyError("No MCMC kwargs found in the model. Expected key 'kwargs_MCMC'.")

        mcmc_block = self.local['kwargs_MCMC']

        columns, labels = [], []
        for block_key in ('kwargs_lens', 'kwargs_source', 'kwargs_source_ps'):
            if block_key not in mcmc_block:
                continue
            prefix = 'src' if block_key == 'kwargs_source' else 'ps' if block_key == 'kwargs_source_ps' else 'lens'
            for i, kw in enumerate(mcmc_block[block_key]):
                for param, val in kw.items():
                    arr = np.array(val)
                    if arr.size <= 1:
                        continue
                    columns.append(arr)
                    labels.append(f"{prefix}{i}_{param}")
        
        lengths = [len(col) for col in columns]
        if len(np.unique(lengths)) > 1 :
            length_min = np.min(lengths)
            for i in range(len(columns)) :
                columns[i] = columns[i][:length_min]
        
        data = np.column_stack(columns)

        kw = dict(labels=labels, show_titles=True, title_fmt=".3f")
        if figsize is not None:
            kw['fig'] = plt.figure(figsize=figsize)
        kw.update(corner_kwargs)

        fig = corner.corner(data, **kw)
        plt.show()
        #return fig
    
    def plot_logL(self) :
        if self.chain_list is not None :
            fig, ax = plt.subplots()
            step = 0
            for i, chain in enumerate(self.chain_list) :
                if chain[0] == 'PSO' :
                    start = step
                    end = step + len(chain[1][0])
                    x = np.arange(start, end)
                    ax.plot(x, chain[1][0])
                    step = end
                elif chain[0] in ['emcee', 'MCMC'] :
                    start = step
                    end = step + len(chain[3])
                    x = np.arange(start, end)
                    ax.plot(x, chain[3])
                    step = end
            plt.show()
            return fig, ax

    def remove_source(self, index) :
        del self.local['models']['source_light_model_list'][index]
        for name in ['kwargs', 'kwargs_fixed', 'kwargs_lower', 'kwargs_upper', 'kwargs_sigma', 'kwargs_init', 'kwargs_opt', 'kwargs_MCMC', 'kwargs_PSO'] :
            del self.local[name]['kwargs_source'][index]
            del self.local[name]['kwargs_source_ps'][index]
        self.to_world()

    def _apply_bounds_if_present(self, roi, block_key, index, plot_widget=None):
        """Apply ``kwargs_lower``/``kwargs_upper`` for ``block_key`` at ``index`` to ``roi`` sliders."""
        if 'kwargs_lower' not in self.local or 'kwargs_upper' not in self.local:
            return
        kl = self.local['kwargs_lower'].get(block_key)
        ku = self.local['kwargs_upper'].get(block_key)
        if kl is None or ku is None or index >= len(kl) or index >= len(ku):
            return
        _apply_kwargs_bounds_to_roi_sliders(
            roi, kl[index], ku[index], plot_widget=self.imsim.source_plane_widget if plot_widget is None else plot_widget
        )

    def _apply_fixed_if_present(self, roi, block_key, index):
        """Apply ``kwargs_fixed`` for ``block_key`` at ``index`` to ROI sliders."""
        if 'kwargs_fixed' not in self.local:
            return
        kf = self.local['kwargs_fixed'].get(block_key)
        if kf is None or index >= len(kf) or not isinstance(kf[index], dict):
            return

        fixed_kwargs = kf[index]
        sliders = getattr(roi, "sliders", None)
        if not isinstance(sliders, dict):
            return

        # Scalar params map directly to slider names except source_amp -> amp.
        for k_kwargs, k_slider in (
            ('amp', 'amp'),
            ('n_sersic', 'n_sersic'),
            ('sigma', 'sigma'),
            ('R_sersic', 'R_sersic'),
            ('sigma0', 'sigma0'),
            ('Ra', 'Ra'),
            ('Rs', 'Rs'),
            ('q', 'q'),
            ('phi', 'phi'),
            ('source_amp', 'amp'),
        ):
            if k_kwargs in fixed_kwargs and k_slider in sliders:
                sliders[k_slider].fixed = True
                sliders[k_slider].vmid = fixed_kwargs[k_kwargs]
                sliders[k_slider].update()

        # Position fixing is controlled by the dpos (link_box) slider.
        if any(k in fixed_kwargs for k in ('center_x', 'center_y', 'ra_source', 'dec_source')):
            _, sbox = _roi_link_box_slider(roi)
            if sbox is not None:
                sbox.fixed = True
                sbox.update()

    def _get_kwargs_name(self, sub_kwargs_name, i) :
        if getattr(self, '_kwargs_name_to_send', None) is not None :
            return self._kwargs_name_to_send
        elif not 'kwargs_opt' in self.local :
            return 'kwargs'
        elif len(self.local['kwargs_opt'][sub_kwargs_name][i])==0 :
            return 'kwargs'
        else :
            return 'kwargs_opt'

    def _make_sample_kwargs(self, step, sample_index) :
        new_kwargs_name = 'kwargs_' + step + '_' + str(sample_index)
        kwargs_samples_name = 'kwargs_' + step
        """ Create sample dict and fill it with values extracted from the PSO or MCMC (step) samples """
        self.local[new_kwargs_name] = {}
        for kwargs_name, kwargs in self.local[kwargs_samples_name].items() :
            self.local[new_kwargs_name][kwargs_name] = []
            for i, indiv_kwargs in enumerate(kwargs) :
                self.local[new_kwargs_name][kwargs_name].append({})
                for param_name, param_array in indiv_kwargs.items() :
                    self.local[new_kwargs_name][kwargs_name][i][param_name] = param_array[sample_index]
        """ Complete the sample dict with fixed parameters """
        for kwargs_name, kwargs in self.local['kwargs_fixed'].items() :
            if not kwargs_name in self.local[new_kwargs_name] :
                self.local[new_kwargs_name][kwargs_name] = []
            for i, indiv_kwargs in enumerate(kwargs) :
                for param_name, param_value in indiv_kwargs.items() :
                    if param_name in self.local[new_kwargs_name][kwargs_name][i] :
                        print('Debug: both fixed and sampled values present for same model and parameter!')
                    self.local[new_kwargs_name][kwargs_name][i][param_name] = self.local['kwargs_fixed'][kwargs_name][i][param_name]
        self.to_world()
        return new_kwargs_name


    def send_to_imsim(self, step=None, sample_index=None, kwargs_name=None) :
        PlotWidget = self.imsim.source_plane_widget
        PlotWidget_lens = self.imsim.image_plane_rgb
        if step is not None and sample_index is not None :
            self._kwargs_name_to_send = self._make_sample_kwargs(step, sample_index)
        elif kwargs_name is not None :
            self._kwargs_name_to_send = kwargs_name

        for i, model in enumerate(self.local['models'].get('lens_model_list', [])):
            if model != 'PJAFFE_ELLIPSE_POTENTIAL_Q_PHI':
                continue
            # This is in case we have added two models together where one was optimized but not the other
            kwargs_name = self._get_kwargs_name('kwargs_lens', i)
            kwargs = self.local[kwargs_name]['kwargs_lens'][i]
            pix_scale = self.imsim.image.pix_deg_scale * 3600.0 # because this is the RGB panel
            npix = self.imsim._crop_npix
            x_center = kwargs['center_x'] / pix_scale + npix / 2.0
            y_center = kwargs['center_y'] / pix_scale + npix / 2.0
            semi_major = kwargs['Rs'] / np.sqrt(kwargs['q']) / pix_scale
            semi_minor = kwargs['Rs'] * np.sqrt(kwargs['q']) / pix_scale
            x_corner, y_corner, a, b, angle = transform_ROI_params_inverse(
                x_center, y_center, semi_major, semi_minor, kwargs['phi']
            )

            PlotWidget_lens.last_roi = SelectableEllipseROI([x_corner, y_corner], [a, b], angle=angle, pen='orange', invertible=True)
            PlotWidget_lens.last_roi.type_general = 'ELLIPSE'
            PlotWidget_lens.last_roi.type = 'PJAFFE_ELLIPSE_POTENTIAL_Q_PHI'
            PlotWidget_lens.roi_list.append(PlotWidget_lens.last_roi)
            for handle in PlotWidget_lens.last_roi.handles:
                PlotWidget_lens.last_roi.removeHandle(handle['item'])
            PlotWidget_lens.addItem(PlotWidget_lens.last_roi)

            make_handles(PlotWidget_lens.last_roi)
            attach_sliders(PlotWidget_lens, ['sigma0', 'Ra', 'Rs', 'q', 'phi', 'dpos'], size_sliders_scaling=PlotWidget_lens.size_sliders_scaling)
            for p, v in kwargs.items():
                if p in PlotWidget_lens.last_roi.sliders:
                    PlotWidget_lens.last_roi.sliders[p].vmid = v

            self._apply_bounds_if_present(PlotWidget_lens.last_roi, 'kwargs_lens', i, plot_widget=PlotWidget_lens)
            self._apply_fixed_if_present(PlotWidget_lens.last_roi, 'kwargs_lens', i)
        
        for i, model in enumerate(self.local['models']['source_light_model_list']) :
            if model=='SERSIC' :
                # This is in case we have added two models together where one was optimized but not the other
                kwargs_name = self._get_kwargs_name('kwargs_source', i)
                kwargs = self.local[kwargs_name]['kwargs_source'][i]
                x = kwargs['center_x'] - self.imsim.source_center_coordinates[0]
                y = kwargs['center_y'] - self.imsim.source_center_coordinates[1]
                radius = kwargs['R_sersic']
                x_corner = x - radius
                y_corner = y - radius
                
                PlotWidget.last_roi = SelectableCircleROI([x_corner, y_corner], radius=radius, pen='purple', invertible=True)
                PlotWidget.last_roi.type_general = 'CIRCLE2'
                PlotWidget.last_roi.type = 'SERSIC'
                PlotWidget.roi_list.append(PlotWidget.last_roi)
                for handle in PlotWidget.last_roi.handles:
                    PlotWidget.last_roi.removeHandle(handle['item'])
                PlotWidget.addItem(PlotWidget.last_roi)
                
                PlotWidget.last_roi.addScaleHandle([0.5+2**-1.5, 0.5+2**-1.5], [0.5, 0.5])
                
                # Add sliders to control light source parameters
                params = ['amp', 'R_sersic', 'n_sersic', 'dpos']
                attach_sliders(PlotWidget, params, size_sliders_scaling=PlotWidget.size_sliders_scaling)
                
                for p, v in kwargs.items() :
                    if p in PlotWidget.last_roi.sliders :
                        PlotWidget.last_roi.sliders[p].vmid = v

                self._apply_bounds_if_present(PlotWidget.last_roi, 'kwargs_source', i, plot_widget=PlotWidget)
                self._apply_fixed_if_present(PlotWidget.last_roi, 'kwargs_source', i)

            if model in ('SERSIC_ELLIPSE_Q_PHI', 'SERSIC_ELLIPSE') :
                # This is in case we have added two models together where one was optimized but not the other
                kwargs_name = self._get_kwargs_name('kwargs_source', i)
                kwargs = self.local[kwargs_name]['kwargs_source'][i]
                x_center = kwargs['center_x'] - self.imsim.source_center_coordinates[0]
                y_center = kwargs['center_y'] - self.imsim.source_center_coordinates[1]
                R_sersic = kwargs['R_sersic']
                #e1, e2 = kwargs['e1'], kwargs['e2']
                #e = (e1**2 + e2**2)**0.5
                #q = (1-e) / (1+e)
                #a = 2*R_sersic / (1+q)
                #b = a*q
                #theta = 0.5*np.arctan(e2/e1)
                #x_corner, y_corner, a, b, angle = transform_ROI_params_inverse(x_center, y_center, a, b, theta)
                if model == 'SERSIC_ELLIPSE_Q_PHI' :
                    q = kwargs['q']
                    phi = kwargs['phi']
                else :
                    e1, e2 = kwargs['e1'], kwargs['e2']
                    phi, q = param_util.ellipticity2phi_q(float(e1), float(e2))
                semi_major = R_sersic / np.sqrt(q)
                semi_minor = R_sersic * np.sqrt(q)
                x_corner, y_corner, a, b, angle = transform_ROI_params_inverse(
                    x_center, y_center, semi_major, semi_minor, phi
                )
                
                PlotWidget.last_roi = SelectableEllipseROI([x_corner, y_corner], [a, b], angle=angle, pen='orange', invertible=True)
                PlotWidget.last_roi.type_general = 'ELLIPSE'
                PlotWidget.last_roi.type = 'SERSIC_ELLIPSE_Q_PHI'
                PlotWidget.roi_list.append(PlotWidget.last_roi)
                for handle in PlotWidget.last_roi.handles:
                    PlotWidget.last_roi.removeHandle(handle['item'])
                PlotWidget.addItem(PlotWidget.last_roi)
                
                make_handles(PlotWidget.last_roi)
                
                # Add sliders to control light source parameters
                params = ['amp', 'R_sersic', 'n_sersic', 'q', 'phi', 'dpos']
                attach_sliders(PlotWidget, params, size_sliders_scaling=PlotWidget.size_sliders_scaling)
                
                for p, v in kwargs.items() :
                    if p in PlotWidget.last_roi.sliders :
                        PlotWidget.last_roi.sliders[p].vmid = v

                self._apply_bounds_if_present(PlotWidget.last_roi, 'kwargs_source', i, plot_widget=PlotWidget)
                self._apply_fixed_if_present(PlotWidget.last_roi, 'kwargs_source', i)

            if model=='GAUSSIAN' :
                # This is in case we have added two models together where one was optimized but not the other
                kwargs_name = self._get_kwargs_name('kwargs_source', i)
                kwargs = self.local[kwargs_name]['kwargs_source'][i]
                x = kwargs['center_x'] - self.imsim.source_center_coordinates[0]
                y = kwargs['center_y'] - self.imsim.source_center_coordinates[1]
                radius = kwargs['sigma'] / 2.0
                x_corner = x - radius
                y_corner = y - radius
                
                PlotWidget.last_roi = SelectableCircleROI([x_corner, y_corner], radius=radius, pen='red', invertible=True)
                PlotWidget.last_roi.type_general = 'CIRCLE1'
                PlotWidget.last_roi.type = 'GAUSSIAN'
                PlotWidget.roi_list.append(PlotWidget.last_roi)
                for handle in PlotWidget.last_roi.handles:
                    PlotWidget.last_roi.removeHandle(handle['item'])
                PlotWidget.addItem(PlotWidget.last_roi)
                
                PlotWidget.last_roi.addScaleHandle([0.5+2**-1.5, 0.5+2**-1.5], [0.5, 0.5])
                
                # Add sliders to control light source parameters
                params = ['amp', 'sigma', 'dpos']
                attach_sliders(PlotWidget, params, size_sliders_scaling=PlotWidget.size_sliders_scaling)
                
                for p, v in kwargs.items() :
                    if p in PlotWidget.last_roi.sliders :
                        PlotWidget.last_roi.sliders[p].vmid = v

                self._apply_bounds_if_present(PlotWidget.last_roi, 'kwargs_source', i, plot_widget=PlotWidget)
                self._apply_fixed_if_present(PlotWidget.last_roi, 'kwargs_source', i)

        for i, model in enumerate(self.local['models']['point_source_model_list']) :
            if model == 'SOURCE_POSITION' :
                # This is in case we have added two models together where one was optimized but not the other
                kwargs_name = self._get_kwargs_name('kwargs_source_ps', i)
                kwargs = self.local[kwargs_name]['kwargs_source_ps'][i]
                x = kwargs['ra_source'] - self.imsim.source_center_coordinates[0] #kwargs['ra_source']
                y = kwargs['dec_source'] - self.imsim.source_center_coordinates[1] #kwargs['dec_source']
                radius = 1e-3
                x_corner = x - radius
                y_corner = y - radius

                PlotWidget.last_roi = SelectableCircleROI([x_corner, y_corner], radius=radius, pen='cyan', invertible=True)
                PlotWidget.last_roi.type_general = 'POINT'
                PlotWidget.last_roi.type = 'SOURCE_POSITION'
                PlotWidget.roi_list.append(PlotWidget.last_roi)
                for handle in PlotWidget.last_roi.handles:
                    PlotWidget.last_roi.removeHandle(handle['item'])
                PlotWidget.addItem(PlotWidget.last_roi)

                params = ['amp', 'dpos']
                attach_sliders(PlotWidget, params, size_sliders_scaling=PlotWidget.size_sliders_scaling)

                amp_val = kwargs.get('source_amp', kwargs.get('amp'))
                if amp_val is not None and 'amp' in PlotWidget.last_roi.sliders :
                    PlotWidget.last_roi.sliders['amp'].vmid = amp_val

                self._apply_bounds_if_present(PlotWidget.last_roi, 'kwargs_source_ps', i, plot_widget=PlotWidget)
                self._apply_fixed_if_present(PlotWidget.last_roi, 'kwargs_source_ps', i)

        self._kwargs_name_to_send = None
    
    
    def _make_source_array(self, pix_scale=0.001, size_arcsec=0.4) :
        LensModel_empty = LensModel(lens_model_list=[])
        LightModel_source = LightModel(light_model_list=self.local['models']['source_light_model_list'])
        PSModel_source = PointSource(point_source_type_list=self.local['models']['point_source_model_list'], fixed_magnification_list=[True for src in self.local['models']['point_source_model_list']])

        offset_source = []
        offset_ps = []
        
        kwargs_name = 'kwargs_opt' if 'kwargs_opt' in self.local else 'kwargs'
        print('Using ' + kwargs_name)

        x_list = [src['ra_source'] for src in self.local[kwargs_name]['kwargs_source_ps']] + [src['center_x'] for src in self.local[kwargs_name]['kwargs_source']]
        y_list = [src['dec_source'] for src in self.local[kwargs_name]['kwargs_source_ps']] + [src['center_y'] for src in self.local[kwargs_name]['kwargs_source']]
        #x_mean, y_mean = np.mean(x_list), np.mean(y_list)
        x_center = (np.max(x_list) + np.min(x_list)) / 2.0
        y_center = (np.max(y_list) + np.min(y_list)) / 2.0
        
        for i, src in enumerate(self.local[kwargs_name]['kwargs_source']) :
            offset_source.append({})
            for param in src :
                if param=='center_x' :
                    offset_source[i][param] = src[param] - x_center
                elif param=='center_y' :
                    offset_source[i][param] = src[param] - y_center
                else :
                    offset_source[i][param] = src[param]
        for i, src in enumerate(self.local[kwargs_name]['kwargs_source_ps']) :
            offset_ps.append({})
            for param in src :
                if param=='ra_source' :
                    offset_ps[i]['ra_source'] = src[param] - x_center
                elif param=='dec_source' :
                    offset_ps[i]['dec_source'] = src[param] - y_center
                else :
                    offset_ps[i][param] = src[param]
        
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * pix_scale
        PixelGrid_kwargs = {'nx': round(size_arcsec / pix_scale),
                            'ny': round(size_arcsec / pix_scale),
                            'ra_at_xy_0': - size_arcsec/2,
                            'dec_at_xy_0': - size_arcsec/2,
                            'transform_pix2angle': transform_pix2angle} 
        pixel_grid = PixelGrid(**PixelGrid_kwargs)
                    
        image_model_class = ImageModel( data_class=pixel_grid, psf_class=self.imsim.PSF,
                                        lens_model_class=LensModel_empty,
                                        source_model_class=LightModel_source,
                                        point_source_class=PSModel_source,
                                        #lens_light_model_class=,
                                        kwargs_numerics=self.imsim.kwargs_numerics )

        source_array = image_model_class.image(kwargs_source=offset_source,
                                                kwargs_ps=offset_ps,
                                                #kwargs_lens_light=kwargs_light_lens,
                                                kwargs_lens=[{}], unconvolved=False)
        return source_array
        
    def _make_image_array(self) :
        LightModel_source = LightModel(light_model_list=self.local['models']['source_light_model_list'])
        PSModel_source = PointSource(point_source_type_list=self.local['models']['point_source_model_list'], fixed_magnification_list=[True for src in self.local['models']['point_source_model_list']])

        ps = []
        kwargs_name = 'kwargs_opt' if 'kwargs_opt' in self.local else 'kwargs'
        print('Using ' + kwargs_name)
        
        for i, src in enumerate(self.local[kwargs_name]['kwargs_source_ps']) :
            ps.append({})
            for param in src :
                if param=='ra_source' :
                    ps[i]['ra_source'] = src[param]
                elif param=='dec_source' :
                    ps[i]['dec_source'] = src[param]
                else :
                    ps[i][param] = src[param]

        image_model_class = ImageModel( data_class=self.imsim.PixelGrid, psf_class=self.imsim.PSF,
                                        lens_model_class=self.imsim.LensModel,
                                        source_model_class=LightModel_source,
                                        point_source_class=PSModel_source,
                                        #lens_light_model_class=,
                                        kwargs_numerics=self.imsim.kwargs_numerics )

        image_array = image_model_class.image(kwargs_source=self.local[kwargs_name]['kwargs_source'],
                                                kwargs_ps=ps,
                                                #kwargs_lens_light=kwargs_light_lens,
                                                kwargs_lens=self.local[kwargs_name]['kwargs_lens'], unconvolved=False)
        return image_array
    
    def optimize(self, fitting_kwargs_list=None) :
        fitting_kwargs_list = getattr(self.imsim, 'fitting_kwargs_list', None) if fitting_kwargs_list is None else fitting_kwargs_list
        lm_dict_opt = copy.deepcopy(self.local)
        if 'kwargs_opt' in lm_dict_opt :
            del lm_dict_opt['kwargs_opt']
        if 'kwargs_MCMC' in lm_dict_opt :
            del lm_dict_opt['kwargs_MCMC']
        self.imsim.optimize(fitting_kwargs_list=fitting_kwargs_list, lm_dict_opt=lm_dict_opt)


    def make_lens_table(self, columns=None) :
        self.lens_table, self.lens_columns = make_lm_table(self, kwargs_name_list=['kwargs_lens'], columns=columns)
        return self.lens_table, self.lens_columns

    def make_source_table(self, columns=None) :
        self.source_table, self.source_columns = make_lm_table(self, kwargs_name_list=['kwargs_source', 'kwargs_source_ps'], columns=columns)
        return self.source_table, self.source_columns
    
    def make_table(self, kwargs_name_list=['kwargs_lens', 'kwargs_source', 'kwargs_source_ps'], columns=None) :
        self.table, self.columns = make_lm_table(self, kwargs_name_list=kwargs_name_list, columns=columns)
        return self.table, self.columns

    def make_lens_latex(self, columns=None, uncertainty='best', deluxetable=True) :
        if getattr(self, 'lens_table', None) is None :
            return make_lm_LaTeX(self, kwargs_name_list=['kwargs_lens'], columns=columns, uncertainty=uncertainty, deluxetable=deluxetable)
        else :
            return table_to_LaTeX_str(self.lens_table, columns=self.lens_columns, uncertainty=uncertainty, deluxetable=deluxetable)

    def make_source_latex(self, columns=None, uncertainty='best', deluxetable=True) :
        if getattr(self, 'source_table', None) is None :
            return make_lm_LaTeX(self, kwargs_name_list=['kwargs_source', 'kwargs_source_ps'], columns=columns, uncertainty=uncertainty, deluxetable=deluxetable)
        else :
            return table_to_LaTeX_str(self.source_table, columns=self.source_columns, uncertainty=uncertainty, deluxetable=deluxetable)

    def make_latex(self, kwargs_name_list=['kwargs_lens', 'kwargs_source', 'kwargs_source_ps'], columns=None, uncertainty='best', deluxetable=True) :
        if getattr(self, 'table', None) is None :
            return make_lm_LaTeX(self, kwargs_name_list=kwargs_name_list, columns=columns, uncertainty=uncertainty, deluxetable=deluxetable)
        else :
            return table_to_LaTeX_str(self.table, columns=self.columns, uncertainty=uncertainty, deluxetable=deluxetable)


def make_model_path(path) :
    dirname = os.path.dirname(path)
    if not dirname or not os.path.isdir(dirname) :
        path = os.path.join(os.curdir, path)
    base, ext = os.path.splitext(path)
    if not ext :
        path = f"{base}.pkl"
    return path
