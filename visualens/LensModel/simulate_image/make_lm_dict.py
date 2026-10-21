import numpy as np
from ...utils.utils_Qt.utils_general import transform_ROI_params
from ...utils.utils_Qt.Special_QWidgets import _roi_link_box_slider
from lenstronomy.Sampling.parameters import Param
from lenstronomy.LensModel.lens_model import LensModel




def make_lm_dict(imsim) :

    """ Get the lens model parameters from the rgb image panel's ROIs """
    # First, reset the lens model list and kwargs to just the base interpol model
    imsim.LensModel_list = [imsim.LensModel_list[0]]
    imsim.LensModel_kwargs = [imsim.LensModel_kwargs[0]]
    # Then, add the PJAFFE_ELLIPSE_POTENTIAL_Q_PHI model for each ellipse ROI on the rgb image plane
    for roi in imsim.image_plane_rgb.roi_list :
        if roi.type=='PJAFFE_ELLIPSE_POTENTIAL_Q_PHI' :
            imsim.LensModel_list.append('PJAFFE_ELLIPSE_POTENTIAL_Q_PHI')
            x_center, y_center, semi_major, semi_minor, angle = transform_ROI_params(roi)
            # This time, the intrinsic units of the rgb panel are pixels, so we need to convert the ROI's parameters to arcsec
            npix = imsim._crop_npix
            pix_scale = imsim.image.pix_deg_scale * 3600.0
            center_x = (x_center - npix / 2.0) * pix_scale
            center_y = (y_center - npix / 2.0) * pix_scale
            q = semi_minor / semi_major
            to_add = {
                'sigma0': roi.sliders['sigma0'].vmid,
                'Ra': roi.sliders['Ra'].vmid,
                'Rs': roi.sliders['Rs'].vmid,
                'q': q,
                'phi': angle,
                'center_x': center_x,
                'center_y': center_y,
            }
            imsim.LensModel_kwargs.append(to_add)

    imsim.LensModel = LensModel(lens_model_list=imsim.LensModel_list, z_lens=imsim.lensmodel.z_lens, z_source=imsim.z_source)
    # Might be useful at some point to shift the source plane widget by the new source center coordinates for the curent lens model
    imsim.source_center_coordinates_new = imsim.LensModel.ray_shooting(0.0, 0.0, imsim.LensModel_kwargs)

    """ Get the light model parameters from the source plane widget's ROIs """
    LightModel_source_list = []
    LightModel_source_kwargs = []
    PSModel_source_list = []
    PSModel_source_kwargs = []

    for roi in imsim.source_plane_widget.roi_list :
        if roi.type=='SERSIC_ELLIPSE_Q_PHI' :
            LightModel_source_list.append('SERSIC_ELLIPSE_Q_PHI')
            amp = roi.sliders['amp'].vmid
            n_sersic = roi.sliders['n_sersic'].vmid
            x_center, y_center, semi_major, semi_minor, angle = transform_ROI_params(roi)
            q = semi_minor / semi_major
            #e = (1-q) / (1+q)
            #e1 = e * np.cos(2*angle)
            #e2 = e * np.sin(2*angle)
            R_sersic = np.sqrt(semi_major * semi_minor)
            src_xr = x_center + imsim.source_center_coordinates[0]
            src_yr = y_center + imsim.source_center_coordinates[1]
            to_add = {'amp': amp, 'R_sersic': R_sersic, 'n_sersic': n_sersic, 'q': q, 'phi': angle, 'center_x': src_xr, 'center_y': src_yr}
            LightModel_source_kwargs.append(to_add)
        elif roi.type=='GAUSSIAN' :
            LightModel_source_list.append('GAUSSIAN')
            amp = roi.sliders['amp'].vmid
            x_center, y_center, semi_major, semi_minor, angle = transform_ROI_params(roi)
            src_xr = x_center + imsim.source_center_coordinates[0]
            src_yr = y_center + imsim.source_center_coordinates[1]
            sigma = abs(roi.size()[0])
            to_add = {'amp': amp, 'sigma': sigma, 'center_x': src_xr, 'center_y': src_yr}
            LightModel_source_kwargs.append(to_add)
        elif roi.type=='SERSIC' :
            LightModel_source_list.append('SERSIC')
            amp = roi.sliders['amp'].vmid
            n_sersic = roi.sliders['n_sersic'].vmid
            x_center, y_center, semi_major, semi_minor, angle = transform_ROI_params(roi)
            src_xr = x_center + imsim.source_center_coordinates[0]
            src_yr = y_center + imsim.source_center_coordinates[1]
            R_sersic = abs(roi.size()[0]) / 2.0
            to_add = {'amp': amp, 'R_sersic': R_sersic, 'n_sersic': n_sersic, 'center_x': src_xr, 'center_y': src_yr}
            LightModel_source_kwargs.append(to_add)
        elif roi.type=='SOURCE_POSITION' :
            PSModel_source_list.append('SOURCE_POSITION')
            amp = roi.sliders['amp'].vmid
            x_center, y_center, semi_major, semi_minor, angle = transform_ROI_params(roi)
            src_xr = x_center + imsim.source_center_coordinates[0]
            src_yr = y_center + imsim.source_center_coordinates[1]
            to_add = {'source_amp': amp, 'ra_source': src_xr, 'dec_source': src_yr} # point_amp?
            PSModel_source_kwargs.append(to_add)
    
    
    ######### Create the current lenstronomy model class #########
    lm_dict = {}
    lm_dict['models'] = {'lens_model_list': imsim.LensModel_list,
                         #'lens_light_model_list': imsim.LensModel_light_list,
                         'source_light_model_list': LightModel_source_list,
                         'point_source_model_list': PSModel_source_list}
    lm_dict['kwargs'] = {'kwargs_lens': imsim.LensModel_kwargs, 
                         'kwargs_source': LightModel_source_kwargs,
                         'kwargs_source_ps': PSModel_source_kwargs}
    return lm_dict


def make_lm_dict_opt(imsim, N_sigma=6.) :
    lm_dict_opt = make_lm_dict(imsim)

    def _position_is_fixed(roi):
        _, sbox = _roi_link_box_slider(roi)
        return bool(sbox is not None and getattr(sbox, "fixed", False))

    def _get_param_slider(roi, param_name):
        slider_name = 'amp' if param_name == 'source_amp' else param_name
        return getattr(roi, "sliders", {}).get(slider_name, None)

    def _param_is_fixed(roi, param_name):
        if param_name in ('center_x', 'center_y', 'ra_source', 'dec_source'):
            return _position_is_fixed(roi)
        s = _get_param_slider(roi, param_name)
        return bool(s is not None and getattr(s, "fixed", False))
    
    """ Lens models """
    roi_indices = np.where( [roi.type != 'LENS_LIGHT_MODEL' for roi in imsim.image_plane_rgb.roi_list] )[0]
    
    kwargs_lens_fixed = []
    kwargs_lens_lower = []
    kwargs_lens_upper = []
    kwargs_lens_sigma = []
    kwargs_lens_init = []
    for i, lens in enumerate(lm_dict_opt['models']['lens_model_list']) :
        if lens == 'INTERPOL' :
            kwargs_lens_fixed.append( imsim.LensModel_kwargs[i] )
            kwargs_lens_lower.append({})
            kwargs_lens_upper.append({})
            kwargs_lens_sigma.append({})
            kwargs_lens_init.append({})
        else :
            kwargs = lm_dict_opt['kwargs']['kwargs_lens'][i]
            kwargs_fixed = {}
            kwargs_lower = {}
            kwargs_upper = {}
            kwargs_sigma = {}
            kwargs_init = {}
            
            opt_params = lm_dict_opt['kwargs']['kwargs_lens'][i].keys()
            roi = imsim.image_plane_rgb.roi_list[roi_indices[i-1]]
            for param in opt_params :
                if _param_is_fixed(roi, param):
                    kwargs_fixed[param] = kwargs[param]
                    continue
                if param in ['center_x', 'center_y'] :
                    _, sbox = _roi_link_box_slider(roi)
                    dpos = sbox.vmid
                    v = kwargs[param]
                    kwargs_lower[param] = v - dpos # * imsim.image.pix_deg_scale*3600
                    kwargs_upper[param] = v + dpos # * imsim.image.pix_deg_scale*3600
                    kwargs_sigma[param] = 2*dpos/N_sigma
                    kwargs_init[param] = v
                #elif param in ['e1', 'e2'] :
                #    kwargs_lower[param] = -1.
                #    kwargs_upper[param] = 1.
                #    kwargs_sigma[param] = 2/N_sigma
                else : # param in ['q', 'phi'] :
                    vmax = roi.sliders[param].vmax
                    vmin = roi.sliders[param].vmin
                    kwargs_lower[param] = vmin
                    kwargs_upper[param] = vmax
                    kwargs_sigma[param] = (vmax - vmin) / N_sigma
                    kwargs_init[param] = kwargs[param]
            
            kwargs_lens_fixed.append(kwargs_fixed)
            kwargs_lens_lower.append(kwargs_lower)
            kwargs_lens_upper.append(kwargs_upper)
            kwargs_lens_sigma.append(kwargs_sigma)
            kwargs_lens_init.append(kwargs_init)


    """ Source Light models """
    roi_indices = np.where( [roi.type != 'SOURCE_POSITION' for roi in imsim.source_plane_widget.roi_list] )[0]
    
    kwargs_source_fixed = []
    kwargs_source_lower = []
    kwargs_source_upper = []
    kwargs_source_sigma = []
    kwargs_source_init = []
    for i, src in enumerate(lm_dict_opt['models']['source_light_model_list']) :
        kwargs = lm_dict_opt['kwargs']['kwargs_source'][i]
        kwargs_fixed = {}
        kwargs_lower = {}
        kwargs_upper = {}
        kwargs_sigma = {}
        kwargs_init = {}
        
        opt_params = lm_dict_opt['kwargs']['kwargs_source'][i].keys()
        roi = imsim.source_plane_widget.roi_list[roi_indices[i]]
        for param in opt_params :
            if _param_is_fixed(roi, param):
                kwargs_fixed[param] = kwargs[param]
                continue
            if param in ['center_x', 'center_y'] :
                _, sbox = _roi_link_box_slider(roi)
                dpos = sbox.vmid
                v = kwargs[param]
                kwargs_lower[param] = v - dpos
                kwargs_upper[param] = v + dpos
                kwargs_sigma[param] = 2*dpos/N_sigma
                kwargs_init[param] = v
            #elif param in ['e1', 'e2'] :
            #    kwargs_lower[param] = -1.
            #    kwargs_upper[param] = 1.
            #    kwargs_sigma[param] = 2/N_sigma
            else : # param in ['q', 'phi'] :
                vmax = roi.sliders[param].vmax
                vmin = roi.sliders[param].vmin
                kwargs_lower[param] = vmin
                kwargs_upper[param] = vmax
                kwargs_sigma[param] = (vmax - vmin) / N_sigma
                kwargs_init[param] = kwargs[param]
        
        kwargs_source_fixed.append(kwargs_fixed)
        kwargs_source_lower.append(kwargs_lower)
        kwargs_source_upper.append(kwargs_upper)
        kwargs_source_sigma.append(kwargs_sigma)
        kwargs_source_init.append(kwargs_init)


    """ Point source models """
    roi_indices = np.where( [roi.type == 'SOURCE_POSITION' for roi in imsim.source_plane_widget.roi_list] )[0]
    
    kwargs_ps_fixed = []
    kwargs_ps_lower = []
    kwargs_ps_upper = []
    kwargs_ps_sigma = []
    kwargs_ps_init = []
    for i, src in enumerate(lm_dict_opt['models']['point_source_model_list']) :
        kwargs = lm_dict_opt['kwargs']['kwargs_source_ps'][i]
        kwargs_fixed = {}
        kwargs_lower = {}
        kwargs_upper = {}
        kwargs_sigma = {}
        kwargs_init = {}
        
        opt_params = lm_dict_opt['kwargs']['kwargs_source_ps'][i].keys()
        roi = imsim.source_plane_widget.roi_list[roi_indices[i]]
        for param in opt_params :
            if _param_is_fixed(roi, param):
                kwargs_fixed[param] = kwargs[param]
                continue
            if param in ['ra_source', 'dec_source'] :
                _, sbox = _roi_link_box_slider(roi)
                dpos = sbox.vmid
                v = kwargs[param]
                kwargs_lower[param] = v - dpos
                kwargs_upper[param] = v + dpos
                kwargs_sigma[param] = 2*dpos/N_sigma
                kwargs_init[param] = v
            else : # Here only source_amp
                param_slider = 'amp' if param=='source_amp' else param
                vmax = roi.sliders[param_slider].vmax
                vmin = roi.sliders[param_slider].vmin
                vmid = roi.sliders[param_slider].vmid
                kwargs_lower[param] = vmin
                kwargs_upper[param] = vmax
                kwargs_sigma[param] = (vmax + vmin)/N_sigma
                kwargs_init[param] = vmid
        
        
        kwargs_ps_fixed.append(kwargs_fixed)
        kwargs_ps_lower.append(kwargs_lower)
        kwargs_ps_upper.append(kwargs_upper)
        kwargs_ps_sigma.append(kwargs_sigma)
        kwargs_ps_init.append(kwargs_init)
        
    
    if False : # This produces an error if there is a SERSIC_ELLIPSE_Q_PHI light model
        param = Param(lm_dict_opt['models'],
                      kwargs_fixed_lens=imsim.LensModel_kwargs,
                      kwargs_fixed_source=kwargs_source_fixed,#self.imsim.fixed_source_kwargs,
                      #kwargs_fixed_lens_light=kwargs_fixed_lens_light,
                      kwargs_fixed_ps=kwargs_ps_fixed, 
                      kwargs_lower_lens=[{}],
                      kwargs_lower_source=kwargs_source_lower,
                      #kwargs_lower_lens_light=kwargs_lower_lens_light,
                      kwargs_lower_ps=kwargs_ps_lower,
                      kwargs_upper_lens=[{}],
                      kwargs_upper_source=kwargs_source_upper,
                      #kwargs_upper_lens_light=kwargs_upper_lens_light,
                      kwargs_upper_ps=kwargs_ps_upper,
                      #kwargs_lens_init=kwargs_lens,
                      #joint_lens_with_light: [[0, 0, ['center_x', 'center_y']]],
                      linear_solver=imsim.use_linear_solver)
        param.print_setting()
    
    lm_dict_opt['kwargs_fixed'] = {'kwargs_lens': kwargs_lens_fixed, 'kwargs_source': kwargs_source_fixed, 'kwargs_source_ps': kwargs_ps_fixed}
    lm_dict_opt['kwargs_lower'] = {'kwargs_lens': kwargs_lens_lower, 'kwargs_source': kwargs_source_lower, 'kwargs_source_ps': kwargs_ps_lower}
    lm_dict_opt['kwargs_upper'] = {'kwargs_lens': kwargs_lens_upper, 'kwargs_source': kwargs_source_upper, 'kwargs_source_ps': kwargs_ps_upper}
    lm_dict_opt['kwargs_sigma'] = {'kwargs_lens': kwargs_lens_sigma, 'kwargs_source': kwargs_source_sigma, 'kwargs_source_ps': kwargs_ps_sigma}
    lm_dict_opt['kwargs_init'] = {'kwargs_lens': kwargs_lens_init, 'kwargs_source': kwargs_source_init, 'kwargs_source_ps': kwargs_ps_init}
    
    return lm_dict_opt


