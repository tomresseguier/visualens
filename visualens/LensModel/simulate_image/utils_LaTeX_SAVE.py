import numpy as np
import math
from ...utils.utils_astro.utils_general import world_to_relative, arcsec_to_kpc
from ...utils.utils_astro.get_default_cosmology import get_default_cosmology



def make_lm_LaTeX(lm_instance, ref=None, use_parsecs=True, uncertainty='hpd', flux_per_amp=None, flux_unit=None) :
    """Return (and print) a LaTeX table of all light-model component parameters.

    When ``kwargs_MCMC`` is present, each cell shows the central value with
    asymmetric uncertainties ``$v^{+hi}_{-lo}$``.  The method used depends
    on ``uncertainty``:

    * ``'hpd'`` *(default)* - centre is the best-fit (``kwargs_opt``);
      bounds are the 68.3 % Highest Posterior Density interval of the MCMC
      chain.
    * ``'median'`` - centre is the MCMC median (50th percentile); bounds
      are the 16th and 84th percentiles.  When ``ref`` is an integer index
      the reference position is also taken from the MCMC median of that
      component (not the best-fit).

    Parameters not MCMC-sampled fall back to the plain point estimate.

    Parameters
    ----------
    ref : None, tuple, or int
        * ``None``  - coordinates are left as-is (arcsec relative to the
          lenstronomy center; no unit conversion is applied).
        * ``(ra, dec)`` - world coordinates (degrees) used as the origin;
          each component's position is expressed relative to this point.
        * ``int``   - index into the combined list of components (source
          light models first, then point-source models); the position of
          that component becomes the origin.
    use_parsecs : bool
        When ``ref`` is not ``None``, convert the relative arcsec offsets
        to parsecs using ``imsim.z_source`` and the default cosmology.
        Has no effect when ``ref=None``.
    uncertainty : {'hpd', 'median'}
        Method used to derive uncertainties from the MCMC chain.

    Returns
    -------
    str  - the complete LaTeX ``table`` environment as a string.
    """
    if uncertainty not in ('hpd', 'median'):
        raise ValueError(f"uncertainty must be 'hpd' or 'median', got {uncertainty!r}")
    
    cosmo = get_default_cosmology()
    z = lm_instance.imsim.z_source
    
    if flux_unit is None :
        if 'BUNIT' in lm_instance.imsim.individual_filter.header :
            flux_unit = lm_instance.imsim.individual_filter.header['BUNIT']
        else :
            flux_unit = 'arbitrary units'
    flux_per_amp = 1. if flux_per_amp is None else flux_per_amp
    
    kwargs_key = 'kwargs_opt' if 'kwargs_opt' in lm_instance.local else 'kwargs'
    kw_block = lm_instance.local[kwargs_key]
    models = lm_instance.local['models']

    # --- collect all components as (kind, local_index, model_name, kwargs) ---
    components = []
    for i, model in enumerate(models.get('source_light_model_list', [])):
        components.append(('src', i, model, dict(kw_block['kwargs_source'][i])))
    for i, model in enumerate(models.get('point_source_model_list', [])):
        components.append(('ps', i, model, dict(kw_block['kwargs_source_ps'][i])))

    # --- MCMC samples (optional) ---
    mcmc_block = lm_instance.local.get('kwargs_MCMC')
    mcmc_src   = (mcmc_block or {}).get('kwargs_source',    [])
    mcmc_ps    = (mcmc_block or {}).get('kwargs_source_ps', [])

    def _mcmc_kw(kind, idx):
        """Return the MCMC kwargs dict for this component, or None."""
        if mcmc_block is None:
            return None
        pool = mcmc_src if kind == 'src' else mcmc_ps
        return pool[idx] if idx < len(pool) else None

    def _samples(mcmc_kw, col):
        """Return a 1-D float array of MCMC samples for ``col``, or None."""
        if mcmc_kw is None:
            return None
        keys = ('amp', 'source_amp') if col == 'amp' else (col,)
        for k in keys:
            if k in mcmc_kw:
                arr = np.asarray(mcmc_kw[k], dtype=float).ravel()
                if arr.size > 1:
                    return arr
        return None

    # --- resolve reference position in local arcsec ---
    do_offset = ref is not None
    ref_x, ref_y = 0.0, 0.0

    if isinstance(ref, (tuple, list)):
        ref_x, ref_y = world_to_relative(ref[0], ref[1], lm_instance.imsim.center_world)
    elif isinstance(ref, int):
        if ref < 0 or ref >= len(components):
            raise IndexError(
                f"ref={ref} is out of range for {len(components)} component(s)."
            )
        ref_kind, ref_idx, _, kw_ref = components[ref]
        ref_x = float(kw_ref.get('center_x', kw_ref.get('ra_source', 0.0)))
        ref_y = float(kw_ref.get('center_y', kw_ref.get('dec_source', 0.0)))
        # For the median method, override with the MCMC median position
        if uncertainty == 'median' and mcmc_block is not None:
            ref_mk = _mcmc_kw(ref_kind, ref_idx)
            for raw_key, alt_key, attr in (
                ('center_x', 'ra_source',  'ref_x'),
                ('center_y', 'dec_source', 'ref_y'),
            ):
                samp = _samples(ref_mk, raw_key)
                if samp is None:
                    samp = _samples(ref_mk, alt_key)
                if samp is not None:
                    if attr == 'ref_x':
                        ref_x = float(np.median(samp))
                    else:
                        ref_y = float(np.median(samp))
    elif ref is not None:
        raise TypeError(f"ref must be None, a (ra, dec) tuple, or an int; got {type(ref)}")

    # arcsec → pc scale factor (used for both coordinate offsets and size params)
    _pc_scale    = float(arcsec_to_kpc(1.0, z).value) * 1000.0 if use_parsecs else 1.0
    _coord_scale = _pc_scale if do_offset else 1.0

    coord_unit   = 'pc' if (do_offset and use_parsecs) else 'arcsec'
    size_unit    = 'pc' if use_parsecs else 'arcsec'
    coord_prefix = r'$\Delta$' if do_offset else ''
    x_header = rf'{coord_prefix}$x$ [{coord_unit}]'
    y_header = rf'{coord_prefix}$y$ [{coord_unit}]'

    # --- parameter display order and LaTeX header names ---
    _latex_name = {
        'flux':     f'Flux density [{flux_unit}]',
        'ab_mag':   r'$m_{\rm AB}$',
        'size':     f'sigma or R_sersic [{size_unit}]',
        'n_sersic': r'$n$',
        'e1':       r'$e_1$',
        'e2':       r'$e_2$',
        'q':        r'$q$',
        'phi':      r'$\phi$',
        'r':        rf'{coord_prefix}$r$ [{coord_unit}]',
    }
    _param_order = ['flux', 'ab_mag', 'size', 'n_sersic', 'e1', 'e2', 'q', 'phi']
    _amp_aliases = {'source_amp'}
    _size_keys   = {'R_sersic', 'sigma'}   # angular sizes converted alongside coords
    _coord_keys  = {'center_x', 'center_y', 'ra_source', 'dec_source'}

    # --- formatters ---
    def _fmt(v):
        if isinstance(v, (float, np.floating)):
            return f'{float(v):.4g}'
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v).replace('_', r'\_')

    def _hpd(samples, credible_level=0.6827):
        """Return the shortest interval containing ``credible_level`` of the samples."""
        s = np.sort(samples)
        n = len(s)
        n_in = max(1, int(np.floor(credible_level * n)))
        if n_in >= n:
            return s[0], s[-1]
        widths  = s[n_in:] - s[:n - n_in]
        i_min   = int(np.argmin(widths))
        return float(s[i_min]), float(s[i_min + n_in])

    def _fmt_unc(val, lo, hi):
        """Format ``val +hi / -lo`` with precision set by the uncertainty."""
        ref_unc = min(lo, hi)
        if ref_unc <= 0:
            return _fmt(val)
        mag      = math.floor(math.log10(ref_unc))
        decimals = max(0, 1 - mag)          # 2 sig figs on the uncertainty
        fs = f'.{decimals}f'
        return (rf'${val:{fs}}^{{+{hi:{fs}}}}_{{-{lo:{fs}}}}$')

    def _cell_from_samples(samples, point_val):
        """Format a cell using the chosen uncertainty method, or plain point estimate."""
        if samples is None:
            return _fmt(point_val)
        if uncertainty == 'median':
            p16, p50, p84 = np.percentile(samples, [16, 50, 84])
            return _fmt_unc(p50, p50 - p16, p84 - p50)
        else:  # 'hpd'
            v = float(point_val)
            hpd_lo, hpd_hi = _hpd(samples)
            return _fmt_unc(v, v - hpd_lo, hpd_hi - v)

    # --- build table rows ---
    table_rows = []
    extra_cols_seen = []
    extra_cols_set  = set()

    for kind, idx, model, kw in components:
        mk = _mcmc_kw(kind, idx)

        cx = float(kw.get('center_x', kw.get('ra_source', 0.0)))
        cy = float(kw.get('center_y', kw.get('dec_source', 0.0)))

        # Coordinate cells (with possible MCMC)
        coord_data = {}
        for coord_key, raw_val, ref_val, col in (
            ('center_x', cx, ref_x, 'x'),
            ('center_y', cy, ref_y, 'y'),
        ):
            point_out = (raw_val - ref_val) * _coord_scale
            samp_raw  = _samples(mk, coord_key) if mk is not None else None
            if samp_raw is None and mk is not None:
                # try the alternate coordinate key name
                alt = 'ra_source' if coord_key == 'center_x' else 'dec_source'
                samp_raw = _samples(mk, alt)
            samp_out = (samp_raw - ref_val) * _coord_scale if samp_raw is not None else None
            if samp_out is not None:
                # HPD: centre on best-fit; median: centre on sample median (point_out unused)
                center = point_out if uncertainty == 'hpd' else None
                cell = _cell_from_samples(samp_out, center)
            else:
                cell = _fmt(point_out)
            coord_data[col] = {
                'cell': cell,
                'point': float(point_out),
                'samples': samp_out,
            }

        row = {
            '_type': model.replace('_', r'\_'),
            'x': coord_data['x']['cell'],
            'y': coord_data['y']['cell'],
        }
        if do_offset:
            x_point = coord_data['x']['point']
            y_point = coord_data['y']['point']
            x_samp = coord_data['x']['samples']
            y_samp = coord_data['y']['samples']
            r_point = float(np.hypot(x_point, y_point))
            r_samp = None
            if x_samp is not None and y_samp is not None:
                n = min(len(x_samp), len(y_samp))
                if n > 1:
                    r_samp = np.hypot(x_samp[:n], y_samp[:n])
            elif x_samp is not None and len(x_samp) > 1:
                r_samp = np.hypot(x_samp, y_point)
            elif y_samp is not None and len(y_samp) > 1:
                r_samp = np.hypot(x_point, y_samp)
            center = r_point if uncertainty == 'hpd' else None
            row['r'] = _cell_from_samples(r_samp, center)

        for k, v in kw.items():
            if k in _coord_keys:
                continue
            col = 'flux' if k in _amp_aliases or k == 'amp' else k
            if col == 'flux':
                scaled_v    = float(v) * flux_per_amp
                samp        = _samples(mk, 'amp')
                scaled_samp = samp * flux_per_amp if samp is not None else None
                row[col] = _cell_from_samples(scaled_samp, scaled_v)
                if flux_unit == 'nJy':
                    if scaled_v > 0:
                        mag_v = 31.4 - 2.5 * np.log10(scaled_v)
                        mag_samp = None
                        if scaled_samp is not None:
                            pos = scaled_samp[scaled_samp > 0]
                            if pos.size > 1:
                                mag_samp = 31.4 - 2.5 * np.log10(pos)
                        center = mag_v if uncertainty == 'hpd' else None
                        row['ab_mag'] = _cell_from_samples(mag_samp, center)
                    else:
                        row['ab_mag'] = '--'
            elif col in _size_keys:
                scaled_v    = float(v) * _pc_scale
                samp        = _samples(mk, col)
                scaled_samp = samp * _pc_scale if samp is not None else None
                row['size'] = _cell_from_samples(scaled_samp, scaled_v)
            else:
                row[col] = _cell_from_samples(_samples(mk, col), v)
            out_col = 'size' if col in _size_keys else col
            if out_col not in extra_cols_set and out_col not in {'x', 'y'}:
                extra_cols_seen.append(out_col)
                extra_cols_set.add(out_col)
            if col == 'flux' and flux_unit == 'nJy' and 'ab_mag' not in extra_cols_set:
                extra_cols_seen.append('ab_mag')
                extra_cols_set.add('ab_mag')

        table_rows.append(row)

    # --- ordered non-coordinate param columns ---
    param_cols = [k for k in _param_order if k in extra_cols_set]
    leftover   = [k for k in extra_cols_seen if k not in set(_param_order)]
    param_cols += leftover

    all_col_keys  = ['_type', 'x', 'y'] + (['r'] if do_offset else []) + param_cols
    all_col_heads = [
        'Type', x_header, y_header,
    ] + ([_latex_name['r']] if do_offset else []) + [_latex_name.get(k, k.replace('_', r'\_')) for k in param_cols]

    ncols    = len(all_col_keys)
    col_spec = 'l' + 'r' * (ncols - 1)

    # --- assemble LaTeX ---
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        rf'\begin{{tabular}}{{{col_spec}}}',
        r'\hline\hline',
        ' & '.join(all_col_heads) + r' \\',
        r'\hline',
    ]

    for row in table_rows:
        cells = [row.get(k, '--') for k in all_col_keys]
        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')

    # caption / note
    if isinstance(ref, (tuple, list)):
        note = (
            rf'Coordinates relative to '
            rf'$(\alpha,\delta)=({ref[0]:.6f}^\circ,\,{ref[1]:.6f}^\circ)$.'
        )
    elif isinstance(ref, int):
        note = rf'Coordinates relative to component {ref} ({components[ref][2]}).'
    else:
        note = 'Coordinates in arcsec relative to the image-cutout centre.'
    if mcmc_block is not None:
        if uncertainty == 'hpd':
            note += (r' Central values are best-fit (kwargs\_opt); '
                     r'uncertainties are 68.3\,\% HPD intervals.')
        else:
            note += (r' Central values are MCMC medians; '
                     r'uncertainties are 16th/84th percentiles.')

    lines.append(rf'\caption{{{note}}}')
    lines.append(r'\end{table}')

    latex = '\n'.join(lines)
    print(latex)
    return latex