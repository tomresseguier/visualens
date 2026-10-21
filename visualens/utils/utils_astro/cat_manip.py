from astropy.table import Table, Column, MaskedColumn, vstack
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, UnitSphericalRepresentation, SphericalRepresentation
from scipy import spatial



def vstack_catalogs(catalogs_paths):
    catalogs = [Table.read(path, format='fits') for path in catalogs_paths]
    for i, catalog in enumerate(catalogs):
        print(f"length of catalog {i}: {len(catalog)}")
    return vstack(catalogs)


def match_cat2(cats, match_radius=0.5, fill_in_value=np.nan, return_match_idx=False, keep_all_col=False, column_to_transfer=None):
    """
    Matches two catalogs based on their sky coordinates and transfers columns from one to the other.

    Parameters
    ----------
    cats : list of (str or astropy.table.Table)
        A list containing two catalogs. The first is the receiver catalog and the second is the giver.
    match_radius : float, optional
        The radius in arcseconds within which to consider sources as matched. Default is 0.5.
    fill_in_value : float, optional
        The value to use for non-matched sources in the transferred columns. Default is np.nan.
        For string columns, NaN is replaced by "...". For integer/boolean columns where the value
        cannot be represented (e.g. NaN), the column is masked instead. SkyCoord columns are filled with NaN coordinates.
    return_match_idx : bool, optional
        If True, returns the indices of the matched sources in the giver catalog. Default is False.
    keep_all_col : bool, optional
        If True, appends '_CAT2' to the column name in case of a name conflict. Default is False.
    column_to_transfer : list of str, optional
        A list of column names to transfer. If None, all columns are transferred. Default is None.

    Returns
    -------
    astropy.table.Table or tuple
        The matched catalog. If `return_match_idx` is True, it returns a tuple of the catalog and the match indices.
    """
    catalogs = []
    for cat in cats:
        if isinstance(cat, str):
            catalogs.append(Table.read(cat, format='fits'))
        else:
            catalogs.append(cat.copy())
    cat_receiver, cat_giver = catalogs

    ra_dec_names = [('RA', 'DEC'), ('ra', 'dec')]
    ra_dec_names_receiver = next(((ra, dec) for ra, dec in ra_dec_names if ra in cat_receiver.colnames), (None, None))
    ra_dec_names_giver = next(((ra, dec) for ra, dec in ra_dec_names if ra in cat_giver.colnames), (None, None))

    if not ra_dec_names_receiver[0] or not ra_dec_names_giver[0]:
        raise ValueError("RA/DEC columns not found in one of the catalogs.")
    
    coords_cat_receiver = SkyCoord(cat_receiver[ra_dec_names_receiver[0]], cat_receiver[ra_dec_names_receiver[1]], unit=u.deg)
    coords_cat_giver = SkyCoord(cat_giver[ra_dec_names_giver[0]], cat_giver[ra_dec_names_giver[1]], unit=u.deg)
    
    idx, d2d, _ = coords_cat_receiver.match_to_catalog_sky(coords_cat_giver)
    match_mask = d2d < match_radius * u.arcsec
    
    matched_cat = cat_receiver.copy()
    
    columns_to_process = cat_giver.colnames if column_to_transfer is None else column_to_transfer

    for colname in columns_to_process:
        if colname not in matched_cat.colnames:
            new_colname = colname
        elif keep_all_col:
            new_colname = f"{colname}_CAT2"
        else:
            continue
        matched_cat[new_colname] = make_transferred_column(cat_giver[colname], idx, match_mask, fill_in_value)

    if return_match_idx:
        return matched_cat, idx
    return matched_cat


def _is_nan(value):
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_representable(value, dtype):
    """True if `value` can be stored exactly in an array of the given (integer/bool) dtype."""
    try:
        if isinstance(value, (str, bytes)) or _is_nan(value) or not np.isfinite(value):
            return False
        if dtype.kind == 'b':
            return value in (0, 1)
        if value != int(value):
            return False
        info = np.iinfo(dtype)
        return info.min <= int(value) <= info.max
    except (TypeError, ValueError, OverflowError):
        return False


def _fill_skycoord(rows, unmatched, fill_value):
    """SkyCoord with NaN coordinates (or `fill_value` if it is itself a SkyCoord) on unmatched rows."""
    if isinstance(fill_value, SkyCoord):
        rows = rows.copy()
        rows[unmatched] = fill_value
        return rows
    sph = rows.spherical
    lon, lat = sph.lon.value.copy(), sph.lat.value.copy()
    lon[unmatched] = np.nan
    lat[unmatched] = np.nan
    lon, lat = lon * sph.lon.unit, lat * sph.lat.unit
    if sph.distance.unit.is_equivalent(u.dimensionless_unscaled):
        rep = UnitSphericalRepresentation(lon, lat)
    else:
        dist = sph.distance.value.copy()
        dist[unmatched] = np.nan
        rep = SphericalRepresentation(lon, lat, dist * sph.distance.unit)
    return SkyCoord(rep, frame=rows.frame.replicate_without_data())


def _fill_plain_column(rows, unmatched, fill_value):
    """`rows` is a (Masked)Column already holding giver values; overwrite `unmatched` entries with the fill value."""
    dtype = rows.dtype
    kind = dtype.kind
    fill_is_nan = _is_nan(fill_value)

    if kind in ('U', 'S'):
        if fill_is_nan:
            fill_str = "..."
        elif isinstance(fill_value, bytes):
            fill_str = fill_value.decode()
        else:
            fill_str = str(fill_value)
        if kind == 'S':
            fill_item = fill_str.encode()
            width = max(dtype.itemsize, len(fill_item), 1)
            rows = rows.astype(f'S{width}')
        else:
            fill_item = fill_str
            width = max(dtype.itemsize // 4, len(fill_item), 1)
            rows = rows.astype(f'<U{width}')
        rows[unmatched] = fill_item
        return rows

    if kind in ('f', 'c'):
        rows[unmatched] = np.nan if fill_is_nan else fill_value
        return rows

    if kind in ('i', 'u', 'b'):
        if _is_representable(fill_value, dtype):
            rows[unmatched] = fill_value
            return rows
        rows = MaskedColumn(rows)
        rows[unmatched] = np.ma.masked
        return rows

    if kind in ('M', 'm'):
        rows[unmatched] = np.array('NaT', dtype=dtype) if fill_is_nan else fill_value
        return rows

    rows[unmatched] = None if fill_is_nan else fill_value
    return rows


def make_transferred_column(giver_col, idx, match_mask, fill_in_value=np.nan):
    """
    Build a column of length len(idx) holding giver_col[idx] where match_mask is True and a fill value elsewhere.

    The output keeps the giver's type: dtype, shape (multidimensional columns), unit, description, metadata,
    SkyCoord / Quantity / Time mixin columns. Unmatched rows are filled with:
      - float/complex columns: `fill_in_value` (NaN by default)
      - string columns: "..." if `fill_in_value` is NaN, otherwise str(`fill_in_value`) (width is expanded if needed)
      - integer/bool columns: `fill_in_value` if exactly representable, otherwise the entry is masked
      - datetime columns: NaT if `fill_in_value` is NaN
      - SkyCoord columns: NaN coordinates
      - other mixins (e.g. Time): masked entries
    """
    unmatched = ~np.asarray(match_mask, dtype=bool)
    rows = giver_col[idx]

    if isinstance(giver_col, SkyCoord):
        return _fill_skycoord(rows, unmatched, fill_in_value)

    if isinstance(giver_col, Column):
        return _fill_plain_column(rows, unmatched, fill_in_value)

    if isinstance(giver_col, u.Quantity):
        rows = rows.copy()
        if not isinstance(fill_in_value, u.Quantity):
            fill_in_value = fill_in_value * rows.unit
        rows[unmatched] = fill_in_value
        return rows

    try:
        rows = rows.copy()
        rows[unmatched] = np.ma.masked
    except Exception as err:
        raise TypeError(f"Cannot build a filled column for giver column of type {type(giver_col).__name__}: {err}") from err
    return rows


def _get_filled_and_empty_columns(catalog):
    if catalog.mask is None:
        return catalog.colnames, []
    
    empty_columns = [name for name in catalog.colnames if np.all(catalog.mask[name])]
    filled_columns = [name for name in catalog.colnames if name not in empty_columns]
    
    return filled_columns, empty_columns


def combine_catalogs_remove_double_detections(cats, match_radius=0.5, preference_criterion='area'):
    """
    Combines a list of catalogs, removing double detections based on a preference criterion.

    Parameters
    ----------
    cats : list of (str or astropy.table.Table)
        A list of catalogs to combine.
    match_radius : float, optional
        The matching radius in arcseconds. Default is 0.5.
    preference_criterion : str, optional
        The criterion to decide which detection to keep ('area' or other). Default is 'area'.

    Returns
    -------
    astropy.table.Table
        The combined catalog with double detections removed.
    """
    catalogs = [Table.read(cat, format='fits') if isinstance(cat, str) else cat.copy() for cat in cats]
    
    initial_len = sum(len(cat) for cat in catalogs)
    
    catalog_previous = catalogs[0]

    for i in range(1, len(catalogs)):
        catalog_new = catalogs[i]
        print(f'##############\nStep {i}')
        print(f"length of catalog {i-1} before removal: {len(catalog_previous)}")
        print(f"length of catalog {i} before removal: {len(catalog_new)}")

        coords_previous = SkyCoord(catalog_previous['RA'], catalog_previous['DEC'], unit=u.deg)
        coords_new = SkyCoord(catalog_new['RA'], catalog_new['DEC'], unit=u.deg)
        idx, d2d, _ = coords_previous.match_to_catalog_sky(coords_new)
        separation_mask = d2d < match_radius * u.arcsec

        filled_columns_previous, _ = _get_filled_and_empty_columns(catalog_previous)
        filled_columns_new, _ = _get_filled_and_empty_columns(catalog_new)

        from_previous_to_new = set(filled_columns_previous) - set(filled_columns_new)
        from_new_to_previous = set(filled_columns_new) - set(filled_columns_previous)

        for col in from_new_to_previous:
            if col not in catalog_previous.colnames:
                catalog_previous[col] = np.full(len(catalog_previous), np.nan)
            catalog_previous[col][separation_mask] = catalog_new[col][idx[separation_mask]]

        for col in from_previous_to_new:
            if col not in catalog_new.colnames:
                catalog_new[col] = np.full(len(catalog_new), np.nan)
            catalog_new[col][idx[separation_mask]] = catalog_previous[col][separation_mask]

        if preference_criterion == 'area':
            areas_previous = catalog_previous["a"][separation_mask] * catalog_previous["b"][separation_mask]
            areas_new = catalog_new["a"][idx[separation_mask]] * catalog_new["b"][idx[separation_mask]]
            to_keep = np.argmax([areas_previous, areas_new], axis=0)

        where_separation = np.where(separation_mask)[0]
        to_remove_previous = where_separation[to_keep == 1]
        to_remove_new = idx[separation_mask][to_keep == 0]

        catalog_previous.remove_rows(to_remove_previous)
        catalog_new.remove_rows(to_remove_new)

        print(f"length of catalog {i-1} after removal: {len(catalog_previous)}")
        print(f"length of catalog {i} after removal: {len(catalog_new)}")

        catalog_previous = vstack([catalog_previous, catalog_new], join_type='outer')

    final_len = len(catalog_previous)
    print(f'initial length: {initial_len}')
    print(f'final length: {final_len}')
    print(f'{initial_len - final_len} matched sources removed')
    return catalog_previous


def combine_DIM_catalogs(cats):
    """
    Combines Double Image Mode catalogs from SourceExtractor by taking the mean of columns with incomplete data.

    Parameters
    ----------
    cats : list of (str or astropy.table.Table)
        A list of catalogs to be combined.

    Returns
    -------
    astropy.table.Table
        The combined catalog.
    """
    catalogs = [Table.read(cat, format='fits') if isinstance(cat, str) else cat.copy() for cat in cats]
    
    col_info = {}
    for i, cat in enumerate(catalogs):
        for colname in cat.colnames:
            if colname not in col_info:
                col_info[colname] = {'complete': [], 'incomplete': []}
            
            if np.isnan(cat[colname]).any():
                col_info[colname]['incomplete'].append(i)
            else:
                col_info[colname]['complete'].append(i)

    combined_cat = Table()
    for colname, info in col_info.items():
        if info['complete']:
            combined_cat[colname] = catalogs[info['complete'][0]][colname]
        else:
            incomplete_cols = np.array([catalogs[i][colname] for i in info['incomplete']])
            combined_cat[colname] = np.nanmean(incomplete_cols, axis=0)
            
    return combined_cat


def remove_object(cat, FWHM_to_radius=1):
    """
    Removes objects that are too close to each other, keeping the one with the larger FWHM.

    Parameters
    ----------
    cat : str or astropy.table.Table
        The input catalog.
    FWHM_to_radius : float, optional
        A factor to convert FWHM to a radius for overlap checking. Default is 1.

    Returns
    -------
    astropy.table.Table
        The catalog with overlapping objects removed.
    """
    if isinstance(cat, str):
        cat = Table.read(cat, format='fits')

    print(f"num of objects in the catalogue: {len(cat)}")

    x = np.array(cat['X_IMAGE'])
    y = np.array(cat['Y_IMAGE'])
    fwhm = np.array(cat['FWHM_IMAGE']) * FWHM_to_radius
    
    ori_order = np.arange(len(x))
    
    # Sort by FWHM in descending order
    sort_indices = np.argsort(fwhm)[::-1]
    x_sorted, y_sorted, fwhm_sorted, ori_order_sorted = x[sort_indices], y[sort_indices], fwhm[sort_indices], ori_order[sort_indices]

    pos = np.column_stack((x_sorted, y_sorted))
    tree = spatial.KDTree(pos)
    
    keep_mask = np.ones(len(x), dtype=bool)
    
    for i in range(len(x)):
        if not keep_mask[i]:
            continue
            
        # Find neighbors within 2 * FWHM
        neighbors_idx = tree.query_ball_point(pos[i], 2.0 * fwhm_sorted[i])
        
        for index in neighbors_idx:
            if i == index:
                continue
            
            distance = np.hypot(pos[index, 0] - pos[i, 0], pos[index, 1] - pos[i, 1])
            if distance < (fwhm_sorted[i] + fwhm_sorted[index]):
                keep_mask[index] = False

    final_indices = ori_order_sorted[keep_mask]
    
    cat_filtered = cat[final_indices]
    
    print(f"Num of objects after removing double-detection: {len(cat_filtered)}")
    return cat_filtered




















