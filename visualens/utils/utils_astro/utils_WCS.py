import numpy as np
import warnings
import pyqtgraph as pg


def get_ImageItem_projected_onto_reference(reference_map_data, reference_wcs, to_project_map_data, to_project_wcs) :
    a, c, b, d, e, f = get_transform_parameters_for_project_map_onto_reference(reference_map_data, reference_wcs, to_project_map_data, to_project_wcs)
    ImageItem = pg.ImageItem(to_project_map_data)
    ImageItem.setTransform(pg.QtGui.QTransform(a, c, b, d, e, f))
    return ImageItem

def get_transform_parameters_for_project_map_onto_reference(reference_map_data, reference_wcs, to_project_map_data, to_project_wcs) :
    # Affine transform from the map's pixel grid to the view coordinates of the
    # displayed image (which is flipped vertically and has pixel centers at +0.5),
    # fitted on a grid of points spanning the map.
    ny_to_project_map, nx_to_project_map = to_project_map_data.shape
    cols, rows = np.meshgrid(np.linspace(-0.5, nx_to_project_map - 0.5, 5), np.linspace(-0.5, ny_to_project_map - 0.5, 5))
    cols, rows = cols.ravel(), rows.ravel()
    x_image, y_image = reference_wcs.celestial.world_to_pixel(to_project_wcs.pixel_to_world(cols, rows))
    x_view = x_image + 0.5
    y_view = reference_map_data.shape[0] - 0.5 - y_image
    M = np.column_stack([cols + 0.5, rows + 0.5, np.ones_like(cols)])
    (a, b, e), *_ = np.linalg.lstsq(M, x_view, rcond=None)
    (c, d, f), *_ = np.linalg.lstsq(M, y_view, rcond=None)
    residuals = np.hypot(M @ [a, b, e] - x_view, M @ [c, d, f] - y_view)
    if np.max(residuals) > 1. :
        warnings.warn(f"The map projection deviates from an affine transform by up to {np.max(residuals):.1f} "
                    "image pixels; consider reprojecting the map onto the image grid for exact alignment.")
    return a, c, b, d, e, f


