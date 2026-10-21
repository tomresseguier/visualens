"""
Custom 5-control-point versions of matplotlib colormaps, for pyqtgraph.

pyqtgraph's HistogramLUTWidget shows one draggable gradient tick per colormap
control point. The stock viridis/plasma/... maps carry 256 points (and
coolwarm/RdBu 33/11), which clutters the histogram with sliders. The maps below
are the same colormaps reduced to 5 evenly spaced control points
(positions 0, 0.25, 0.5, 0.75, 1), with colors sampled from matplotlib.
"""
import pyqtgraph as pg


N_CONTROL_POINTS = 5
_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

# name -> 5 RGB colors (0-255), sampled from the matplotlib colormap of the same name.
_CUSTOM_COLORS = {
    'viridis' : [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    'plasma' : [(13, 8, 135), (126, 3, 168), (204, 71, 120), (248, 149, 64), (240, 249, 33)],
    'inferno' : [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)],
    'magma' : [(0, 0, 4), (81, 18, 124), (183, 55, 121), (252, 137, 97), (252, 253, 191)],
    'cividis' : [(0, 34, 78), (67, 78, 108), (125, 124, 120), (188, 174, 108), (254, 232, 56)],
    'turbo' : [(48, 18, 59), (40, 188, 235), (164, 252, 60), (251, 126, 33), (122, 4, 3)],
    'coolwarm' : [(59, 76, 192), (141, 176, 254), (221, 220, 220), (244, 152, 122), (180, 4, 38)],
    'RdBu' : [(103, 0, 31), (229, 131, 104), (246, 247, 247), (104, 171, 208), (5, 48, 97)],
}


def _make_colormap(name, colors) :
    return pg.ColorMap(list(_POSITIONS), colors, name=name)


CUSTOM_COLORMAPS = {name : _make_colormap(name, colors) for name, colors in _CUSTOM_COLORS.items()}


def get_colormap(cmap) :
    """
    Resolve a colormap to a ``pg.ColorMap``.

    ``cmap`` may be a ``pg.ColorMap`` (returned unchanged) or a name. Names
    with a custom 5-point version in ``CUSTOM_COLORMAPS`` return it; any other
    name is looked up in pyqtgraph, then in matplotlib.
    """
    if not isinstance(cmap, str) :
        return cmap
    if cmap in CUSTOM_COLORMAPS :
        return CUSTOM_COLORMAPS[cmap]
    try :
        return pg.colormap.get(cmap)
    except FileNotFoundError :
        return pg.colormap.get(cmap, source='matplotlib')
