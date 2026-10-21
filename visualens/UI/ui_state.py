import colorsys
import inspect
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


ELLIPSE_MARKER = 'ellipse'
MARKER_CHOICES = (ELLIPSE_MARKER, 'o', '+', 'x', 's')


def _plot_defaults(cat) :
    """Default marker / scale / linewidth of ``cat.plot`` (they differ between catalog kinds)."""
    try :
        params = inspect.signature(cat.plot).parameters
    except (TypeError, ValueError) :
        params = {}
    def default(name, fallback) :
        param = params.get(name)
        return fallback if param is None or param.default is inspect.Parameter.empty else param.default
    marker = default('marker', None)
    return dict(
        marker=ELLIPSE_MARKER if marker is None else marker,
        scale=float(default('scale', 1.)),
        linewidth=float(default('linewidth', 3.)),
    )


@dataclass
class CatalogUIState :
    """
    Plotting/selection state of one catalog, as driven by the side panel.

    It is stored on the ``Catalog`` instance itself (see ``of``) so that it
    survives the panel's sub-tabs being rebuilt whenever a catalog is added.
    """
    plot : bool = True
    plot_column : bool = False
    colname : Optional[str] = None
    selection_panel : bool = False
    x_colname : Optional[str] = None
    y_colname : Optional[str] = None
    use_default_color : bool = True
    h : float = 0.
    s : float = 0.
    v : float = 0.
    opacity : float = 1.
    default_color : list = field(default_factory=lambda : [0., 1., 1., 1., 0.])
    marker : str = ELLIPSE_MARKER   # one of MARKER_CHOICES; ELLIPSE_MARKER means marker=None in ``cat.plot``
    scale : float = 1.
    linewidth : float = 3.

    _ATTR = '_plot_ui_state'

    @classmethod
    def existing(cls, cat) :
        """Return the state attached to ``cat``, or None if the panel never built one."""
        return getattr(cat, cls._ATTR, None)

    @classmethod
    def of(cls, cat, **defaults) :
        """Return the state attached to ``cat``, creating it from ``cat.color`` if needed."""
        state = cls.existing(cat)
        if state is None :
            default_color = [float(channel) for channel in cat.color]
            h, s, v = colorsys.rgb_to_hsv(*default_color[:3])
            opacity = default_color[3] if len(default_color)>3 else 1.
            defaults = {**_plot_defaults(cat), **defaults}
            state = cls(h=h, s=s, v=v, opacity=opacity, default_color=default_color, **defaults)
            setattr(cat, cls._ATTR, state)
        return state

    def apply_color_to(self, cat) :
        """
        Set ``cat.color`` to either its original colour or the one picked in the panel.
        ``cat.color`` is reassigned rather than edited in place: the list may be shared
        with other catalogs (it comes from a default argument).
        """
        if self.use_default_color :
            cat.color = list(self.default_color)
            return
        color = [*colorsys.hsv_to_rgb(self.h, self.s, self.v), self.opacity] + [0. if len(self.default_color)<5 else self.default_color[4]]
        cat.color = color

    def replot_column(self, cat, **plot_kwargs) :
        """
        Redraw column labels according to the current state (no-op if column plotting is off).

        With the default colour, ``cat.plot_column`` picks its own per-family palette
        for multiple images; a custom colour is passed explicitly so that it overrides
        such palettes.
        """
        if not self.plot_column or self.colname is None :
            return
        if self.use_default_color :
            cat.plot_column(self.colname)
        else :
            color = plot_kwargs.get('color', np.array(cat.color, dtype=float))
            cat.plot_column(self.colname, color=color)

    def replot(self, cat, **plot_kwargs) :
        """
        Redraw ``cat`` according to the current state.

        Markers are skipped when plotting is off; column labels are refreshed
        whenever column plotting is enabled.

        With the default colour, ``cat.plot`` / ``cat.plot_column`` pick their own
        (``cat.color``, or a per-family palette for multiple images); a custom colour
        is passed explicitly so that it also overrides such palettes.
        """
        self.apply_color_to(cat)
        plot_kwargs.setdefault('marker', None if self.marker == ELLIPSE_MARKER else self.marker)
        plot_kwargs.setdefault('scale', self.scale)
        plot_kwargs.setdefault('linewidth', self.linewidth)
        if not self.use_default_color :
            plot_kwargs.setdefault('color', np.array(cat.color, dtype=float))
        if self.plot :
            cat.plot(**plot_kwargs)
        self.replot_column(cat, **plot_kwargs)


@dataclass
class LensModelUIState :
    """Which broad multiple-image families are ticked in the Lens model tab."""
    checked_broad_families : set = field(default_factory=set)
    seen_broad_families : set = field(default_factory=set)

    _ATTR = '_ui_state'

    @classmethod
    def of(cls, lens_model) :
        state = getattr(lens_model, cls._ATTR, None)
        if state is None :
            state = cls()
            setattr(lens_model, cls._ATTR, state)
        return state

    def sync_with(self, broad_families) :
        """Register ``broad_families``; families not seen before are checked by default."""
        broad_families = list(broad_families)
        self.checked_broad_families.update(n for n in broad_families if n not in self.seen_broad_families)
        self.seen_broad_families = set(broad_families)

    def set_checked(self, name, checked) :
        if checked :
            self.checked_broad_families.add(name)
        else :
            self.checked_broad_families.discard(name)

    def checked_in_order(self, broad_families) :
        return [n for n in broad_families if n in self.checked_broad_families]
