import pyqtgraph as pg
from PyQt5.QtCore import Qt
from astropy.table import Table


class _HandSelectionGroup :
    """Shared state for a group of linked ``ImageView_custom_selector`` instances.

    Linking views (via ``ImageView_custom_selector.link()``) makes them share a
    single hand-selected catalog, the same pending/confirmed marker lists and
    the same prefix/suffix id counter. As a result, double-clicking in any
    linked view appends to the same catalog and the same markers are shown in
    every linked view, and pressing Enter in any one of them confirms the
    pending markers (and advances the shared id counter) for the whole group.
    """
    def __init__(self) :
        self.catalog = Table(names=['id', 'ra', 'dec'], dtype=['str', 'float64', 'float64'])
        self.prefix = '1'
        self.suffix = 0
        self.pending_x = []
        self.pending_y = []
        self.confirmed_x = []
        self.confirmed_y = []
        self.members = []

    def refresh_scatters(self) :
        for view in self.members :
            view._plus_scatter.setData(self.pending_x, self.pending_y)
            view._circle_scatter.setData(self.confirmed_x, self.confirmed_y)


class ImageView_custom_selector(pg.ImageView) :
    def __init__(self, wcs, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.wcs = wcs
        self._group = _HandSelectionGroup()
        self._group.members.append(self)
        
        self._plus_scatter = pg.ScatterPlotItem(
            size=18, symbol='+', pen=pg.mkPen('w', width=2), brush=pg.mkBrush('w')
        )
        self._circle_scatter = pg.ScatterPlotItem(
            size=14, symbol='o', pen=pg.mkPen('w', width=2), brush=pg.mkBrush(0, 0, 0, 0)
        )
        self.addItem(self._plus_scatter)
        self.addItem(self._circle_scatter)
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.scene.sigMouseClicked.connect(self._on_mouse_clicked)
    
    
    @property
    def catalog(self) :
        return self._group.catalog
    
    @property
    def prefix(self) :
        return self._group.prefix
    
    @prefix.setter
    def prefix(self, value) :
        self._group.prefix = value
    
    
    @staticmethod
    def link(views) :
        """Link several ``ImageView_custom_selector`` instances into one group.

        After linking, all views in ``views`` share the same hand-selected
        catalog, pending/confirmed marker lists and prefix/suffix counter, so
        double-clicking in any of them appends to the same catalog (with the
        markers shown in every linked view), and pressing Enter in any one of
        them confirms the pending markers for the whole group. Any rows or
        markers already present in the views being merged in are preserved.
        Calling this again with an updated set of views (e.g. after opening
        or closing extra panes) re-syncs the group without losing data; views
        left out of ``views`` are simply dropped from the group.
        """
        views = [v for v in views if isinstance(v, ImageView_custom_selector)]
        if not views :
            return
        
        canonical = views[0]._group
        for view in views[1:] :
            group = view._group
            if group is canonical :
                continue
            if len(group.catalog) > 0 :
                for row in group.catalog :
                    canonical.catalog.add_row({'id': row['id'], 'ra': row['ra'], 'dec': row['dec']})
            canonical.pending_x.extend(group.pending_x)
            canonical.pending_y.extend(group.pending_y)
            canonical.confirmed_x.extend(group.confirmed_x)
            canonical.confirmed_y.extend(group.confirmed_y)
            canonical.suffix = max(canonical.suffix, group.suffix)
            if int(group.prefix) > int(canonical.prefix) :
                canonical.prefix = group.prefix
            view._group = canonical
        
        canonical.members = [view for view in canonical.members if view in views]
        for view in views :
            if view not in canonical.members :
                canonical.members.append(view)
        
        canonical.refresh_scatters()
    
    
    def _on_mouse_clicked(self, evt) :
        if not evt.double() :
            return
        pos = evt.scenePos()
        if not self.getView().sceneBoundingRect().contains(pos) :
            return
        
        mouse_point = self.getView().mapSceneToView(pos)
        x, y_display = mouse_point.x(), mouse_point.y()
        
        # Display y -> FITS/WCS pixel y (image is typically flipped for pyqtgraph)
        if self.image is not None :
            y_wcs = self.image.shape[0] - y_display
        else :
            y_wcs = y_display
        
        world = self.wcs.pixel_to_world(x, y_wcs)
        ra, dec = float(world.ra.deg), float(world.dec.deg)
        
        group = self._group
        group.suffix += 1
        obj_id = group.prefix + '.' + str(group.suffix)
        group.catalog.add_row({'id': obj_id, 'ra': ra, 'dec': dec})
        
        group.pending_x.append(x)
        group.pending_y.append(y_display)
        group.refresh_scatters()
    
    
    def keyPressEvent(self, event) :
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) :
            self._confirm_markers()
        else :
            super().keyPressEvent(event)
    
    
    def _confirm_markers(self) :
        group = self._group
        group.confirmed_x.extend(group.pending_x)
        group.confirmed_y.extend(group.pending_y)
        group.pending_x.clear()
        group.pending_y.clear()
        group.prefix = str(int(group.prefix) + 1)
        group.suffix = 0
        group.refresh_scatters()
    
    
    def clear_selection(self) :
        group = self._group
        group.prefix = '1'
        group.suffix = 0
        group.pending_x.clear()
        group.pending_y.clear()
        group.confirmed_x.clear()
        group.confirmed_y.clear()
        if len(group.catalog) > 0 :
            group.catalog.remove_rows(slice(0, len(group.catalog)))
        group.refresh_scatters()
