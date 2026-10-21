import numpy as np
import pyqtgraph as pg
from .utils.operations import make_image_to_source
from astropy.table import Table
from PyQt5.QtCore import Qt







def start_im2source(self) :
    self._initial_ngrid_value = self.lt.G.ngrid
    self.lt.set_grid(50, 0)
    
    self.transform_coords_radec = make_image_to_source(self.alpha[0], self.alpha[1], self._maps_wcs)
    
    def transform_coords(x, y) :
        ra, dec = self.image.image_to_world(x, y)
        
        ra_source, dec_source = self.transform_coords_radec(ra, dec)
        x_source, y_source = self.image.world_to_image(ra_source, dec_source)
        return x_source, y_source, ra_source, dec_source
        
    self.transform_coords = transform_coords
    
    # Set up label if needed (optional)
    if not hasattr(self, 'transform_label'):
        self.transform_label = pg.TextItem(anchor=(0, 1), color='w')
        self.image.ImageView.addItem(self.transform_label)
    
    # Create scatter point for transformed location
    self.transformed_point = pg.ScatterPlotItem(size=10, brush='r')
    self.image.ImageView.addItem(self.transformed_point)
    
    self.images_scatter = pg.ScatterPlotItem(size=10, symbol='+', brush='g')
    self.image.ImageView.addItem(self.images_scatter)

    # Ensure view does not auto-range when updating
    self.image.ImageView.getView().enableAutoRange(pg.ViewBox.XAxis, False)
    self.image.ImageView.getView().enableAutoRange(pg.ViewBox.YAxis, False)

    # Mouse tracking setup
    def mouse_moved(evt):
        pos = evt[0]
        if self.image.ImageView.getView().sceneBoundingRect().contains(pos):
            mouse_point = self.image.ImageView.getView().mapSceneToView(pos)
            x, y_flipped = mouse_point.x(), mouse_point.y()
            x, y = x, self.image.image_data.shape[0] - y_flipped
            try:
                x_source, y_source, ra_source, dec_source = self.transform_coords(x, y)
                xr_source, yr_source = self.world_to_relative(ra_source, dec_source)
                
                #if int( importlib.metadata.version('lenstool').split('.')[1] )>=6 :
                    ### For Lenstool version 8.6.3 ###
                #print(str(ra_source)[3:], str(dec_source)[3:])
                source = Table( rows=[('test', ra_source, dec_source, 1, 1, 0, self.z_source, 25)],names=['n','x','y','a','b','theta','z','mag'], dtype=['str',*['float',]*7] )
                #else :
                    ### For Lenstool version ?? ###
                    #source = Table( rows=[('test', xr_source, yr_source, 1, 1, 0, self.z_source, 25)],names=['n','x','y','a','b','theta','z','mag'], dtype=['str',*['float',]*7] )
                
                self.lt.set_sources(source)
                self.lt.e_lensing()
                image_cat = self.lt.get_images()
                
                x_images = []
                y_images = []
                for image in image_cat :
                    ra_image, dec_image = self.relative_to_world(image['x'], image['y'])
                    x_image, y_image = self.image.world_to_image(ra_image, dec_image)
                    x_images.append(x_image)
                    y_images.append(y_image)
                    
                    #ellipse = QGraphicsEllipseItem(x_image, self.image.image_data.shape[0] - y_image, image['a'], image['b'])
                    #ellipse.setTransformOriginPoint( PyQt5.QtCore.QPointF(x_image, self.image.image_data.shape[0] - y_image) )
                    #ellipse.setRotation(-image['theta'])
                self.images_scatter.setData( x_images, self.image.image_data.shape[0] - np.array(y_images) )
                
                self.transformed_point.setData([x_source], [self.image.image_data.shape[0] - y_source])
                self.transform_label.setText(f"({x:.2f}, {y:.2f}) → ({x_source:.2f}, {y_source:.2f})")
                self.transform_label.setPos(x, y_flipped)
                
                self._last_transform_coords = {'x': x, 'y': y, 'x_source': x_source, 'y_source': y_source, 'x_images': x_images, 'y_images': y_images}
            except Exception as e:
                self.transform_label.setText(f"Error: {e}")
                self.transform_label.setPos(x, y_flipped)
    
    self._transform_proxy = pg.SignalProxy(self.image.ImageView.getView().scene().sigMouseMoved, rateLimit=60, slot=mouse_moved)
    
    
    
    
    self.doubleclick_image_marker = pg.ScatterPlotItem(size=12, symbol='o', brush='b', pen='b')
    self.doubleclick_source_marker = pg.ScatterPlotItem(size=12, symbol='+', brush='r', pen='r')
    self.doubleclick_images_marker = pg.ScatterPlotItem(size=12, symbol='x', brush='g', pen='g')
    self.image.ImageView.addItem(self.doubleclick_image_marker)
    self.image.ImageView.addItem(self.doubleclick_source_marker)
    self.image.ImageView.addItem(self.doubleclick_images_marker)
    
    self.image_markers_x = []
    self.image_markers_y = []
    self.source_markers_x = []
    self.source_markers_y = []
    self.images_markers_x = []
    self.images_markers_y = []
    
    def mouse_clicked(evt):
        if evt.double():
            if hasattr(self, '_last_transform_coords'):
                coords = self._last_transform_coords
                x, y = coords['x'], coords['y']
                x_source, y_source = coords['x_source'], coords['y_source']
                x_images, y_images = coords['x_images'], coords['y_images']
                
                y0 = self.image.image_data.shape[0]
                self.image_markers_x.append(x)
                self.image_markers_y.append(y0 - y)
                self.source_markers_x.append(x_source)
                self.source_markers_y.append(y0 - y_source)
                self.images_markers_x += x_images
                self.images_markers_y += [ y0 - y_im for y_im in y_images]
                
                self.doubleclick_image_marker.setData( self.image_markers_x, self.image_markers_y )
                self.doubleclick_source_marker.setData( self.source_markers_x, self.source_markers_y )
                self.doubleclick_images_marker.setData( self.images_markers_x, self.images_markers_y )
    self._doubleclick_connection = self.image.ImageView.scene.sigMouseClicked.connect(mouse_clicked)
    
    
    def keyPressEvent(event):
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Space :
            self.stop_im2source()
            
            self.image.ImageView.keyPressEvent = self._original_keyPressEvent
            #event.accept()
            print('im2source stopped')
    
    self._original_keyPressEvent = self.image.ImageView.keyPressEvent
    self.image.ImageView.keyPressEvent = keyPressEvent
    
def stop_im2source(self) :
    self.lt.set_grid(self._initial_ngrid_value, 0)
    self.image.ImageView.removeItem(self.images_scatter)
    self.image.ImageView.removeItem(self.transformed_point)
    self.image.ImageView.removeItem(self.transform_label)
    self._transform_proxy.disconnect()
    del self._transform_proxy
    del self.images_scatter
    del self.transformed_point
    del self.transform_label
    self.image.QMainWindow.keyPressEvent = self._original_keyPressEvent
    if hasattr(self, 'doubleclick_image_marker'):
        self.image.ImageView.removeItem(self.doubleclick_image_marker)
        del self.doubleclick_image_marker
    if hasattr(self, 'doubleclick_source_marker'):
        self.image.ImageView.removeItem(self.doubleclick_source_marker)
        del self.doubleclick_source_marker
    if hasattr(self, 'doubleclick_images_marker'):
        self.image.ImageView.removeItem(self.doubleclick_images_marker)
        del self.doubleclick_images_marker
    if hasattr(self, '_doubleclick_connection'):
        self.image.ImageView.scene.sigMouseClicked.disconnect(self._doubleclick_connection)
        del self._doubleclick_connection
    print('Interactive source & images viewer closed.')