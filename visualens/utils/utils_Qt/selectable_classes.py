from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsSceneMouseEvent, QApplication
from PyQt5.QtCore import Qt, QObject, QEvent
from pyqtgraph.Qt import QtCore
import PyQt5
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm

from .utils_general import transform_rectangle, InRectangle, make_full_color
from .Special_QWidgets import DragPlotWidget

##############################################################################
# Default color format is color = [R, G, B, contour opacity, inside opacity] #
##############################################################################


class SelectableEllipse(QGraphicsEllipseItem) :
    def __init__(self, x, y, width, height, idx, selection_mask, initial_color, selection_color=None, 
                 scatter_pos=None, Scatter_widget=None, linewidth=3):
        #super(SelectableEllipse, self).__init__(x, y, width, height)
        super().__init__(x, y, width, height)
        self.idx = idx
        self.is_selected = False
        self.selection_mask = selection_mask
        self.linewidth = linewidth
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)

        self.initial_color = make_full_color(initial_color)
        if selection_color is not None :
            self.selection_color = make_full_color(selection_color)
        else :
            # Default selection color is white with the same opacities as the initial color
            self.selection_color = self.initial_color.copy()
            for i in range(3) :
                self.selection_color[i] = 255
        
        self.setPen( pg.mkPen(self.initial_color[:4], width=linewidth) )
        self.setBrush( pg.mkBrush(self.initial_color[:3] + [self.initial_color[4]]) )
        
        self.scatter_pos = scatter_pos
        self.Scatter_widget = Scatter_widget
        self.selection_scatter = None

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_selected = not self.is_selected
            print(f"Object {self.idx} selected") if self.is_selected else print(f"Object {self.idx} deselected")
            
            self.selection_mask[self.idx] = not self.selection_mask[self.idx]
            if self.is_selected :
                self.setPen( pg.mkPen(self.selection_color[:4], width=self.linewidth) )
                self.setBrush( pg.mkBrush(self.selection_color[:3] + [self.selection_color[4]]) )
            else :
                self.setPen( pg.mkPen(self.initial_color[:4], width=self.linewidth) )
                self.setBrush( pg.mkBrush(self.initial_color[:3] + [self.initial_color[4]]) )
            
            
            if self.Scatter_widget is not None :
                if self.selection_scatter is None :
                    self.selection_scatter = pg.ScatterPlotItem()
                    self.Scatter_widget.addItem(self.selection_scatter)
                if self.selection_mask[self.idx] :
                    self.selection_scatter.setData([self.scatter_pos[0]], [self.scatter_pos[1]], pen=(255, 0, 0), size=20, symbol='x')
                else :
                    self.selection_scatter.setData([], [])
            
            
        
class SelectableScatter(DragPlotWidget) :
    def __init__(self, cat_instance, selection_color=[1, 0, 0]) :
        super().__init__()
        self.cat_instance = cat_instance
        self.xy = (cat_instance.x_axis_cleaned, cat_instance.y_axis_cleaned)
        self.plot(self.xy[0], self.xy[1], pen=None, symbol='o', symbolBrush='g', symbolSize=2)
        
        self.full_mask = cat_instance.selection_mask.copy()
        self.qtItems = cat_instance.qtItems
        #self.initial_color = make_full_color(cat_instance.color)
        self.selection_color = make_full_color(selection_color)
        
        self.selection_scatter = pg.ScatterPlotItem()
        self.selection_scatter.setData([],[], pen=(255, 0, 0), size=2)
        self.addItem(self.selection_scatter)
    
    def checkLongPress(self) :
        super().checkLongPress()
        if not (QApplication.mouseButtons() & Qt.LeftButton) and not (QApplication.mouseButtons() & Qt.ControlModifier):
            current_roi = self.last_roi
            def ROI_changed() :
                x0 = current_roi.getState()['pos'][0]
                y0 = current_roi.getState()['pos'][1]
                a = current_roi.getState()['size'][0]
                b = current_roi.getState()['size'][1]
                angle = current_roi.getState()['angle']
                
                rect_params = transform_rectangle(x0, y0, a, b, angle*np.pi/180)
                current_roi.selection_mask = InRectangle(self.xy[0], self.xy[1], rect_params)
                
                mask_matrix = np.full((len(self.roi_list), len(self.xy[0])), False)
                for i, roi in enumerate(self.roi_list) :
                    mask_matrix[i] = roi.selection_mask
                    
                self.full_mask_previous = self.full_mask.copy()
                self.full_mask = np.any(mask_matrix, axis=0)
                to_plot_x = self.xy[0][self.full_mask]
                to_plot_y = self.xy[1][self.full_mask]
                self.selection_scatter.setData(to_plot_x, to_plot_y)                
                
                to_change = np.where( np.logical_xor(self.full_mask_previous, self.full_mask) )[0]
                for i in tqdm(to_change) :
                    if self.full_mask[i] :
                        self.qtItems[i].setPen( pg.mkPen(self.selection_color[:3] + self.selection_color[-1:], width=self.qtItems[i].linewidth) )
                        self.qtItems[i].setBrush( pg.mkBrush(self.selection_color[:4]) )
                    else :
                        self.qtItems[i].setPen( pg.mkPen(self.qtItems[i].initial_color[:3] + self.qtItems[i].initial_color[-1:], width=self.qtItems[i].linewidth) )
                        self.qtItems[i].setBrush( pg.mkBrush(self.qtItems[i].initial_color[:4]) )
                
                self.cat_instance.selection_mask = self.full_mask.copy()
            
            ROI_changed()
            current_roi.sigRegionChangeFinished.connect(ROI_changed)
    
    def keyPressEvent(self, event) : #To insure galaxies get deselected when ROIs get deleted
        if event.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete) :
            for roi in self.roi_list :
                if roi.selected :
                    state = roi.getState()
                    state['size'] = (0,0)
                    roi.setState(state)
        if event.key() == Qt.Key_Escape :
            for roi in self.roi_list :
                state = roi.getState()
                state['size'] = (0,0)
                roi.setState(state)
        super().keyPressEvent(event)
        
        
class SelectSources() : #pg.PlotWidget()
    def __init__(self, cat_instance, selection_color=[1, 0, 0]) :
        self.cat_instance = cat_instance
        self.selection_ROI = cat_instance.image.QWidget.current_ROI
        self.selection_mask_temp = np.full(len(self.cat_instance.cat), False)
        self.selection_ROI.sigRegionChangeFinished.connect(self.selection_ROI_changed)
        self.selection_scatter = None
        #self.initial_color = make_full_color(cat_instance.color)
        self.selection_color = make_full_color(selection_color)
        self.confirm_selection_filter = SelectSources_KeyPressFilter(self)
        #self.QMainWindow = window
        cat_instance.image.QMainWindow.installEventFilter(self.confirm_selection_filter)
        self.make_selection()
        
        
    def selection_ROI_changed(self) :
        self.make_selection()
    
        
    def make_selection(self) :
        size_y = self.cat_instance.image.ImageView.image.shape[0]
        
        x = self.cat_instance.cat['x']
        y = size_y - self.cat_instance.cat['y']
        
        x0 = self.selection_ROI.getState()['pos'][0]
        y0 = self.selection_ROI.getState()['pos'][1]
        a = self.selection_ROI.getState()['size'][0]
        b = self.selection_ROI.getState()['size'][1]
        angle = self.selection_ROI.getState()['angle']
        
        rect_params = transform_rectangle(x0, y0, a, b, angle*np.pi/180)
        full_mask = InRectangle(x, y, rect_params)
        
        self.selection_mask_temp[np.where(full_mask)] = True
        self.selection_mask_temp[np.where( np.logical_not(full_mask) )] = False
        
        
        for qtItem in np.array(self.cat_instance.qtItems)[~self.cat_instance.selection_mask] :
            qtItem.setPen( pg.mkPen(qtItem.initial_color[:3] + qtItem.initial_color[-1:], width=qtItem.linewidth) )
            qtItem.setBrush( pg.mkBrush(qtItem.initial_color[:4]) )
        
        for qtItem in np.array(self.cat_instance.qtItems)[self.selection_mask_temp | self.cat_instance.selection_mask] :
            qtItem.setPen( pg.mkPen(self.selection_color[:3] + self.selection_color[-1:], width=qtItem.linewidth) )
            qtItem.setBrush( pg.mkBrush(self.selection_color[:4]) )
        
        #for i in np.where(self.selection_mask_temp)[0] :
        #    self.cat_instance.qtItems[i].setPen( pg.mkPen(self.selection_color + [255]) )
        #    self.cat_instance.qtItems[i].setBrush( pg.mkBrush(self.selection_color + [127]) )
    

class SelectSources_KeyPressFilter(QObject) :
    def __init__(self, SelectSources_instance) :
        super().__init__()
        self.SelectSources_instance = SelectSources_instance
        self.selection_ROI = SelectSources_instance.selection_ROI
        self.selection_mask = SelectSources_instance.cat_instance.selection_mask
        self.selection_mask_temp = SelectSources_instance.selection_mask_temp
        self.ImageView = SelectSources_instance.cat_instance.image.ImageView
        self.qtItems = SelectSources_instance.cat_instance.qtItems
        #self.initial_color = SelectSources_instance.initial_color
        self.selection_regions = SelectSources_instance.cat_instance.selection_regions

    def eventFilter(self, obj, event) :
        if event.type() == QEvent.KeyPress :
            key = event.key()
            if key in [Qt.Key_Enter, Qt.Key_Return, Qt.Key_Space] :
                print('Current selection confirmed')
                self.selection_mask[self.selection_mask_temp] = True
                
                x0 = self.selection_ROI.getState()['pos'][0]
                y0 = self.selection_ROI.getState()['pos'][1]
                a = self.selection_ROI.getState()['size'][0]
                b = self.selection_ROI.getState()['size'][1]
                angle = self.selection_ROI.getState()['angle']
                rect_params = transform_rectangle(x0, y0, a, b, angle*np.pi/180)
                self.selection_regions.append(rect_params)
                
            if key in [Qt.Key_Backspace, Qt.Key_Escape, Qt.Key_D] and self.selection_ROI in self.ImageView.getView().allChildren() :
                self.ImageView.removeItem(self.selection_ROI)
                for qtItem in self.qtItems[~self.selection_mask] :
                    qtItem.setPen( pg.mkPen(qtItem.initial_color[:3] + qtItem.initial_color[-1:], width=qtItem.linewidth) )
                    qtItem.setBrush( pg.mkBrush(qtItem.initial_color[:4]) )
            return True  # Event has been handled
        return False  # Pass the event to the parent









class ellipse_maker_ROI(pg.EllipseROI) :
    def __init__(self, pos, size, ImageView, QMainWindow, cat, color=[1, 1, 0]) :
        super(ellipse_maker_ROI, self).__init__(pos, size, removable=True)
        self.ImageView = ImageView
        self.color = color
        self.cat = cat
        QMainWindow.keyPressEvent = self.keyPressEvent.__get__(QMainWindow, QMainWindow)
        
    def keyPressEvent(self, event) :
        print(self.pos())
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter :
            x, y = self.pos()
            a, b = self.size()
            theta = self.angle()
            
            color=list(np.array(self.color)*255)
            
            ellipse = QGraphicsEllipseItem(x, y, a, b)
            ellipse.setTransformOriginPoint( PyQt5.QtCore.QPointF(x, y) ) 
            
            ellipse.setRotation(theta)
            ellipse.setPen( pg.mkPen(color + [255]) )
            ellipse.setBrush( pg.mkBrush(color + [127]) )
            
            self.ImageView.addItem(ellipse)
            
            size_y = self.ImageView.image.shape[0]
            y = size_y-y
            theta = -theta
            
            alpha = np.arctan(b/a)
            beta = theta*np.pi/180 - alpha
            r = (a**2 + b**2)**0.5/2
            
            x = x + r*np.cos(beta)
            y = y + r*np.sin(beta)
            
            self.cat.add_row([x, y, a, b, theta])
    
    
    
    
    
            
            