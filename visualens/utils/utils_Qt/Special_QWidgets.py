from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QTimer, QPointF
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from .utils_general import make_handles, transform_ROI_params
from .sliders import TripleSlider




class DragWidget(QWidget):
    """
    This class implements the option to create a ROI anywhere by just pressing SHIFT and clicking.
    """
    def __init__(self, ImageView):
        super().__init__()
        self.initUI()
        self.ImageView = ImageView
        self.ImageView.scene.sigMouseMoved.connect(self.mouse_moved)
        
        layout = QHBoxLayout(self) #These lines are necessary for the image to actually display in the QMainWindow later
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.ImageView)
        self.setLayout(layout)
        
        self.cat = None
        self.drawing = False
        self.current_ROI = pg.RectROI([-100, -100], [0, 0], pen='r', invertible=True) #Try initializing with a None
        
        self._floating_widgets = []  # list of (widget, corner, margin)
        
    def initUI(self):
        self.timer = QTimer()
        self.timer.setInterval(1)  # Check every 10 ms
        self.timer.timeout.connect(self.checkLongPress)
    
    def add_floating_widget(self, widget, corner='top-left', margin=8, x_margin=None, y_margin=None):
        """
        Parent a widget to this DragWidget and keep it pinned to one of its
        corners (floating above the ImageView without taking any layout space),
        repositioning it automatically whenever the widget is resized.
        """
        if x_margin is None :
            x_margin = margin
        if y_margin is None :
            y_margin = margin
        widget.setParent(self)
        # Reparenting onto an already-visible widget does NOT make the child
        # visible automatically in Qt (only the *first* show() of a brand new
        # top-level window cascades visibility down to its children) - an
        # explicit show() is required here regardless of whether `self` was
        # already on screen when this widget was added.
        widget.show()
        widget.raise_()
        self._floating_widgets.append((widget, corner, x_margin, y_margin))
        self._reposition_floating_widgets()
        return widget

    def remove_floating_widget(self, widget):
        """Stop tracking `widget` for auto-repositioning (e.g. before deleting it).

        Without this, a deleted (deleteLater()'d) widget would stay in
        `_floating_widgets` and the next resizeEvent would try to `.move()` an
        already-destroyed C/C++ QWidget, raising a RuntimeError.
        """
        self._floating_widgets = [entry for entry in self._floating_widgets if entry[0] is not widget]

    def _reposition_floating_widgets(self):
        still_alive = []
        for widget, corner, x_margin, y_margin in self._floating_widgets:
            try :
                x = x_margin if 'left' in corner else self.width() - widget.width() - x_margin
                y = y_margin if 'top' in corner else self.height() - widget.height() - y_margin
                widget.move(x, y)
            except RuntimeError :
                # The underlying C/C++ widget was already deleted; drop it.
                continue
            still_alive.append((widget, corner, x_margin, y_margin))
        self._floating_widgets = still_alive
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_floating_widgets()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ShiftModifier :
            try:
                self.ImageView.removeItem(self.current_ROI)
            except Exception:
                pass
            self.ImageView.view.setMouseEnabled(x=False, y=False)
            self.timer.start()
            ###################################################################
            self.start_pos = self.ImageView.view.mapToView(event.pos() + QPointF(-13, -14)) #the QPointF(-13, -14) offset corrects a bug where the anchor point would move.
            self.current_ROI = pg.RectROI([self.start_pos.x(), self.start_pos.y()], [0, 0], pen='r', invertible=True)
            for handle in self.current_ROI.handles :
                self.current_ROI.removeHandle(handle['item'])
            self.ImageView.addItem(self.current_ROI)
            self.drawing = True
    
    def keyPressEvent(self, event) :
        if event.key() == Qt.Key_Escape :
            try:
                self.ImageView.removeItem(self.current_ROI)
            except Exception:
                pass
    
    def checkLongPress(self):
        if not (QApplication.mouseButtons() & Qt.LeftButton) :
            self.drawing = False
            if self.timer.isActive() :
                self.timer.stop()
            make_handles(self.current_ROI)
            self.ImageView.view.setMouseEnabled(x=True, y=True)
            if self.cat is not None :
                self.cat.make_selection_ROI()
            
    def mouse_moved(self, pos):
        if self.drawing and self.current_ROI is not None:
            current_pos = self.ImageView.view.mapToView(pos)
            width = current_pos.x() - self.start_pos.x()
            height = current_pos.y() - self.start_pos.y()
            self.current_ROI.setSize([width, height])
            

class DragPlotWidget(pg.PlotWidget):
    """
    Similar to DragWidget but for pyqtgraph's PlotWidget.
    Let's the user create two types of selectable ROI.
    """
    def __init__(self):
        super().__init__()
        self.scene().sigMouseMoved.connect(self.mouse_moved)
        self.view = self.getViewBox()
        self.view.setAspectLocked(True)
        self.initUI()
        self.last_roi = None
        self.roi_list = []
        self.drawing1 = False
        self.drawing2 = False
        
    def initUI(self):
        self.timer = QTimer()
        self.timer.setInterval(1) #Check every 10 ms
        self.timer.timeout.connect(self.checkLongPress)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ShiftModifier:
            self.view.setMouseEnabled(x=False, y=False)
            self.timer.start()
            mouse_point = self.view.mapSceneToView(event.pos()) # Convert click position to data coordinates
            self.start_pos = QPointF(mouse_point.x(), mouse_point.y())
            self.last_roi = SelectableRectangleROI([self.start_pos.x(), self.start_pos.y()], [1e-9, 1e-9], pen='r', invertible=True)
            self.roi_list.append(self.last_roi)
            for handle in self.last_roi.handles:
                self.last_roi.removeHandle(handle['item'])
            self.addItem(self.last_roi)
            self.drawing1 = True
        elif event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            self.view.setMouseEnabled(x=False, y=False)
            self.timer.start()
            mouse_point = self.view.mapSceneToView(event.pos()) # Convert click position to data coordinates
            self.start_pos = QPointF(mouse_point.x(), mouse_point.y())
            self.last_roi = SelectableEllipseROI([self.start_pos.x(), self.start_pos.y()], radius=1e-9, pen='g', invertible=True)
            self.roi_list.append(self.last_roi)
            for handle in self.last_roi.handles:
                self.last_roi.removeHandle(handle['item'])
            self.addItem(self.last_roi)
            self.drawing2 = True
        else:
            super().mousePressEvent(event) #Default PlotWidget behavior, so that the user is still able to move around by click & drag
    
    def remove_all_ROIs(self):
        for roi in self.roi_list :
            self.removeItem(roi)
            del roi
        self.last_roi = None
        self.roi_list = []
    
    def remove_selected_ROIs(self):
        roi_list_new = []
        for roi in self.roi_list :
            if roi.selected :
                self.removeItem(roi)
                del roi
            else :
                roi_list_new.append(roi)
        self.roi_list = roi_list_new
        if len(self.roi_list)>0:
            self.last_roi = self.roi_list[-1]

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.remove_all_ROIs()
            event.accept()
        elif event.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete) :
            self.remove_selected_ROIs()
            event.accept()
            
    def checkLongPress(self):
        if not (QApplication.mouseButtons() & Qt.LeftButton) and not (QApplication.mouseButtons() & Qt.ControlModifier):
            self.drawing1 = False
            self.drawing2 = False
            if self.timer.isActive():
                self.timer.stop()
            if len(self.roi_list)>0:
                make_handles(self.last_roi)
            self.view.setMouseEnabled(x=True, y=True)

    def mouse_moved(self, pos):
        if self.drawing1 or self.drawing2:
            mouse_point = self.view.mapSceneToView(pos)
            width = max( mouse_point.x() - self.start_pos.x(), 1e-9, key=abs )
            height = max( mouse_point.y() - self.start_pos.y(), 1e-9, key=abs )
            self.last_roi.setSize([width, height])




class ThrottledPlotWidget(pg.PlotWidget) :
    """
    Touchpad scrolls to zoom in/out trigger too many heavy repaints and can overload pyQt and crash the pyQt app (not an issue when using discrete mouse scrolls).
    This class fixes the issue by overwritting the native scroll behaviour.
    """
    def __init__(self, throttle_mode=0, min_scroll_increment=0.) :
        super().__init__()
        self._throttle_mode = throttle_mode
        if throttle_mode==1 :
            self._update_rate = 20 # ms (~50 Hz)
            self._min_scroll_increment = min_scroll_increment if min_scroll_increment!=0. else 0.
        if throttle_mode==2 :
            self._update_rate = 20 # ms (~25 Hz)
            self._min_scroll_increment = min_scroll_increment if min_scroll_increment!=0. else 0.5
        if throttle_mode==3 :
            self._update_rate = 40 # ms (~25 Hz)
            self._min_scroll_increment = min_scroll_increment if min_scroll_increment!=0. else 1.
        if throttle_mode!=0 :
            self._wheel_delta = 0
            self._wheel_timer = QtCore.QTimer(self)
            self._wheel_timer.setInterval(self._update_rate)
            self._wheel_timer.timeout.connect(self._applyWheelZoom)

    def wheelEvent(self, ev) :
        if not self._throttle_mode :
            super().wheelEvent(ev)
        else :
            delta = ev.angleDelta().y() or ev.pixelDelta().y()
            self._wheel_delta += delta
            self._mouse_pos = ev.position()
            if not self._wheel_timer.isActive() :
                self._wheel_timer.start()
                self._stop_timer = False
            ev.accept()

    def _applyWheelZoom(self) :
        if self._throttle_mode :
            if self._stop_timer:
                self._wheel_timer.stop()
                return
            
            # Convert accumulated delta to a zoom factor
            steps = self._wheel_delta / 60.0 #120.0
            if abs(steps)>self._min_scroll_increment : # _min_scroll_increment = 0.5 to 1. reduces number of repaints while conserving a decent zooming experience
                scale = 1.15**steps
                
                # to zoom in around where the mouse is pointing
                vb = self.getViewBox()
                mouse_point = vb.mapSceneToView(self._mouse_pos)
                vb.scaleBy((1 / scale, 1 / scale), center=mouse_point)
                #self.getViewBox().scaleBy((1 / scale, 1 / scale))
                
                self._wheel_delta = 0
                
            self._stop_timer = True


class PlotWidget_xyZooming(pg.PlotWidget) :
    """
    PlotWidget where the mouse wheel zooms only the y-axis, and Shift + mouse wheel zooms only the x-axis.
    """
    def __init__(self, *args, zoom_factor=1.15, **kwargs) :
        super().__init__(*args, **kwargs)
        self._zoom_factor = zoom_factor

        self.x_log_button = QPushButton("x log scale", self)
        self.y_log_button = QPushButton("y log scale", self)
        for button in (self.x_log_button, self.y_log_button) :
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(22)
            button.toggled.connect(self._update_log_mode)
            button.show()
            button.raise_()
        self._reposition_log_buttons()

    def _update_log_mode(self, *_) :
        self.getPlotItem().setLogMode(x=self.x_log_button.isChecked(), y=self.y_log_button.isChecked())
        self.getViewBox().autoRange()

    def set_log_mode(self, x=None, y=None) :
        """Programmatically set log scale on the x and/or y axis (None leaves an axis unchanged)."""
        if x is not None :
            self.x_log_button.setChecked(bool(x))
        if y is not None :
            self.y_log_button.setChecked(bool(y))

    def _reposition_log_buttons(self) :
        if 'y_log_button' not in self.__dict__ :
            return
        margin = 6
        x = margin
        for button in (self.x_log_button, self.y_log_button) :
            button.adjustSize()
            button.move(x, margin)
            button.raise_()
            x += button.width() + margin

    def resizeEvent(self, ev) :
        if ev is not None :
            super().resizeEvent(ev)
        self._reposition_log_buttons()

    def wheelEvent(self, ev) :
        # Qt may report the wheel delta on the x component when Shift is held (e.g. on macOS)
        angle = ev.angleDelta()
        pixel = ev.pixelDelta()
        delta = angle.y() or angle.x() or pixel.y() or pixel.x()
        if delta == 0 :
            ev.ignore()
            return

        scale = self._zoom_factor**(delta / 120.)
        shift_held = bool(ev.modifiers() & Qt.ShiftModifier)
        factors = (1 / scale, 1.) if shift_held else (1., 1 / scale)

        vb = self.getViewBox()
        center = vb.mapSceneToView(ev.position())
        vb.scaleBy(factors, center=center)
        ev.accept()


class DragPlotWidget_special(ThrottledPlotWidget):
    """
    Same as DragPlotWidget but with selecatble elliptical and two different types of circular ROI.
    """
    def __init__(self, ROI_param_dict=None, slider_init_dict=None, size_sliders_scaling=1., throttle_mode=0) :
        super().__init__(throttle_mode=throttle_mode)
        self.scene().sigMouseMoved.connect(self.mouse_moved)
        self.view = self.getViewBox()
        self.view.setAspectLocked(True)
        self.initUI()
        self.last_roi = None
        self.roi_list = []
        self.drawing1 = False
        self.drawing2 = False
        self.drawing3 = False
        
        self.ROI_param_dict = ROI_param_dict if ROI_param_dict is not None else {'CIRCLE1': 'CIRCLE1', 'CIRCLE2': 'CIRCLE2', 'ELLIPSE': 'ELLIPSE', 'POINT': 'POINT'}
        self.slider_init_dict = slider_init_dict if slider_init_dict is not None else {'param1': [1e-3, 1000, 'log', 'triple_handle'], 'param2': [0, 10, 'linear', 'single_handle']}
        
        # Setting size_sliders_scaling is necessary in case you want the slider units to be different from the ViewBox units.
        self.size_sliders_scaling = size_sliders_scaling
        
        # This line because the many paint events can crash pyQt
        #self._run_low_performance_settings()
        axis = self.getAxis('left')
        axis.setWidth(0)
        
    def initUI(self):
        self.timer = QTimer()
        self.timer.setInterval(16) #Check every 16 ms
        self.timer.timeout.connect(self.checkLongPress)

    def mousePressEvent(self, event):
        # CIRCLE2
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.AltModifier) and (event.modifiers() & Qt.ControlModifier or event.modifiers() & Qt.MetaModifier) :
            self.view.setMouseEnabled(x=False, y=False)
            self.timer.start()
            mouse_point = self.view.mapSceneToView(event.pos()) # Convert click position to data coordinates
            self.start_pos = QPointF(mouse_point.x(), mouse_point.y())
            self.last_roi = SelectableCircleROI([self.start_pos.x(), self.start_pos.y()], radius=1e-9, pen='purple', invertible=True)
            self.last_roi.type_general = 'CIRCLE2'
            self.last_roi.type = self.ROI_param_dict['CIRCLE2'][0]
            self.roi_list.append(self.last_roi)
            for handle in self.last_roi.handles:
                self.last_roi.removeHandle(handle['item'])
            self.addItem(self.last_roi)
            self.drawing3 = True
            
            # Add sliders to control light source parameters
            params = self.ROI_param_dict['CIRCLE2'][1]
            attach_sliders(self, params, size_sliders_scaling=self.size_sliders_scaling)
        # POINT
        elif event.button() == Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier) and (event.modifiers() & Qt.ControlModifier or event.modifiers() & Qt.MetaModifier) :
            self.view.setMouseEnabled(x=False, y=False)
            self.timer.start()
            mouse_point = self.view.mapSceneToView(event.pos()) # Convert click position to data coordinates
            self.start_pos = QPointF(mouse_point.x(), mouse_point.y())
            self.last_roi = SelectableCircleROI([self.start_pos.x(), self.start_pos.y()], radius=1e-3, pen='cyan', invertible=True)
            self.last_roi.type_general = 'POINT'
            self.last_roi.type = self.ROI_param_dict['POINT'][0]
            self.roi_list.append(self.last_roi)
            for handle in self.last_roi.handles:
                self.last_roi.removeHandle(handle['item'])
            self.addItem(self.last_roi)
            #self.drawing4 = True
            
            # Add sliders to control light source parameters
            params = self.ROI_param_dict['POINT'][1]
            attach_sliders(self, params, size_sliders_scaling=self.size_sliders_scaling)        
        #
        elif event.button() == Qt.LeftButton and event.modifiers() == Qt.ShiftModifier :
            print('Shift modifier function not yet implemented')
        # ELLIPSE
        elif event.button() == Qt.LeftButton and event.modifiers() == Qt.AltModifier :
            self.view.setMouseEnabled(x=False, y=False)
            self.timer.start()
            mouse_point = self.view.mapSceneToView(event.pos()) # Convert click position to data coordinates
            self.start_pos = QPointF(mouse_point.x(), mouse_point.y())
            self.last_roi = SelectableEllipseROI([self.start_pos.x(), self.start_pos.y()], [1e-9, 1e-9], pen='orange', invertible=True)
            self.last_roi.type_general = 'ELLIPSE'
            self.last_roi.type = self.ROI_param_dict['ELLIPSE'][0]
            self.roi_list.append(self.last_roi)
            for handle in self.last_roi.handles:
                self.last_roi.removeHandle(handle['item'])
            self.addItem(self.last_roi)
            self.drawing1 = True
            
            # Add sliders to control light source parameters
            params = self.ROI_param_dict['ELLIPSE'][1]
            attach_sliders(self, params, size_sliders_scaling=self.size_sliders_scaling)
        # CIRCLE1
        elif event.button() == Qt.LeftButton and event.modifiers() in (Qt.ControlModifier, QtCore.Qt.MetaModifier) :
            self.view.setMouseEnabled(x=False, y=False)
            self.timer.start()
            mouse_point = self.view.mapSceneToView(event.pos()) # Convert click position to data coordinates
            self.start_pos = QPointF(mouse_point.x(), mouse_point.y())
            self.last_roi = SelectableCircleROI([self.start_pos.x(), self.start_pos.y()], radius=1e-9, pen='red', invertible=True)
            self.last_roi.type_general = 'CIRCLE1'
            self.last_roi.type = self.ROI_param_dict['CIRCLE1'][0]
            self.roi_list.append(self.last_roi)
            for handle in self.last_roi.handles:
                self.last_roi.removeHandle(handle['item'])
            self.addItem(self.last_roi)
            self.drawing2 = True
            
            # Add sliders to control light source parameters
            params = self.ROI_param_dict['CIRCLE1'][1]
            attach_sliders(self, params, size_sliders_scaling=self.size_sliders_scaling)
        else:
            super().mousePressEvent(event) #Default PlotWidget behavior, so that the user is still able to move around by click & drag

    def remove_all_ROIs(self):
        for roi in self.roi_list :
            remove_dpos_square_overlay(self, roi)
            self.removeItem(roi)
            del roi
        self.last_roi = None
        self.roi_list = []
    
    def remove_selected_ROIs(self):
        roi_list_new = []
        for roi in self.roi_list :
            if roi.selected :
                remove_dpos_square_overlay(self, roi)
                self.removeItem(roi)
                del roi
            else :
                roi_list_new.append(roi)
        self.roi_list = roi_list_new
        if len(self.roi_list)>0:
            self.last_roi = self.roi_list[-1]

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.remove_all_ROIs()
            event.accept()

        elif event.modifiers() in (Qt.ControlModifier, QtCore.Qt.MetaModifier) and event.key() == Qt.Key_D:
            # Cmd+D: hide all ROIs
            for roi in self.roi_list:
                try:
                    roi.hide_roi()
                except Exception:
                    pass
            event.accept()
        elif event.modifiers() in (Qt.ControlModifier, QtCore.Qt.MetaModifier) and event.key() == Qt.Key_S:
            # Cmd+S: show all ROIs
            for roi in self.roi_list:
                try:
                    roi.show_roi()
                except Exception:
                    pass
            event.accept()
        elif event.key() == Qt.Key_D:
            # D: hide selected ROIs
            for roi in self.roi_list:
                if getattr(roi, "selected", False):
                    try:
                        roi.hide_roi()
                    except Exception:
                        pass
            event.accept()
        elif event.key() == Qt.Key_S:
            # S: show selected ROIs
            for roi in self.roi_list:
                if getattr(roi, "selected", False):
                    try:
                        roi.show_roi()
                    except Exception:
                        pass
            event.accept()
        elif event.modifiers() in (Qt.ControlModifier, QtCore.Qt.MetaModifier) and event.key() == Qt.Key_F:
            # Cmd+F: hide all sliders
            for roi in self.roi_list:
                try:
                    show_or_hide_sliders(roi)
                except Exception:
                    pass
            event.accept()
        elif event.modifiers() in (Qt.ControlModifier, QtCore.Qt.MetaModifier) and event.key() == Qt.Key_A:
            # Cmd+A: Select all ROIs
            selection_mask = [roi.selected for roi in self.roi_list]
            selection_bool = False in selection_mask
            for roi in self.roi_list:
                try:
                    roi.selected = selection_bool
                    pen = roi.selected_pen if selection_bool else roi.normal_pen
                    roi.setPen(pen)
                except Exception:
                    pass
            event.accept()

        elif event.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete) :
            self.remove_selected_ROIs()
            event.accept()
            
    def checkLongPress(self):
        if not (QApplication.mouseButtons() & Qt.LeftButton) and not (QApplication.mouseButtons() & Qt.ControlModifier):
            self.drawing1 = False
            self.drawing2 = False
            self.drawing3 = False
            if self.timer.isActive():
                self.timer.stop()
            if len(self.roi_list)>0:
                if type(self.last_roi).__name__ == 'SelectableEllipseROI' :
                    make_handles(self.last_roi)
                elif type(self.last_roi).__name__ == 'SelectableCircleROI' and getattr(self.last_roi, "type_general", None) != 'POINT' :
                    #make_handles(self.roi2, make_rotation_handles=False, fixed_ratio=True)
                    self.last_roi.addScaleHandle([0.5+2**-1.5, 0.5+2**-1.5], [0.5, 0.5])
            self.view.setMouseEnabled(x=True, y=True)

    def mouse_moved(self, pos):
        if self.drawing1 :
            mouse_point = self.view.mapSceneToView(pos)
            width = max( mouse_point.x() - self.start_pos.x(), 1e-9, key=abs )
            height = max( mouse_point.y() - self.start_pos.y(), 1e-9, key=abs )
            self.last_roi.setSize([width, height])
        if self.drawing2 :
            mouse_point = self.view.mapSceneToView(pos)
            w2 = abs(mouse_point.x() - self.start_pos.x())
            h2 = abs(mouse_point.y() - self.start_pos.y())
            radius = max( (w2**2 + h2**2)**0.5, 1e-9 )
            new_corner_x = self.start_pos.x() - radius
            new_corner_y = self.start_pos.y() - radius
            self.last_roi.setPos([new_corner_x, new_corner_y])
            self.last_roi.setSize([2*radius, 2*radius])
        if self.drawing3 :
            mouse_point = self.view.mapSceneToView(pos)
            w2 = abs(mouse_point.x() - self.start_pos.x())
            h2 = abs(mouse_point.y() - self.start_pos.y())
            radius = max( (w2**2 + h2**2)**0.5, 1e-9 )
            new_corner_x = self.start_pos.x() - radius
            new_corner_y = self.start_pos.y() - radius
            self.last_roi.setPos([new_corner_x, new_corner_y])
            self.last_roi.setSize([2*radius, 2*radius])
    
    #def _run_low_performance_settings(self) :
    #    vb = self.getViewBox()
    #    vb.setMouseEnabled(x=True, y=True)
    #    vb.enableAutoRange(False)
    #    pg.setConfigOptions(antialias=False)
        

class DragImagePlotWidget_special(DragPlotWidget_special):
    """
    DragPlotWidget_special variant that also displays raster images.

    This mirrors the API needed from pg.ImageView (`setImage`, `addItem`,
    `getView`) while preserving ROI interactions from DragPlotWidget_special.
    """
    def __init__(self, ROI_param_dict=None, slider_init_dict=None, size_sliders_scaling=1., throttle_mode=0):
        super().__init__(ROI_param_dict=ROI_param_dict, slider_init_dict=slider_init_dict, size_sliders_scaling=size_sliders_scaling, throttle_mode=throttle_mode)
        self._image_item = pg.ImageItem()
        self.addItem(self._image_item)

    def setImage(self, image):
        self._image_item.setImage(image)

    def clearImage(self):
        self._image_item.clear()


class SelectableRectangleROI(pg.ROI):
    def __init__(self, pos, size, **kwargs):
        super().__init__(pos, size, **kwargs)
        self.selected = False
        self.normal_pen = self.pen #pg.mkPen('r', width=1)
        self.selected_pen = pg.mkPen('y', width=3)
        self.setPen(self.normal_pen)

    def mouseClickEvent(self, event):
        handle_mouse_click(self, event)
        #super().mouseClickEvent(event)

    def hide_roi(self) :
        hide_roi(self)

    def show_roi(self) :
        show_roi(self)

class SelectableEllipseROI(pg.EllipseROI):
    def __init__(self, pos, size, **kwargs):
        super().__init__(pos, size, **kwargs)
        self.selected = False
        self.normal_pen = self.pen #pg.mkPen('r', width=1)
        self.selected_pen = pg.mkPen('y', width=3)
        self.setPen(self.normal_pen)
        
    def mouseClickEvent(self, event):
        handle_mouse_click(self, event)
        #super().mouseClickEvent(event)

    def hide_roi(self) :
        hide_roi(self)

    def show_roi(self) :
        show_roi(self)

class SelectableCircleROI(pg.CircleROI):
    def __init__(self, pos, **kwargs):
        super().__init__(pos, **kwargs)
        self.selected = False
        self.normal_pen = self.pen #pg.mkPen('g', width=1)
        self.selected_pen = pg.mkPen('y', width=3)
        self.setPen(self.normal_pen)
        
    def mouseClickEvent(self, event):
        handle_mouse_click(self, event)
        #super().mouseClickEvent(event)

    def hide_roi(self) :
        hide_roi(self)

    def show_roi(self) :
        show_roi(self)
    
    # Functions from old implementation. Keep here just in case.
    #def _updateLabel(self, vmin, vmid, vmax):
    #    self.slider_label.setText(f"n_Sersic: [{vmin:.2f}, {vmid:.2f}, {vmax:.2f}]")

    #def _updatePanelPos(self):
    #    """
    #    Position floating widget BELOW the ROI, but keep it horizontal.
    #    """
    #    br = self.boundingRect()
    #    local_anchor = QPointF(br.left(), br.top())
    #    scene_pos = self.pos() + local_anchor
    #    self.proxy.setPos(scene_pos)

def _roi_link_box_slider(roi):
    """
    Return ``(slider_key, slider)`` for the search-range box (``roi_link_mode == 'link_box'``),
    or legacy ``('dpos', slider)`` if present. Otherwise ``(None, None)``.
    """
    sliders = getattr(roi, "sliders", None)
    if not isinstance(sliders, dict):
        return None, None
    for name, s in sliders.items():
        if getattr(s, "roi_link_mode", None) == "link_box":
            return name, s
    if "dpos" in sliders:
        return "dpos", sliders["dpos"]
    return None, None


def _update_dpos_square_overlay(plot_widget, roi):
    """Axis-aligned dashed square centered on ROI; half-side = |vmid| of the ``link_box`` slider."""
    item = getattr(roi, "_dpos_square_plot", None)
    if item is None:
        return
    if roi not in getattr(plot_widget, "roi_list", []):
        return
    if not getattr(roi, "isVisible", lambda: True)():
        item.setVisible(False)
        return
    _, sbox = _roi_link_box_slider(roi)
    if sbox is None:
        item.setVisible(False)
        return
    d = abs(float(sbox.vmid)) / sbox.roi_link_size_scale
    if d < 1e-30:
        d = 1e-30
    x_center, y_center, _, _, _ = transform_ROI_params(roi)
    xs = [x_center - d, x_center + d, x_center + d, x_center - d, x_center - d]
    ys = [y_center - d, y_center - d, y_center + d, y_center + d, y_center - d]
    item.setData(xs, ys)
    item.setVisible(True)


def attach_dpos_square_overlay(plot_widget, roi):
    """Add dashed square for the ``link_box`` search range (legacy key ``dpos`` still supported)."""
    _, sbox = _roi_link_box_slider(roi)
    if sbox is None:
        return
    remove_dpos_square_overlay(plot_widget, roi)
    pen = pg.mkPen(color=(255, 255, 255, 200), width=1, style=Qt.DashLine)
    item = pg.PlotDataItem()
    item.setPen(pen)
    plot_widget.addItem(item)
    roi._dpos_square_plot = item

    def _on_change(*_a, **_k):
        _update_dpos_square_overlay(plot_widget, roi)

    roi._dpos_square_update = _on_change
    roi._dpos_square_slider_ref = sbox
    roi.sigRegionChanged.connect(_on_change)
    sbox.valuesChanged.connect(_on_change)
    _on_change()


def remove_dpos_square_overlay(plot_widget, roi):
    item = getattr(roi, "_dpos_square_plot", None)
    if item is None:
        return
    if hasattr(roi, "_dpos_square_update"):
        try:
            roi.sigRegionChanged.disconnect(roi._dpos_square_update)
        except (TypeError, RuntimeError):
            pass
        sref = getattr(roi, "_dpos_square_slider_ref", None)
        if sref is not None:
            try:
                sref.valuesChanged.disconnect(roi._dpos_square_update)
            except (TypeError, RuntimeError):
                pass
        del roi._dpos_square_update
    if hasattr(roi, "_dpos_square_slider_ref"):
        del roi._dpos_square_slider_ref
    try:
        plot_widget.removeItem(item)
    except Exception:
        pass
    roi._dpos_square_plot = None


def attach_sliders(self, params, size_sliders_scaling=1.) :
    for i, param in enumerate(params) :
        spec = self.slider_init_dict[param]
        min_range, max_range = spec[0], spec[1]
        log_scale = spec[2] == 'log'
        single_handle = spec[3] == 'single_handle'
        roi_link_mode = spec[4] if len(spec) > 4 else None
        if not hasattr(self.last_roi, 'sliders') :
            self.last_roi.sliders = {
                param: TripleSlider(
                    min_range,
                    max_range,
                    PlotWidget=self,
                    label=param,
                    roi=self.last_roi,
                    log_scale=log_scale,
                    single_handle=single_handle,
                    roi_link_mode=roi_link_mode,
                    roi_link_size_scale=size_sliders_scaling,
                )
            }
        else :
            offset = (self.last_roi.sliders[params[i-1]].offset
                      + self.last_roi.sliders[params[i-1]].bounding_height
                      + self.last_roi.sliders[params[i-1]].offset_text)
            self.last_roi.sliders[param] = TripleSlider(
                min_range,
                max_range,
                PlotWidget=self,
                label=param,
                roi=self.last_roi,
                offset=offset,
                log_scale=log_scale,
                single_handle=single_handle,
                roi_link_mode=roi_link_mode,
                roi_link_size_scale=size_sliders_scaling,
            )
        self.last_roi.sliders[param].setParentItem(self.last_roi)

    _, sbox = _roi_link_box_slider(self.last_roi)
    if sbox is not None:
        attach_dpos_square_overlay(self, self.last_roi)

def hide_roi(self):
    """Hide ROI and any attached sliders (if present)."""
    self.setVisible(False)
    for s in getattr(self, "sliders", {}).values():
        try:
            s.hide_slider()
        except Exception:
            pass
    sq = getattr(self, "_dpos_square_plot", None)
    if sq is not None:
        sq.setVisible(False)

def show_roi(self):
    """Show ROI and any attached sliders (if present)."""
    self.setVisible(True)
    for s in getattr(self, "sliders", {}).values():
        try:
            s.show_slider()
        except Exception:
            pass
    sq = getattr(self, "_dpos_square_plot", None)
    if sq is not None:
        sq.setVisible(True)

def handle_mouse_click(self, event):
    if event.button() == QtCore.Qt.LeftButton :
        if not event.double():
            self.selected = not self.selected
            self.setPen(self.selected_pen if self.selected else self.normal_pen)
            event.accept()  # Prevent propagation
        else :
            show_or_hide_sliders(self)
            event.accept()
    else:
        super(type(self), self).mouseClickEvent(event)

def show_or_hide_sliders(self):
    sliders = getattr(self, "sliders", {})
    any_visible = any(getattr(s, "isVisible", lambda: False)() for s in sliders.values())
    for s in sliders.values():
        try:
            s.hide_slider() if any_visible else s.show_slider()
        except Exception:
            pass

