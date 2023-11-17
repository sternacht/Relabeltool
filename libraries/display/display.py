from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from .canvas import Canvas
from .zoomWidget import ZoomWidget

class Display(QtWidgets.QWidget):
    edit_shape = QtCore.pyqtSignal(list)
    wheel_up = QtCore.pyqtSignal()
    wheel_down = QtCore.pyqtSignal()
    
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))
    def __init__(self, *args, **kwargs):
        super(Display, self).__init__(*args, **kwargs)
        self.__parameters()
        self.__initUI()           
    
    def open_next(self):
        self.wheel_down.emit()
    def open_prev(self):
        self.wheel_up.emit()
    
    def __parameters(self):
        self.image = QtGui.QImage()
        self.zoom_mode = self.FIT_WINDOW
        self.scalers = {
            self.FIT_WINDOW: self.scale_fit_window,
            self.FIT_WIDTH: self.scale_fit_width,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }
        self.zoomValue = 100
        pass

    def __initUI(self):
        # Zoom Widget
        self.zoomWidget = ZoomWidget()

        #Scroll 
        scroll = QtWidgets.QScrollArea()
        self.canvas = Canvas(epsilon= 10.0, 
                            double_click= "close",
                            num_backups = 10)
        self.canvas.wheel_up.connect(self.open_prev)
        self.canvas.wheel_down.connect(self.open_next)
        self.canvas.zoomRequest.connect(self.zoom_request)
        # self.canvas.newShape.connect(self.new_shape)
        self.canvas.anyShapeChanged.connect(self._set_dirty)
        # self.canvas.drawingPolygon.connect(self.toggle_drawing_sensitive)
        self.canvas.scrollRequest.connect(self.scroll_request)

        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scroll_bars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }

        self.scrollArea = scroll

        self.zoomWidget.setEnabled(True)
        self.zoomWidget.valueChanged.connect(self.paint_canvas)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.scrollArea)
        self.setLayout(layout)
        # self.setBackgroundRole(QtGui.QColor('black'))

    def zoom_request(self, delta, pos):
        canvas_with_old = self.canvas.width()
        units = 10
        if delta < 0:
            units = -8.5

        self.add_zoom(units)
        canvas_with_new = self.canvas.width()

        if canvas_with_old != canvas_with_new:
            canvas_scale_factor = canvas_with_new / canvas_with_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scroll_bars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scroll_bars[Qt.Vertical].value() + y_shift,
            )
    
    def scroll_request(self, delta, orientation):
        
        units = - delta / (8*15)
        bar = self.scroll_bars[orientation]
        value = bar.setValue(bar.value() + bar.singleStep() * units)
        self.setScroll(orientation, value)
        
    def setScroll(self, orientation, value):
        if value is not None:
            self.scroll_bars[orientation].setValue(value)

    def set_zoom(self, value):
        self.zoomWidget.setValue(value)

    def add_zoom(self, increment=10):
        self.set_zoom(self.zoomWidget.value() + increment)
        self.update()

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoom_mode]()
        self.zoomWidget.setValue(int(100 * value))

    def scale_fit_window(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.scrollArea.width() - e
        h1 = self.scrollArea.height() - e
        
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2

        return w1 / w2 if a2 >= a1 else h1 / h2

    def scale_fit_width(self):
        # The epsilon does not seem to work too well here.
        w = self.scrollArea.width() - 2.0
        return w / self.canvas.pixmap.width()

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() and self.zoom_mode != self.MANUAL_ZOOM:
            self.adjust_scale()
        super(Display, self).resizeEvent(event)

    def paint_canvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.label_font_size = int(0.02 * max(self.image.width(), self.image.height()))
        self.canvas.adjustSize()
        self.canvas.update()

    def load_image(self, imageData):
        self.canvas.setEnabled(False)
        if imageData is None:
            return False, QtGui.QImage()

        image = QtGui.QImage(imageData.data, 
                            imageData.shape[1], 
                            imageData.shape[0], 
                            imageData.shape[1]*3, 
                            QtGui.QImage.Format_RGB888)
        
        if image.isNull():
            print("Error Data Input")
            QtWidgets.QMessageBox.critical(self, u'Error opening file',
                                    '<p><b>%s</b></p>%s' % (u'Error opening file', u"<p>Make sure <i></i> is a valid image file."))
            return False
        self.image = image
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        self.canvas.setEnabled(True)
        self.adjust_scale(initial= True)
        self.paint_canvas()
        self.canvas.setFocus(True)
        return True, image

    def update_shape(self, shapes:list, replace= True):
        self.canvas.loadShapes(shapes, replace)
        self.update()

    def shapes(self):
        return self.canvas.shapes

    def _set_dirty(self):
        self.edit_shape.emit(self.canvas.shapes)

