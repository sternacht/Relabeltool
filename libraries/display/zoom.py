from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

from .canvas import Canvas
from .zoomWidget import ZoomWidget

class ZoomCanvas(Canvas):

    def __init__(self, *args, **kwargs):
        super(ZoomCanvas, self).__init__(*args, **kwargs)
        self.setToolTip(self.tr("Zoom Image"))

    def wheelEvent(self, ev):
        mods = ev.modifiers()
        delta = ev.angleDelta()
        if Qt.ControlModifier == int(mods):
            self.zoomRequest.emit(delta.y(), ev.pos())
        else:
            self.scrollRequest.emit(delta.x(), Qt.Horizontal)
            self.scrollRequest.emit(delta.y(), Qt.Vertical)
        ev.accept()

class ZoomDisplay(QtWidgets.QWidget):
    edit_shape = QtCore.pyqtSignal(list)
    create_shape = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        self.zoomValue = kwargs.pop('zoom', 300)
        self.zoomRange = kwargs.pop('zoomRange', (200, 900))
        super(ZoomDisplay, self).__init__(*args, **kwargs)

        self.__parameters()
        self.__build()

    def __parameters(self):
        self.image = QtGui.QImage()

    def __build(self):
        self.zoomWidget = ZoomWidget(self.zoomValue)
        self.zoomWidget.setRange(self.zoomRange[0], self.zoomRange[1])

        ##Scroll
        scroll = QtWidgets.QScrollArea()
        self.canvas =  ZoomCanvas(epsilon= 10.0, 
                            double_click= "close",
                            num_backups = 10)
        
        self.canvas.zoomRequest.connect(self._zoom_request)
        self.canvas.scrollRequest.connect(self._scroll_request)
        self.canvas.newShape.connect(self._new_shape)
        self.canvas.shapeMoved.connect(self._set_dirty)

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
        pass

    def _zoom_request(self, delta, pos):
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
        pass

    def _scroll_request(self, delta, orientation):
        
        units = - delta / (8*15)
        bar = self.scroll_bars[orientation]
        value = bar.setValue(int(bar.value() + bar.singleStep() * units))
        self.setScroll(orientation, value)
        
    def setScroll(self, orientation, value):
        if value is not None:
            self.scroll_bars[orientation].setValue(int(value))

    def set_zoom(self, value):
        self.zoomWidget.setValue(value)

    def add_zoom(self, increment=10):
        self.set_zoom(self.zoomWidget.value() + increment)
        self.update()
        self.zoomValue = self.zoomWidget.value()

    def adjust_scale(self, initial=False):
        value = self.zoomValue / 100
        self.zoomWidget.setValue(int(100 * value))

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull():
            self.adjust_scale()
        super(ZoomDisplay, self).resizeEvent(event)

    def paint_canvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.label_font_size = int(0.02 * max(self.image.width(), self.image.height()))
        self.canvas.adjustSize()
        self.canvas.update()

    def _set_dirty(self):
        self.edit_shape.emit(self.canvas.shapes)

    def toggle_drawing_sensitive(self, drawing=True):
        if not drawing:
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()

    def load_pixmap(self, image:QtGui.QImage):
        self.canvas.setEnabled(False)
        if image.isNull():
            print("Error Data Empty")
            QtWidgets.QMessageBox.critical(self, u'Error opening file',
                                    '<p><b>%s</b></p>%s' % (u'Error opening file', u"<p>Make sure <i></i> is a valid image file."))
            return
        self.image = image
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        self.canvas.setEnabled(True)
        self.adjust_scale(True)
        self.paint_canvas()
        self.canvas.setFocus()
        return

    def _new_shape(self):
        flags = None
        group_id = None
        text = "nodule"
        if text is not None:
            generate_color = QtGui.QColor(54,156,223,150)
            shape = self.canvas.setLastLabel(text, flags, generate_color, generate_color)
            shape.group_id = group_id
            self.canvas.setEditing(False)
            self.create_shape.emit(self.canvas.shapes)
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def update_shape(self, shapes:list, replace = True):
        self.canvas.loadShapes(shapes, replace= replace)
        self.update()

    def shapes(self):
        return self.canvas.shapes

