from PyQt5 import QtGui
from PyQt5 import QtWidgets

class FocusLine(QtWidgets.QWidget):
    def __init__(self):
        self.pen = QtGui.QPen(QtGui.QColor(255, 215, 0, 255))
        self.pen.setWidth(0)

    def paint(self, painter:QtGui.QPainter, center, border_width):
        painter.setPen(self.pen)
        if center is not None:
            x, y = int(center[0]), int(center[1])
            xbw, ybw = border_width
            painter.drawLine(0, y, max(x-xbw, 0), y)
            painter.drawLine(x, 0, x, max(y-ybw, 0))
            painter.drawLine(min(x+xbw, 511), y, 511, y)
            painter.drawLine(x, min(y+ybw, 511), x, 511)