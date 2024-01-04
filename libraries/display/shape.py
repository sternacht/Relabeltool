import copy
import math
from unicodedata import category

from PyQt5 import QtCore
from PyQt5 import QtGui
from math import sqrt

from numpy import zeros
from libraries.utilsUI import distance, distancetoline

import sys


# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


DEFAULT_LINE_COLOR = QtGui.QColor(52, 152, 219, 128)  # bf hovering
DEFAULT_FILL_COLOR = QtGui.QColor(52, 152, 219, 128)  # hovering
DEFAULT_SELECT_LINE_COLOR = QtGui.QColor(255, 255, 255)  # selected
DEFAULT_SELECT_FILL_COLOR = QtGui.QColor(0, 255, 0, 155)  # selected
DEFAULT_VERTEX_FILL_COLOR = QtGui.QColor(0, 255, 0, 255)  # hovering
DEFAULT_HVERTEX_FILL_COLOR = QtGui.QColor(255, 255, 255, 255)  # hovering

DEFAULT_VERTEX_FILL_COLOR_NEG = QtGui.QColor(110, 44, 0, 255) #Draw negative
DEFAULT_HVERTEX_FILL_COLOR_NEG = QtGui.QColor(110, 44, 0, 255)

# def distance(p):
#     return sqrt(p.x() * p.x() + p.y() * p.y())

def calculateDistance(pointA:QtCore.QPointF, pointB:QtCore.QPointF, spacing:tuple):
    delta = QtCore.QPointF((pointB.x() - pointA.x()) * spacing[0], (pointB.y() - pointA.y()) * spacing[1])
    distance = sqrt(delta.x() * delta.x() + delta.y() * delta.y())
    return distance


class Shape(object):

    P_SQUARE, P_ROUND = range(2)

    MOVE_VERTEX, NEAR_VERTEX = range(2)

    CREATE, EDIT = range(2)

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR

    negative_fill_color = DEFAULT_VERTEX_FILL_COLOR_NEG
    point_type = P_ROUND
    scale = 1.0
    label_font_size = 8

    _vertex_fill_color = vertex_fill_color

    def __init__(
        self,
        label=None,
        line_color=None,
        shape_type=None,
        flags=None,
        group_id=None,
        category = 0,
        conf = 1.0,
        checked =True,
        paint_label = False,
        positive=True,
        rect = None,
        mode = CREATE,
        mask = None,
        point_size = 8
    ):
        self.label = label
        self.group_id = group_id
        self.points = []
        self.fill = False
        self.selected = False
        self.shape_type = shape_type
        self.flags = flags
        self.other_data = {}
        self.point_size = point_size
        self.conf = conf
        self.paint_label = paint_label
        self.checked = checked
        self.category = category
        self.positive = positive

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (3, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False
        self.rect = rect

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

        self.shape_type = shape_type
        self.spacing = (0.6,0.6)
        self.mode = mode
        self.mask = mask

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
        ]:
            raise ValueError("Unexpected shape_type: {}".format(value))
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)

    def canAddPoint(self):
        return self.shape_type in ["polygon", "linestrip"]

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def insertPoint(self, i, point):
        self.points.insert(i, point)

    def removePoint(self, i):
        self.points.pop(i)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def getRectFromLine(self, pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

    def paint(self, painter, hide_vertex = False):
        if self.points:
            # print(self.points)
            color = (
                self.select_line_color if self.selected else self.line_color
            )
            pen = QtGui.QPen(color)
            # Try using integer sizes for smoother drawing(?)
            pen.setWidth(max(1, int(round(4.0 / self.scale))))
            painter.setPen(pen)

            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()

            polygon_rect_path = QtGui.QPainterPath()

            if self.shape_type == "rectangle":
                # assert len(self.points) in [1, 2]
                if len(self.points) == 4:
                    self.points = [self.points[0], self.points[2]]
                if len(self.points) == 2:
                    rectangle = self.getRectFromLine(*self.points)
                    line_path.addRect(rectangle)
                if not hide_vertex:
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getCircleRectFromLine(self.points)
                    line_path.addEllipse(rectangle)
                if not hide_vertex:
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    if not hide_vertex:
                        self.drawVertex(vrtx_path, i)

            elif self.shape_type == "line":
                # print(self.spacing)
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    # if not hide_vertex:
                        # self.drawVertex(vrtx_path, i)
                    self.drawCross(vrtx_path, i)
                distanceMM = calculateDistance(self.points[0], self.points[-1], self.spacing)
                if distanceMM > 1E-5:
                    min_x = sys.maxsize
                    min_y = sys.maxsize
                    min_y_label = int(1.25 * self.label_font_size)
                    
                    min_x = min(min_x, p.x())
                    min_y = min(min_y, p.y())
                    if min_x != sys.maxsize and min_y != sys.maxsize:
                        font = QtGui.QFont()
                        font.setPointSize(self.label_font_size)
                        font.setBold(True)
                        painter.setFont(font)

                        font_metric = QtGui.QFontMetrics(font)

                        # pen = painter.pen()
                        # pen.setColor = QtGui.QColor(0,0,0,255)

                        if min_y < min_y_label:
                            min_y += min_y_label
                        string = "{:.1f} mm".format(distanceMM)
                        char_width = font_metric.width(string)
                        char_height = font_metric.height()
                        # print(char_width, char_height)
                        outline = QtCore.QRect(int(min_x), int(min_y),  int(char_width),  int(char_height))
                        # painter.setPen(pen)
                        painter.fillRect(outline, QtGui.QColor(0,0,0,255))
                        pen = QtGui.QPen(QtGui.QColor(255,255,255,255))
                        painter.setPen(pen)
                        painter.drawText(outline, 1, string)
                        pen = QtGui.QPen(color)
                        painter.setPen(pen)
                    
                

            elif self.shape_type == 'point':
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    if not hide_vertex:
                        self.drawVertex(vrtx_path, i)
            else:
                line_path.moveTo(self.points[0])
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    if not hide_vertex:
                        self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self.points[0])
                # stop showing rectangle on polygon type
                # if self.isClosed():
                #     rect_outline = self.boundingRect()
                #     polygon_rect_path.addRect(rect_outline)

            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self._vertex_fill_color)
            # painter.fillPath(line_path, self._vertex_fill_color)
            if self.isClosed():
                if hide_vertex:
                    pen.setStyle(QtCore.Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawPath(polygon_rect_path)

            if self.paint_label:
                min_x = sys.maxsize
                min_y = sys.maxsize
                min_y_label = int(1.25 * self.label_font_size)
                for point in self.points:
                    min_x = min(min_x, point.x())
                    min_y = min(min_y, point.y())
                if min_x != sys.maxsize and min_y != sys.maxsize:
                    font = QtGui.QFont()
                    font.setPointSize(self.label_font_size)
                    font.setBold(True)
                    painter.setFont(font)
                    if self.label is None:
                        self.label = ""
                    if min_y < min_y_label:
                        min_y += min_y_label
                    string = f'{self.label}'
                    if self.group_id:
                        string += f':{self.group_id:3d}'
                    # painter.drawText(min_x, min_y, self.label)
                    painter.drawText(min_x, min_y, string)
            if self.fill:
                color = (
                    self.select_fill_color
                    if self.selected
                    else self.fill_color
                )
                painter.fillPath(line_path, color)
            self.setRect()

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self._vertex_fill_color = self.hvertex_fill_color
        else:
            self._vertex_fill_color = self.line_color #self.vertex_fill_color
        positive = self.positive
        if not positive:
            self._vertex_fill_color = self.negative_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def drawCross(self, path, i):
        d = self.point_size / self.scale
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        self._vertex_fill_color = QtGui.QColor('red')
        
        path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        

    def nearestVertex(self, point, epsilon):
        min_distance = float("inf")
        min_i = None
        for i, p in enumerate(self.points):
            dist = distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float("inf")
        post_i = None
        for i in range(len(self.points)):
            line = [self.points[i - 1], self.points[i]]
            dist = distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        return self.makePath().contains(point)

    def getCircleRectFromLine(self, line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, point) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QtCore.QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    def makePath(self):
        if self.shape_type == "rectangle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getRectFromLine(*self.points)
                path.addRect(rectangle)
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getCircleRectFromLine(self.points)
                path.addEllipse(rectangle)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        """Highlight a vertex appropriately based on the current action
        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        """Clear the highlighted point"""
        self._highlightIndex = None

    def setPositive(self, value = True):
        self.positive = value

    def getRect(self):
        return self.rect

    def setRect(self):
        rect = self.boundingRect()
        x1 = rect.x()
        y1 = rect.y()
        x2 = x1 + rect.width()
        y2 = y1 + rect.height()
        if rect.width() == 0.0:
            if x2 >= 511:
                x1 = 510
            else:
                x2 = x2+1
        if rect.height() == 0.0:
            if y2 >= 511:
                y1 = 510
            else:
                y2 = y2+1
        self.rect = (x1,y1,x2,y2)

    def shape2dict(self):
        shape_dict = {}
        shape_dict["label"] = self.label
        shape_dict["points"] = [(p.x(), p.y()) for p in self.points]
        shape_dict['conf'] = self.conf
        shape_dict['checked'] = self.checked
        shape_dict['category'] = self.category
        shape_dict['shape_type'] = self.shape_type
        shape_dict['group_id'] = self.group_id
        shape_dict['rect'] = self.rect
        shape_dict['mode'] = self.mode
        if self.mask is not None:
            if self.mask.max() >= 255:
                shape_dict['mask'] = (self.mask//255) * self.category if self.category > 0 else (self.mask//255) * 4
            elif self.mask.max() == 1:
                shape_dict['mask'] = (self.mask) * self.category if self.category > 0 else (self.mask) * 4
        return shape_dict

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value

    def setmode(self, mode):
        self.mode = mode