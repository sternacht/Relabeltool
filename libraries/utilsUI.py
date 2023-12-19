import sys
import os
import hashlib
try:
    from libraries.ustr import ustr
except:
    from ustr import ustr
from math import sqrt
import numpy as np
import re

from PyQt5.QtGui import QPixmap
try: 
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    QT5 = True
except ImportError:
    if sys.version_info.major >=3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    QT5= False

def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())

def distancetoline(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    if np.linalg.norm(p2 - p1) == 0:
        return 0
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

def new_icon(icon):
    return QIcon("./sources/" + icon)
    # icon = QIcon("./icon/" + icon)
    # icon.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    # return icon

def new_button(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b

def new_action(parent, text, slot=None, shortcut=None, icon=None,
               tip=None, checkable=False, enabled=True, trigger= False):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a

def add_actions(widget, actions):
    for action in actions:
        if action is None: 
            widget.addSeparator()
        # elif isinstance(action, QWidget):
        #     widget.addWidget(action)
            # print("add Widget")
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)
            # print("add action")

def new_label_image(image_name:str, parent = None, folder_defaut="sources", h = 50):
    string = os.path.join(folder_defaut, image_name)
    label = QLabel(parent)
    pixmap = QPixmap(string)
    label.setPixmap(pixmap.scaled(h, h, aspectRatioMode= Qt.KeepAspectRatio, 
                    transformMode=Qt.TransformationMode.FastTransformation))
    # label.setScaledContents(True)
    if parent is None:
        return label

class struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def format_shortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)

def generate_color_by_text(text):
    s = ustr(text)
    hash_code = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return QColor(r, g, b, 100)

def have_qstring():
    """p3/qt5 get rid of QString wrapper as py3 has native unicode str type"""
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))

def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)

# QT4 has a trimmed method, in QT5 this is called strip
if QT5:
    def trimmed(text):
        return text.strip()
else:
    def trimmed(text):
        return text.trimmed()



