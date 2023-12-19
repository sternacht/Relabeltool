from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class tool_button(QToolButton):

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(self.__style_sheet)

    def paintEvent(self, a0: QPaintEvent):
        p = QStylePainter()
        opt = QStyleOptionToolButton()
        self.initStyleOption(opt)

        h = opt.rect.height()
        w = opt.rect.width()
        h_min = self.minimumHeight()
        w_min = self.minimumWidth()

        iconSize:int = max(min(h, w) -2 * 5, min(h_min, w_min))
        opt.iconSize = QSize(iconSize, iconSize)
        self.w = iconSize

        p.begin(self)
        p.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, opt)
        p.end()
    
    #9fe2bf
    __style_sheet = """
        QPushButton, QToolButton
        {
            color: #ffffff;
            background-color: #7fb3d5;
            border-width: 2px;
            border-color: #bbe1fa;
            border-style: outset;
            border-radius: 10px;
            padding: 5px;
            font-family: Times New Roman;
            font: 20px;
            padding-left: 5px;
            padding-right: 5px;
        }

        QPushButton::pressed, QToolButton::pressed
        {
            background-color: #559ecd;
        }

        QPushButton::checked, QToolButton::checked
        {
            background-color: #FFD700;
            border-width: 2px;
            border-color: #bbe1fa;
            border-style: outset;
            border-radius: 10px;
            padding: 5px;
            font-family: Times New Roman;
            font-size: 20px;
            padding-left: 5px;
            padding-right: 5px;
        }

        QPushButton:hover, QToolButton:hover
        {
            background-color: #3594d5;
        }
        QPushButton:disabled, QToolButton:disabled
        {
            color: #959595;
            background-color:#bbe1fa;
        }
        """


