import sys
try: 
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QWidget, QToolBar, QWidgetAction, QToolButton, QCheckBox
    from PyQt5.QtCore import Qt, QSize
except ImportError:
    if sys.version_info.major >=3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

class ToolButton(QToolButton):
    """all buttons have the same size"""
    minSize = (60,60)

    def minimumSizeHint(self):
        minimunSize = super(ToolButton, self).minimumSizeHint()
        width1, height1 = minimunSize.width(), minimunSize.height()
        width2, height2 = self.minSize
        ToolButton.minSize = max(width1, width2), max(height1, height2)
        return QSize(*ToolButton.minSize)

class Toolbar(QToolBar):

    def __init__(self, title):
        super(Toolbar, self).__init__(title)
        barLayout = self.layout()
        margins = (3,3,0,0)
        barLayout.setSpacing(5)
        barLayout.setContentsMargins(*margins)
        self.setContentsMargins(*margins)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

    def addAction(self, action):
        if isinstance(action, QWidgetAction):
            return super(Toolbar, self).addAction(action)
        button = ToolButton()
        button.setDefaultAction(action)
        button.setToolButtonStyle(self.toolButtonStyle())
        self.addWidget(button)

    # def addCheckBox(self, checkbox):
    #     if isinstance(checkbox, QCheckBox):
