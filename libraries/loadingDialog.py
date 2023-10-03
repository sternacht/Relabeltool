from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QProgressBar 
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout

# Libraries

class loadingDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()


    def setupUi(self):
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint| Qt.WindowSystemMenuHint & ~Qt.WindowCloseButtonHint)
        self.setWindowTitle("Loading ...")
        
        self.anime = QMovie("./sources/Loading_3.gif")
        self.anime.setCacheMode(QMovie.CacheAll)
        self.anime.setSpeed(90)
        self.anime.setScaledSize(QtCore.QSize(50,50))

        self.label_animation = QLabel(self)
        self.label_animation.setMovie(self.anime)

        self.notify = QLabel("Loading")
        self.notify.setWordWrap(True)
        hBoxLayout = QHBoxLayout(self)
        hBoxLayout.addWidget(self.label_animation)
        hBoxLayout.addWidget(self.notify)
        self.setLayout(hBoxLayout)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)

    def setText(self, text):
        self.notify.setText(text)

    def startAnimation(self):
        self.anime.start()
        self.show()
        
    def stopAnimation(self):
        self.anime.stop()
        self.close()

class ProgressDialog(QDialog):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        # self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint| Qt.WindowSystemMenuHint & ~Qt.WindowCloseButtonHint)
        self.setWindowTitle("Processing...")

        self.pbar = QProgressBar(self)
        self.pbar.setMaximum(100)

        self.title = QLabel(self)
        self.title.setWordWrap(True)
        vBoxLayout = QVBoxLayout(self)
        vBoxLayout.addWidget(self.title)
        vBoxLayout.addWidget(self.pbar)
        self.setLayout(vBoxLayout)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)

    def setText(self, title):
        self.title.setText(title)
    
    def setValue(self, value):
        self.pbar.setValue(value)


