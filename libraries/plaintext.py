
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

class PlainText(QWidget):
    submit_signal = QtCore.pyqtSignal(str)
    def __init__(self, *args, **kwargs) -> None:
        button_height = kwargs.pop("buttonHeight", 30)
        super().__init__(*args, **kwargs)
        
        self.pathology_textEdit = QtWidgets.QPlainTextEdit()
        self.pathology_textEdit.setPlaceholderText("Descrisbe the pathology")
        self.pathology_textEdit.setFont(QtGui.QFont("serif", 13))
        self.pathology_textEdit.canPaste()

        self.submit = QtWidgets.QPushButton("Submit")
        self.submit.setEnabled(False)
        self.submit.setFixedHeight(int(button_height))
        submit_layout = QtWidgets.QHBoxLayout()
        submit_layout.setContentsMargins(0,0,0,0)
        submit_layout.addStretch(1)
        submit_layout.addWidget(self.submit)
        plaintext_layout = QtWidgets.QVBoxLayout()
        plaintext_layout.setContentsMargins(0,0,0,0)
        plaintext_layout.addLayout(submit_layout)
        plaintext_layout.addWidget(self.pathology_textEdit, 1)
        

        self.setLayout(plaintext_layout)
        self.pathology_textEdit.textChanged.connect(self.checkText)
        self.pathology_textEdit.cursorPositionChanged.connect(self.cursorChange)
        self.submit.clicked.connect(self.getText)

        self.content = dict()
        self.save_flags = False

    def checkText(self):
        text = self.pathology_textEdit.document().toPlainText()
        if len(text):
            self.submit.setEnabled(True)
            self.save_flags = True
        else:
            self.submit.setEnabled(False)
        pass

    def setText(self, content:dict = None):
        self.pathology_textEdit.clear()
        if content is None or not len(content):
            return
        self.content = content
        description = content.get("description", None)
        if description is None:
            return
        self.pathology_textEdit.setPlainText(description)
        self.save_flags = False

    def getText(self):
        text = self.pathology_textEdit.document().toPlainText()
        self.submit_signal.emit(text)
        self.submit.setEnabled(False)
        self.pathology_textEdit.setReadOnly(True)
        self.save_flags = False

    def reset_status(self):
        self.pathology_textEdit.clear()
        self.submit.setEnabled(False)

    def cursorChange(self):
        self.pathology_textEdit.setReadOnly(False)

    


# class MyWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setGeometry(300, 200 ,1200, 800)
#         self.setWindowTitle('Test')
#         # self.Param()
#         self.initUI()
#         # self.buildUI()

#     def initUI(self):
#         self.plain_text = PlainText()
#         central_layout = QtWidgets.QHBoxLayout(self)
#         central_layout.addWidget(self.plain_text)

#         self.plain_text.setText("dhvsdhlsd cewhkwejkvdv cwhefkhekfsd vdveojveovjdvsdvldsjvldsjvsdkjvlskdhverhvjsdnva;ndhvabdfbvn/;dnvsjdbvvb;sbn.sndad;bfnbadba/dbandbda\
#             dhvsdhlsd cewhkwejkvdv cwhefkhekfsd vdveojveovjdvsdvldsjvldsjvsdkjvlskdhverhvjsdnva;ndhvabdfbvn/;dnvsjdbvvb;sbn.sndad;bfnbadba/dbandbda\
#                 dhvsdhlsd cewhkwejkvdv cwhefkhekfsd vdveojveovjdvsdvldsjvldsjvsdkjvlskdhverhvjsdnva;ndhvabdfbvn/;dnvsjdbvvb;sbn.sndad;bfnbadba/dbandbda\
#                     dhvsdhlsd cewhkwejkvdv cwhefkhekfsd vdveojveovjdvsdvldsjvldsjvsdkjvlskdhverhvjsdnva;ndhvabdfbvn/;dnvsjdbvvb;sbn.sndad;bfnbadba/dbandbda\
#                         dhvsdhlsd cewhkwejkvdv cwhefkhekfsd vdveojveovjdvsdvldsjvldsjvsdkjvlskdhverhvjsdnva;ndhvabdfbvn/;dnvsjdbvvb;sbn.sndad;bfnbadba/dbandbda")
#         self.plain_text.submit_signal.connect(lambda string: print(string))
#         # self.tableDict._addData(dataset)
#         self.show()


# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     ex = MyWindow()
#     sys.exit(app.exec_())