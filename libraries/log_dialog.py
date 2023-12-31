try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import sys
import os
from typing import List, Tuple, Dict
from .save.pickleData import load_dict, load
from .utilsUI import new_action

class help_dialog(QDialog):
    dict_label_signal = pyqtSignal(dict)

    def __init__(self, parent = None):
        super(help_dialog, self).__init__(parent=parent)
        self.setWindowTitle("HELP")
        self.setup_UI()
        self.setFont(QFont("Times New Roman", 15))

    def setup_UI(self):
        layout = QVBoxLayout(self)
        image_category = QLabel("CATEGORY")
        pixmap = QPixmap("./sources/category.png")
        image_category.setPixmap(pixmap)
        layout.addWidget(image_category)

        QBtn = QDialogButtonBox.Close

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox)

class ExportPatientExcelDialog(QDialog):
    update_text_signal = pyqtSignal(str)
    def __init__(self, parent = None):
        super().__init__(parent=parent)
        self.init_ui()

    def init_ui(self):
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('Select start and end date')

        layout = QVBoxLayout()

        start_label = QLabel('Start date:')
        self.start_date_edit = QDateEdit(self)
        start_date = QDate(2023, 9, 1)
        self.start_date_edit.setDate(start_date)
        self.start_date_edit.setCalendarPopup(True)
        
        end_label = QLabel('End date(included):')
        self.end_date_edit = QDateEdit(self)
        end_date = QDate.currentDate()
        self.end_date_edit.setDate(end_date)
        self.end_date_edit.setCalendarPopup(True)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(start_label)
        layout.addWidget(self.start_date_edit)
        layout.addWidget(end_label)
        layout.addWidget(self.end_date_edit)
        layout.addWidget(button_box)

        self.setLayout(layout)

class ConfirmedStatusDialog(QDialog):
    def __init__(self, confirmed_status: Dict[str, int], parent = None):
        super().__init__(parent=parent)

        self.setGeometry(100, 100, 450, 300)
        # Create table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(confirmed_status))
        header = ['User_Name', 'Confirmed_Number']
        self.tableWidget.setColumnCount(len(header))
        self.tableWidget.setHorizontalHeaderLabels(header)
        
        layout = QVBoxLayout()
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)
        self.setWindowTitle('Confirmed Status')

        self.init_table(confirmed_status)
        
    def init_table(self, confirmed_status: Dict[str, int]):
        for i, (user_name, confirmed_number) in enumerate(confirmed_status.items()):
            item = QTableWidgetItem(user_name)
            self.tableWidget.setItem(i, 0, item)
            item = QTableWidgetItem(str(confirmed_number))
            self.tableWidget.setItem(i, 1, item)
        self.tableWidget.resizeColumnsToContents()
    
class change_size(QDialog):

    def __init__(self, parent = None):
        super(change_size, self).__init__(parent=parent)
        self.parent = parent
        self.setWindowTitle("change size")
        self.selection = {'小':4,'中':6,'大':8,'特大':12}
        self.setup_UI()
        self.setFont(QFont("Times New Roman", 15))

    def point_size_update(self):
        self.parent.point_size = self.selection[self.box.currentText()]
        for shape in self.parent.zoomDisplay.canvas.shapes:
            shape.point_size = self.parent.point_size
        self.parent.zoomDisplay.canvas.update()
        
    def setup_UI(self):
        layout = QVBoxLayout(self)
        self.box = QComboBox(self)   # 加入下拉選單
        self.box.addItems(list(self.selection.keys()))   # 加入四個選項
        self.box.setCurrentIndex(list(self.selection.values()).index(self.parent.point_size))
        self.box.setGeometry(10,10,200,300)
        self.box.currentIndexChanged.connect(self.point_size_update)
        layout.addWidget(self.box)

        QBtn = QDialogButtonBox.Close

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox)


class FileSystemView(QDialog):
    dict_label_signal = pyqtSignal(dict)

    def __init__(self, directory_log, parent = None):
        super(FileSystemView, self).__init__(parent=parent)
        self.setWindowTitle("HISTORY LOG")
        self.directory = directory_log
        self.setup_UI()

    def setup_UI(self):
        self.setMinimumWidth(450)
        layout = QVBoxLayout(self)
        
        self.tree = QTreeView()
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.context_menu)
        self.tree.doubleClicked.connect(self.open_file)
        self.tree.setFont(QFont("Times New Roman", 15))

        self.model = QFileSystemModel()
        self.model.setFilter(QDir.Files)
        self.model.setRootPath(self.directory)
        self.model.setReadOnly(True)
        self.model.setNameFilters(['*.log'])
        self.model.setNameFilterDisables(0)

        
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(self.directory))
        self.tree.setSortingEnabled(True)
        self.tree.hideColumn(1)
        self.tree.hideColumn(2)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree)
 
        QBtn = QDialogButtonBox.Close

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox)
        

    def context_menu(self):
        menu = QMenu(self)
        open = new_action(self, 'Open log', self.open_file, None, 'open', "open log file")
        delete = new_action(self, 'Delete log', self.delete_file, None, 'delete', "open log file")
        menu.addAction(open)
        menu.addAction(delete)
        cursor = QCursor()
        menu.exec_(cursor.pos())

    def open_file(self):
        index = self.tree.currentIndex()
        file_name = self.model.filePath(index)
        if os.path.isfile(file_name):
            dict_label = load_dict(file_name)
            self.dict_label_signal.emit(dict_label)
            self.reject()

    def delete_file(self):
        index = self.tree.currentIndex()
        file_name = self.model.filePath(index)
        if os.path.isfile(file_name):
            os.remove(file_name)
        self.tree.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("QtCurve")
    dialog = FileSystemView(r"D:\02_BME\NoduleDetection_v5\data\NCKUH\0003\log")
    dialog.show()
    sys.exit(app.exec_())
    

