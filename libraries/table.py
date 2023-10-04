from PyQt5.QtCore    import *
from PyQt5.QtGui     import *
from PyQt5.QtWidgets import *
import sys

from libraries.constants import BUTTON_DEFAULT, btnsChangeColor, id_color_nodule, TEXT_FONT_SMALL
from libraries.utilsUI import new_action, new_icon
from libraries.toolButton import tool_button

class TableWidget_List(QTableWidget):
    style_sheet = """
    QTableView
    {
        color: #000000;
        background-color: #ffffff;
        selection-background-color: #00D1FF;
        selection-color: #000000;
    }
    """
    delete_row_signal = pyqtSignal()
    def __init__(self, header = None):
        if header is None or len(header) == 0:
            self.header = ['PatientID','Name', 'Confirmed',
                            'Gender', 'Date_of_Birth',
                            'Modality', 'Date_of_Study', 
                            'Path', 'Style', 'LOG']
            self.title = ['PatientID','Name', 'Confirmed',
                            'Gender', 'Date of Birth',
                            'Modality', 'Date of Study', 
                            'Path', 'Style', 'LOG']
        else:
            self.header = header
        super().__init__(0, len(self.header))
        self.setFont(QFont("Times New Roman", 15))
        self.setHorizontalHeaderLabels(self.header)
        self.resizeColumnsToContents()
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setSelectionMode(QAbstractItemView.NoSelection)
        self.horizontalHeader().sectionClicked.connect(self.sortByHeader)
        self.horizontalHeader().setFont(QFont("Times New Roman", 15))
        self.list_data = []
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setStyleSheet(self.style_sheet)
        self.sortBy = 0
        self.sortReverse = False

    def _addData(self, data:list):
        """
        Data is the list.
        In each items is Dictionary
        """
        row = len(data)
        if row > 0:
            self.setRowCount(row)
            for row, itemvalue in enumerate(data):
                for column, key in enumerate(self.header):
                    newitem = QTableWidgetItem(data[row][key])
                    if key == "Path" or key == 'LOG':
                        newitem.setTextAlignment(Qt.AlignAbsolute | Qt.AlignLeft)
                    else:
                        newitem.setTextAlignment(Qt.AlignAbsolute | Qt.AlignRight)
                    self.setItem(row, column, newitem)
            self.list_data.extend(data)
        self.setHorizontalHeaderLabels(self.title)
        # self.resizeColumnsToContents()
        # self.resizeRowsToContents()

    def _addRow(self):
        rowCount = self.rowCount()
        self.insertRow(rowCount)

    def _addRow_Dict(self, itemData:dict):
        rowCount = self.rowCount()
        if len(self.list_data) > 0:
            if itemData in self.list_data:
                return
        self.insertRow(rowCount)
        for column, key in enumerate(itemData):
            item = QTableWidgetItem(itemData[key])
            item.setTextAlignment(Qt.AlignAbsolute | Qt.AlignRight)
            self.setItem(rowCount, column, item)

        # self.setHorizontalHeaderLabels(itemData.keys())
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.list_data.append(itemData)
        self.update()
    
    def _removeRow_index(self, index):
        if self.rowCount() > 0:
            if index < self.rowCount():
                self.removeRow(index)
                self.list_data.pop(index)
                self.update()

    def getSelectRow(self):
        items = self.selectedItems()
        return items[0].row()
    
    def getListValue(self):
        return self.list_data

    def _context_menu(self):
        delete = new_action(self, "Remove Patient", self.delete_patient, None, "delete", "Remove path of patient file")
        menu = QMenu(self)
        menu.setStyleSheet(TEXT_FONT_SMALL)
        menu.addAction(delete)
        cursor = QCursor()
        menu.exec_(cursor.pos())

    def delete_patient(self):
        index = self.currentRow()
        self._removeRow_index(index)
        self.delete_row_signal.emit()
        self.update()
    
    def sortByHeader(self, logicalIndex):
        if self.sortBy == logicalIndex:
            self.sortReverse = not self.sortReverse
        else:
            self.sortBy = logicalIndex
            self.sortReverse = False
        self.sortByColumn(self.sortBy, Qt.SortOrder.DescendingOrder if self.sortReverse else Qt.SortOrder.AscendingOrder)
        
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        current_value = self.verticalScrollBar().value()
        if delta > 0:
            self.verticalScrollBar().setValue(current_value - 1)
        else:
            self.verticalScrollBar().setValue(current_value + 1)
        event.accept()

class PatientInforTable(QTableWidget):
    def __init__(self, vheader = None):
        if vheader is None:
            self.vheader = ['PatientID', 'Name', 'Gender', 'Date_of_Birth', 'Modality', 'Date_of_Study', 'Slice_Thickness', 'Pixel_Spacing']
        else:
            self.vheader = vheader
        super().__init__(8, 1)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.horizontalHeader().hide()
        self.setVerticalHeaderLabels(self.vheader)
        # self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel) # modified by Ben
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.horizontalHeader().setMinimumWidth(1) # modified by Ben
        self.setFont(QFont("Times New Roman", 15))
        

    def _addData(self, data:dict):
        """
        Data is the Dictionary
        """
        self.setVerticalHeaderLabels(self.vheader)
        for row, header in enumerate(self.vheader):
            try:
                content = str(data[header])
            except:
                content = ""
            self.setItem(row, 0, QTableWidgetItem(content))
            self.resizeRowsToContents()
            self.resizeColumnsToContents()
            self.setWordWrap(True)
        
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        current_value = self.verticalScrollBar().value()
        if delta > 0:
            self.verticalScrollBar().setValue(current_value - 1)
        else:
            self.verticalScrollBar().setValue(current_value + 1)
        event.accept()

class NoduleComboBox(QComboBox):
    comboListDefault = ['Benign', 'Probably Benign','Probably Suspicious','Suspicious']
    commitColor = pyqtSignal(str, str)
    def __init__(self, parent = None) -> None:
        super(NoduleComboBox, self).__init__(parent)
        self.setFont(QFont('Times New Roman', 15))
        
        self.addItems(self.comboListDefault)
        # self.combo.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        # self.lineEdit().setAlignment(Qt.AlignRight)
        self.setStyleSheet("QComboBox{color : #000000; background-color : %s;}" % (QColor(*id_color_nodule[0]).name()))
        self.currentIndexChanged.connect(self.commit_editor)

    def wheelEvent(self, e: QWheelEvent) -> None:
        return None
        
    def commit_editor(self):
        color = QColor()
        if self.currentText() == self.comboListDefault[0]:
            color = QColor(*id_color_nodule[0])
        elif self.currentText() == self.comboListDefault[1]:
            color = QColor(*id_color_nodule[1])
        elif self.currentText() == self.comboListDefault[2]:
            color = QColor(*id_color_nodule[2])
        elif self.currentText() == self.comboListDefault[3]:
            color = QColor(*id_color_nodule[3])
        print(self.currentText(), color.name())
        qss = """QComboBox{color : #000000; background-color : %s;}""" % (color.name(),)
        self.setStyleSheet(qss)
        self.commitColor.emit(color.name(), self.currentText())

    def getCurrentColor(self):
        index = self.currentIndex()
        return QColor(*id_color_nodule[index])

class AnalysisTable(QTableWidget):
    style_sheet = """
    QTableView
    {
        color: #000000;
        background-color: #ffffff;
    }
    """
    delete_signal = pyqtSignal(int, list, list)
    edit_signal = pyqtSignal(bool, list)
    color_signal = pyqtSignal(int, dict, int)
    def __init__(self, header = None):
        if header is None:
            self.header = ['Nodule ID','Slice Range',
                            "Diameter (mm)","Category",
                            "Options"]
        else:
            self.header = header
        super().__init__(0, len(self.header))
        self.setFont(QFont("Times New Roman", 15))
        self.initial()
        class StyleDelegate_for_QTableWidget(QStyledItemDelegate):
            color_default = QColor("#aaedff")

            def paint(self, painter: QPainter, option: 'QStyleOptionViewItem', index: QModelIndex):
                if option.state & QStyle.State_Selected:
                    option.palette.setColor(QPalette.ColorRole.HighlightedText, Qt.black)
                    color = self.combineColors(self.color_default, self.background(option, index))
                    option.palette.setColor(QPalette.ColorRole.Highlight, color)
                QStyledItemDelegate.paint(self, painter, option, index)

            def background(self,option: 'QStyleOptionViewItem', index: QModelIndex):
                item = self.parent().itemFromIndex(index)
                if item:
                    if item.background() != QBrush():
                        return item.background().color()
                if self.parent().alternatingRowColors():
                    if index.row() % 2 == 1:
                        return option.palette.color(QPalette.ColorRole.AlternateBase)
                return option.palette.color(QPalette.ColorRole.Base)

            @staticmethod
            def combineColors(c1, c2):
                c3 = QColor()
                c3.setRed((c1.red() + c2.red())/2)
                c3.setGreen((c1.green() + c2.green())/2)
                c3.setBlue((c1.blue() + c2.blue())/2)
                return c3
        self.setItemDelegate(StyleDelegate_for_QTableWidget(self))

    def initial(self):
        self.setHorizontalHeaderLabels(self.header)
        self.resizeColumnsToContents()
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        try:
            index = self.header.index("Category")
            self.horizontalHeader().setSectionResizeMode(index, QHeaderView.Stretch)
            index = self.header.index("Options")
            self.horizontalHeader().setSectionResizeMode(index, QHeaderView.ResizeToContents)
        except:
            pass
        self.verticalHeader().hide()
        self.horizontalHeader().setSelectionMode(QAbstractItemView.NoSelection)
        self.horizontalHeader().sectionClicked.connect(self.sortByHeader)
        self.horizontalHeader().setFont(QFont("Times New Roman", 15))
        
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(self.style_sheet)
        self.list_data = []
        self.list_button = []
        self.list_combo = []
 
    def _addData(self, data:list):
        """
        Data is the list.
        In each items is Dictionary
        """
        row = len(data)
        if row > 0:
            _column = len(self.header)
            self.setRowCount(row)
            self.setColumnCount(_column)
            for row, itemvalue in enumerate(data):
                color = QColor(*id_color_nodule[0])
                for column, key in enumerate(itemvalue):
                    if isinstance(data[row][key], list) and key == 'Slice_range':
                        newitem = QTableWidgetItem("{:03d}-{:03d}".format(int(data[row][key][0]),int(data[row][key][-1])))
                    elif key == "Diameters":
                        newitem = QTableWidgetItem("{:.1f}".format(data[row][key]))
                        newitem.setToolTip("Diameter: {:5.1f}".format(data[row][key]))
                    elif key == "data" or key == 'description':
                        continue
                    elif key == 'Category':
                        combobox = NoduleComboBox()
                        index = combobox.comboListDefault.index(str(data[row][key]))
                        combobox.setCurrentIndex(index)
                        combobox.commitColor.connect(self.setColorCell)
                        self.setCellWidget(row, column, combobox)
                        continue
                    elif key == "NoduleID":
                        newitem = QTableWidgetItem("{:02d}".format(data[row][key]))
                    # elif key in []:
                    #     newitem = QTableWidgetItem(str(data[row][key]))
                    else:
                        pass
                    newitem.setTextAlignment(Qt.AlignAbsolute | Qt.AlignRight | Qt.AlignVCenter)
                    self.setItem(row, column, newitem)
                # Delete Button
                delete_button = tool_button(self)
                delete_button.setIcon(new_icon('delete'))
                delete_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
                # delete_button.setFixedWidth(100)
                # delete_button.setStyleSheet(BUTTON_DEFAULT)

                # Edit Button
                edit_button = tool_button(self)
                edit_button.setIcon(new_icon('editing'))
                edit_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
                # edit_button.setFixedWidth(100)
                # edit_button.setStyleSheet(BUTTON_DEFAULT)

                btns_widget = QWidget()
                btns_layout = QHBoxLayout(btns_widget)
                btns_layout.setContentsMargins(0,0,0,0)
                btns_layout.addWidget(edit_button)
                btns_layout.addWidget(delete_button)

                self.list_button.append(btns_widget)
                self.list_combo.append(combobox)
                
                
                self.setCellWidget(row, _column-1, btns_widget)
                delete_button.clicked.connect(self.handle_remove)
                edit_button.clicked[bool].connect(self.handle_edit)
                edit_button.setCheckable(True)
                
                color = combobox.getCurrentColor()
                try:
                    index_d = self.header.index('Diameter (mm)')
                    self.item(row, index_d).setBackground(color) 
                except:
                    print("Error set cell color")
            
            self.resizeRowsToContents()
            self.list_data.extend(data)
        else:
            self.clearContents()

    def _addRow_Dict(self, itemData:dict):
        rowCount = self.rowCount()
        if len(self.list_data) > 0:
            if itemData in self.list_data:
                return
        self.insertRow(rowCount)
        for column, key in enumerate(itemData):
            self.setItem(rowCount, column, QTableWidgetItem(itemData[key]))

        
        self.resizeRowsToContents()
        self.list_data.append(itemData)
    
    def _removeRow_index(self, index):
        if self.rowCount() > 0:
            if index < self.rowCount():
                self.removeRow(index)
                self.list_data.pop(index)

    def _copyRow(self):
        self.insertRow(self.rowCount())
        rowCount = self.rowCount()
        columnCount = self.columnCount()

        for j in range(columnCount):
            if not self.item(rowCount - 2, j) is None:
                self.setItem(rowCount - 1, j, QTableWidgetItem(self.item(rowCount - 2, j).text()))

    def getSelectRow(self):
        items = self.selectedItems()
        return items[0].row()
    def getListValue(self):
        return self.list_data

    def clear(self) -> None:
        super().clear()
        self.initial()
        return

    def delete_nodule_changes_dialog(self):
        yes, no= QMessageBox.Yes, QMessageBox.No
        msg = u'The data will not be recoverable.\n Are you sure you want to \npermanently delete the nodule?'
        return QMessageBox.question(self, u'Attention', msg, yes | no)

    def delete_nodule_warning(self):
        delete_nodule_changes = self.delete_nodule_changes_dialog()
        if delete_nodule_changes == QMessageBox.No:
            return False
        elif delete_nodule_changes == QMessageBox.Yes:
            return True
        else:
            return False

    def handle_remove(self):
        if self.delete_nodule_warning():
            button = self.sender()
            index = self.list_button.index(button.parent())
            item = self.list_data[index]
            slice_range = item['Slice_range']
            id_no = item['NoduleID']
            self._removeRow_index(index)
            self.edit_signal.emit(False, slice_range)
            self.delete_signal.emit(id_no, slice_range, self.list_data)
            self.list_button.pop(index)
            self.list_combo.pop(index)
            self.update()

    def handle_edit(self, pressed):
        button = self.sender()
        # print("table: ", pressed)
        index = self.list_button.index(button.parent())
        item = self.list_data[index]
        slice_range = item['Slice_range']
        self.edit_signal.emit(pressed, slice_range)
        self.update()

    def sortByHeader(self, logicalIndex):
        if len(self.list_data) > 0:
            self.sortByColumn(logicalIndex, Qt.SortOrder.AscendingOrder)

    def setColorCell(self, color, category):
        combo = self.sender()
        row = self.list_combo.index(combo)
        try:
            index_d = self.header.index('Diameter (mm)')
            self.item(row, index_d).setBackground(QColor(color)) 
        except:
            print("Error set cell color")
        item = self.list_data[row]
        item['Category'] = category
        noduleIndex = combo.comboListDefault.index(category)
        self.color_signal.emit(row, item, noduleIndex)
        
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        current_value = self.verticalScrollBar().value()
        if delta > 0:
            self.verticalScrollBar().setValue(current_value - 1)
        else:
            self.verticalScrollBar().setValue(current_value + 1)
        event.accept()
