
from PyQt5.QtWidgets import QApplication, QInputDialog

class selectDialog(QInputDialog):

    def __init__(self, parent = None):
        super().__init__(parent=parent)
        self.setStyleSheet("background-color:rgb(213, 219, 219);font: 20px Times New Roman;")
        
        items = ("Image Folder (jpg, jpeg,...)", "Dicom Folder (dcm)", "MetaImage File (mhd)")
        item, ok = self.getItem(
                self, "Select Patient", "List of Input Data", items, 0, False
        )
        if ok and item:
            self.item = items.index(item)
        else:
            self.item = None
    
    def get_option(self):
        return self.item

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ex = selectDialog()
    print(ex.get_option())

