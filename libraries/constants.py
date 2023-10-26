from PyQt5.QtGui import QColor

PATH_DICOM = "./data/dicom"
DICOM_DB = "./data/dicom.db"
# PATH_LOG = r"../Database_test/Log"


SETTING_FILENAME = 'filename'
SETTING_RECENT_FILES = 'recentFiles'
SETTING_WIN_SIZE = 'window/size'
SETTING_WIN_POSE = 'window/position'
SETTING_WIN_GEOMETRY = 'window/geometry'
SETTING_LINE_COLOR = 'line/color'
SETTING_FILL_COLOR = 'fill/color'
SETTING_ADVANCE_MODE = 'advanced'
SETTING_WIN_STATE = 'window/state'
SETTING_SAVE_DIR = 'savedir'
SETTING_PAINT_LABEL = 'paintlabel'
SETTING_LAST_OPEN_DIR = 'lastOpenDir'
SETTING_AUTO_SAVE = 'autosave'
SETTING_SINGLE_CLASS = 'singleclass'
FORMAT_PASCALVOC='PascalVOC'
FORMAT_YOLO='YOLO'
SETTING_DRAW_SQUARE = 'draw/square'
SETTING_LABEL_FILE_FORMAT= 'labelFileFormat'
DEFAULT_ENCODING = 'utf-8'

DEFAULT_WIN_WIDTH = 1920
DEFAULT_WIN_HEIGHT = 1137

ColorBBox = QColor(0, 255, 51, 150)
ColorBBox_fill = QColor(0, 255, 51, 80)
ColorLabel = QColor(3, 60, 23, 150)

TITLE_STRING1 = "Lung Nodule Detection System"
# TITLE_STRING2 = "National Cheng Kung University School of Medicine Medical Imaging Center and Thoracic Surgery"
# TITLE_STRING3 = "Ver 2.0.8, 2022 by Chengda Robot Laboratory"
# TITLE_STRING2 = "國立成功大學醫學院 醫學影像中心暨胸腔外科"
TITLE_STRING2 = "國立成功大學醫學院 醫學影像中心暨胸腔外科及影像醫學部"
TITLE_STRING3 = "Ver 2.0.11b, 2023.10.26 by 成大機器人實驗室"

## Background
BACKGROUND_COLOR_MAIN = "background-color:rgb(33,47,61);"
BACKGROUND_COLOR_BLACK = "background-color:rgb(0,0,0);"
BACKGROUND_COLOR_WHITE = "background-color:rgb(255,255,255);"
BACKGROUND_COLOR_QMAINWINDOW = "QMainWindow{background-color:rgb(33,47,61);}"

## Label
TEXT_COLOR_DEFAULT = "color:#ffffff;"
TEXT_COLOR_BLACK = "color:rgb(0,0,0);"
TEXT_FONT_LARGE = "font: 30px Times New Roman;"
TEXT_FONT_MEDIUM = "font: 24px Times New Roman;"
TEXT_FONT_SMALL = "font: 20px Times New Roman;"
L_FONT = 30
M_FONT = 24
S_FONT = 20

## BUTTON
BUTTON_DEFAULT = "QPushButton, QToolButton {background-color: rgb(127, 179, 213) ;\
                    border-style: outset;\
                    border-width: 2px;\
                    border-radius: 10px;\
                    border-color: rgb( 27, 38, 49 );\
                    font: bold 25px Times New Roman;\
                    padding: 5px;\
                    color:rgb(255,255,255); }\
                QPushButton:hover, QToolButton:hover{background-color : rgb(53, 148, 213);};\
                QPushButton:checked, QToolButton:checked{\
                    background-color: rgb(80, 80, 80);border: none;}"

def btnsChangeColor(r:int, g:int, b:int):

        return "QPushButton, QToolButton {background-color: " + "rgb({r}, {g}, {b}) ;".format(r=r, g=g, b=b) +\
                    "border-style: outset;\
                    border-width: 2px;\
                    border-radius: 10px;\
                    border-color: rgb( 27, 38, 49 );\
                    font: bold 25px Times New Roman;\
                    padding: 5px;\
                    color:rgb(255,255,255); }\
                QPushButton:hover, QToolButton:hover{background-color : rgb(53, 148, 213);};\
                QPushButton:checked, QToolButton:checked{\
                    background-color: rgb(80, 80, 80);border: none;}"

BUTTON_BACKGROUND_COLOR = "background-color:rgb(86, 101, 115);"
BUTTON_COLOR = "color:rgb(255,255,255);"
BUTTON_FONT = "font-size: 25px;"

## Foreground
COLOR_BLACK = "color:rgb(0,0,0);"
COLOR_DARK_BLUE = "color:rgb(40, 55, 71);"
COLOR_GRAY = "color:rgb(191, 201, 202);"

##TABLE
TABLE_DEFAULT = "color: black;\
                background-color: white;\
                selection-color: black;\
                selection-background-color: rgba( 212, 230, 241, 50);"

TABLE_DEFAULT2 ="background-color: white;\
                color: black;\
                QTableView{background-color: white;\
                color: black;\
                font: 28px Times New Roman;};\
                QTableView::item{color: black;\
                font: 28px Times New Roman;\
                spacing: 10px; margin:1px;};\
                QTableWidget::item:selected{background-color: palette(highlight);\
                color: palette(highlightedText);};\
                QHeaderView::section{spacing: 10px;\
                background-color: white;\
                color: black;\
                font: 28px Times New Roman;\
                text-align: center;}"

##SLIDER
SLIDER_DEFAULT = "QSlider::groove:horizontal {\
                border: 1px solid #999999;\
                height: 20px; \
                background-color: black;\
                }QSlider::handle:horizontal {\
                background-color: transparent;\
                border: 5px solid white;\
                height: 15px;\
                width: 3px;\
                margin-top: -3px;\
                margin-bottom: -3px;\
                }"
                
                # border: 4px;\
                # border-radius: 0px;\
                # border-color: black;\
                # height: 25px;\
                # width: 10px;\
                # margin: -8px 2; \
                # }
class_color_nodule = {'Benign': 'green', 'Probably Benign' : 'lightgreen', 'Probably Suspicious': 'orange', 'Suspicious': 'red'}
id_color_nodule = {0: (29, 177, 76, 255),
                1: (156, 230, 29, 255),
                2: (255, 156, 0, 255), 
                3: (255, 0, 0, 255),
                4: (54,156,223,255)}
id_color_nodule_fill = {0: (29, 177, 76, 125),
                1: (156, 230, 29, 125),
                2: (255, 156, 0, 125), 
                3: (255, 0, 0, 125), 
                4: (54, 156, 223, 125)}



