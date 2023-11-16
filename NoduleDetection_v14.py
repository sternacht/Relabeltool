import sys
import os
import time
import cv2
import argparse
import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Optional, Any
# from grabcut_dialog import grabcut_dialog
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try: 
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
except ImportError:
    if sys.version_info.major >=3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
# Libraries
from functools import partial
import utils.mergeOverlapping as olap

# import libraries.constants as const
from libraries import *
import glob

__appname__ = "2D Nodule Detection & Manual Labeling (RoboticLAB NCKU)"

class SHAPETYPE:
    POLYGON = 'polygon'
    RECTANGLE = 'rectangle'

class LoadModelThread(QThread):
    MODEL_LOADED_SIGNAL = pyqtSignal(tuple)
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def run(self):
        from libraries.interactiveSegment import init_model
        xml_path = r"./weights/interactive/fcanet.xml"
        bin_path = r"./weights/interactive/fcanet.bin"
        try:
            interactiveModel, output_keys = init_model(xml_path=xml_path, bin_path=bin_path)
            self.MODEL_LOADED_SIGNAL.emit((interactiveModel, output_keys))
        except Exception as e:
            print(e)
            self.MODEL_LOADED_SIGNAL.emit((None, None))

def points_to_mask(points: List[Tuple[int, int]], shape: Tuple[int, int, int]) -> np.ndarray:
    """Convert a list of points to a binary mask.
    
    Args:
        points (List[Tuple[int, int]]): list of points
        shape (Tuple[int, int, int]): shape of the image
    
    Return:
        A binary mask of shape (height, width) with dtype np.uint8
    """
    if len(shape) != 3:
        raise ValueError("shape must be a tuple of 3 elements")
    
    points = np.array(points, dtype = np.int32)
    
    mask = np.zeros(shape, dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask.copy(),
                            points = points,
                            color = (255,255,255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask

def get_timestamp(is_filename: bool = False) -> str:
    tw_zone = datetime.timezone(datetime.timedelta(hours=+8)) # Taiwan Timezone
    tw_time = datetime.datetime.now(tw_zone)
    # tw_time = tw_time + datetime.timedelta(minutes=3, seconds=5) # For CYCH
    if is_filename:
        timestamp = tw_time.strftime(f"%y%m%d_%H%M%S")
    else:
        timestamp = tw_time.strftime("%Y/%m/%d %H:%M:%S")
    return timestamp

def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default
    
class WindowUI_Mixin(object):
    """
    Set up UI for BME application
    """
    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            add_actions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        """
        Create a tool bar in UI.
        """
        toolbar = Toolbar(title)
        toolbar.setObjectName(u'%sToolBar'% title)
        toolbar.setToolButtonStyle(Qt.ToolButtonFollowStyle |  Qt.AlignLeft)
        if actions:
            add_actions(toolbar, (actions))

        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar

    def setTitleLabel(self, height_default= 50):
        title_layout = QHBoxLayout()
        logo_1 = new_label_image("ncku-logo.jpg", h=height_default)
        logo_2 = new_label_image("nckuh_logo.png", h=height_default)
        
        title_1 = QLabel(const.TITLE_STRING1)
        title_1.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignCenter)
        title_1.setFixedHeight(height_default)
        title_1.setStyleSheet(const.TEXT_COLOR_DEFAULT + ";" + const.TEXT_FONT_MEDIUM)
        
        title_2 = QLabel(const.TITLE_STRING2)
        title_2.setAlignment(Qt.AlignLeading|Qt.AlignRight|Qt.AlignCenter)
        title_2.setFixedHeight(height_default)
        title_2.setStyleSheet(const.TEXT_COLOR_DEFAULT + ";" + const.TEXT_FONT_MEDIUM)
        
        title_layout.addWidget(logo_1)
        title_layout.addWidget(title_1)
        title_layout.addStretch(1)
        title_layout.addWidget(title_2)
        title_layout.addWidget(logo_2)
        return title_layout

    def setStatusBar_custom(self, text = const.TITLE_STRING3):
        self.copyright_label = QLabel(text)

        # adding label to status bar
        self.statusBar().addPermanentWidget(self.copyright_label)
        
    def setNavigateArea(self):
        layout = QVBoxLayout()
        self.cursor= QLabel("")
        # self.cursor.setFixedWidth(400)
        self.cursor.setObjectName('cursor')
        self.cursor.setStyleSheet(const.TEXT_COLOR_DEFAULT + const.TEXT_FONT_MEDIUM)


        bbox_label = QLabel("Manual Labelling")
        bbox_label.setStyleSheet(const.TEXT_FONT_MEDIUM)
        self.segmentation_button = QPushButton("Segmentation")
        self.segmentation_button.setEnabled(False)

        note = QLabel("FG: Mouse Left\nBG: Mouse Right")
        note.setStyleSheet(const.TEXT_FONT_SMALL)

        self.undoSegment_button = QPushButton("Undo")
        self.undoSegment_button.setEnabled(False)

        self.resetSegment_button = QPushButton("Reset")
        self.resetSegment_button.setEnabled(False)

        self.finishSegment_button = QPushButton("Finish Segment")
        self.finishSegment_button.setEnabled(False)

        hlayout = QHBoxLayout()
        progress_label = QLabel("Progress")
        self.progress_bar = QProgressBar()
        self.progress_bar.setEnabled(False)
        self.progress_bar.setMinimum(1)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(1)
        hlayout.addWidget(progress_label)
        hlayout.addWidget(self.progress_bar)

        self.lineEdit = QLineEdit()
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setAlignment(Qt.AlignRight)
        self.lineEdit.setValidator(QIntValidator())
        self.lineEdit.setFixedWidth(50)

        self.TotalImage = QLabel()
        self.TotalImage.setObjectName("/ TotalImage")
        self.TotalImage.setText("total images")
        self.TotalImage.setAlignment(Qt.AlignLeading|Qt.AlignCenter|Qt.AlignCenter)
        self.TotalImage.setStyleSheet(const.TEXT_COLOR_DEFAULT + const.TEXT_FONT_MEDIUM)

        self.save = QPushButton("Save")
        self.save.setStyleSheet(const.TEXT_FONT_MEDIUM)
        self.save.setEnabled(False)

        self.confirm = QPushButton("Confirm")
        self.confirm.setStyleSheet(const.TEXT_FONT_MEDIUM)
        self.confirm.setEnabled(False)

        self.measure = QPushButton("Measure")
        self.measure.setStyleSheet(const.TEXT_FONT_MEDIUM)
        self.measure.setEnabled(False)
        
        line_edit_layout = QHBoxLayout()
        line_edit_layout.addWidget(self.lineEdit)
        line_edit_layout.addWidget(self.TotalImage)
        
        layout.addWidget(self.cursor, alignment=Qt.AlignCenter)

        layout.addWidget(bbox_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.segmentation_button)
        layout.addWidget(note)
        layout.addWidget(self.undoSegment_button)
        layout.addWidget(self.resetSegment_button)
        layout.addWidget(self.finishSegment_button)
        layout.addLayout(hlayout)
        layout.addStretch(1)
        layout.addWidget(self.save)
        layout.addWidget(self.confirm)
        layout.addWidget(self.measure)
        layout.addStretch(1)

        layout.addLayout(line_edit_layout)

        return layout

class MainWindow(QMainWindow, WindowUI_Mixin):
    def __init__(self, 
                 user_name: str,
                 parent = None, 
                 ) -> None:
        super().__init__(parent)
        self._init_params()
        self._reset_params()
        
        self.setup_ui()
        self.set_actions()
        self.actionConnect()
        self.build_ui()
        self.setStatusBar_custom()
        self.user_name = user_name

        self._init_auto_refresh()

    def _init_params(self):
        """
        This function is used to initialize all parameters and just run once when the program is started.
        """
        self.history = [] # A list of patients' information, e.g.: [{'PatientID': '000569', 'Gender':'F'}, {'PatientID': '001569', 'Gender':'M'}]
        self.loaded_path = set() # A set of loaded path, this is used to avoid loading the same path twice

        self.last_loaded_dicom_db_time = None
        
        self.dicom_db_path = const.DICOM_DB
        self.point_size = 4
        # Model
        self.interactiveModel = None
        self.output_keys = ['output']
    
    def _init_auto_refresh(self):
        self.last_refresh_time = time.time()
        self.auto_refresh_freqency = 60 # seconds
        
        self.auto_refresh_timer = QTimer(self)
        self.auto_refresh_timer.timeout.connect(self.refresh_patient_table)
        self.auto_refresh_timer.start(self.auto_refresh_freqency * 1000) # 1 minutes
    
    def _reset_params(self):
        """
        This function is used to initialize all parameters and run every time when select a new patient.
        """
        # Basic
        self.file_path = None
        self.mImgList = []
        self.dirname = ""
        self.image_data = []
        self.mImgSize = None
        
        self.index = 0  # self.cur_img_index
        self.current_slice = None
        self.img_count = len(self.mImgList)
        self.items_to_shapes = {}
        self.shapes_to_items = {}
        # Whether we need to save or not.
        self.dirty = False
        
        self.line_color = None
        self.fill_color = None

        self._no_selection_slot = False
        self.recent_files = []
        self.spacing = (0.6, 0.6, 1)
        self.zoom_x = None
        self.zoom_y = None

        # store the nodule group id need to adapt after propagation
        self.deleted_group_ids_after_propagation = set()
        # Mode default setting
        self.s = [] # keep total nodule on cur_slice for benign hidden mode
        self.edit_on = 0
        self.hidden_on = False
        self.view_lock = False

        # Advanced"
        self.image_data_dict = {}
        self.data_size = None
        self.keyboard_input = False
        self.wheel_scroll = False
        self.patient_infor = None
        # self.history_path = os.path.join(os.getcwd(),"history.dat")
        # self.history = pck.load(self.history_path)

        self.update_recent_file()
        self.results_nodule = None # Dict[int, List[Dict[str, Any]]], dict of pair (int, list of dict), key is slice index, value is list of dict
        self.results_nodule_analysis = None

        self.mask_segment = None
        self.polygons_segment = None
        self.save_file = None
        self.label_file = None
        self.save_dir = "" # selected patient's save dir
        self.save_status = False
        self.path_log = ""

        # Interactive Segmentation
        self.pred_mask = None

    def setup_ui(self):
        self.setObjectName("MainWindow")
        self.setStyleSheet(open(os.path.abspath(r"libraries/BME_UI.stylesheet")).read())     
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        #Main Window
        windowLayout = QVBoxLayout(self.centralwidget)
        title_layout = self.setTitleLabel()
        windowLayout.addLayout(title_layout)

        # Group refresh data
        self.refresh_button = QToolButton(self)
        self.refresh_button.setIcon(new_icon('refresh.png'))
        self.refresh_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.refresh_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Search
        self.query = QLineEdit(self)
        self.query.setPlaceholderText("Search...")
        self.query.textChanged.connect(self.search)

        # search layout
        search_lay = QHBoxLayout()
        search_lay.addWidget(self.query, 1)
        search_lay.addWidget(self.refresh_button, 0)

        # Table File Widget
        self.tableFile = TableWidget_List(header=None)
        self.tableFile.setMaximumHeight(int(self.height()*0.25))
        self.tableFile.setMinimumHeight(int(self.height()*0.20))
        self.group_box_select = QGroupBox("1. Select a Patient")
        
        group_layout_select = QVBoxLayout(self.group_box_select)
        group_layout_select.addLayout(search_lay)
        group_layout_select.addWidget(self.tableFile)
        # Group show patient information
        self.group_box_infor = QGroupBox("2. Patient Information")
        group_box_infor_layout = QHBoxLayout(self.group_box_infor)
        #List File Widget
        self.refresh_patient_table()
        self.listFileWidget = PatientInforTable()
        group_box_infor_layout.addWidget(self.listFileWidget)
        self.group_box_infor.setFixedWidth(int(self.width()*0.25)) # modified by Ben

        # Inference Button
        # self.inference_button = QPushButton("Inference...")
        # self.inference_button.setIcon(new_icon('detect.png'))
        # self.inference_button.setEnabled(False)
        # self.inference_button.setStyleSheet(const.TEXT_FONT_MEDIUM )

        # Log Button
        self.log_button = QPushButton("History Log")
        self.log_button.setEnabled(False)

        self.inference_label = QLabel()
        self.inference_label.setFont(QFont("Times New Roman", 18))
        self.inference_label.setVisible(False)

        group_box_inference = QGroupBox("3. Load History Log")
        group_layout_inference = QVBoxLayout(group_box_inference)
        group_layout_inference.addWidget(self.inference_label,0)
        gli = QHBoxLayout()
        gli.setContentsMargins(0,0,0,0)
        gli.addWidget(self.log_button)
        group_layout_inference.addLayout(gli, 1)
        

        # Table Recall Precision Widget
        self.table_analysis = AnalysisTable(header=None)
        self.table_analysis.setContentsMargins(0, 0, 0, 0)
        self.table_analysis.setEnabled(False)
        # self.table_analysis.setMaximumWidth(QApplication.desktop().window().width()//3)
        # self.table_analysis.setMinimumHeight(QApplication.desktop().window().height()*0.15)
        # self.table_analysis.setMinimumWidth(QApplication.desktop().window().width()*8//20)
        # self.table_analysis.setMaximumSize(QApplication.desktop().window().width()//3, QApplication.desktop().window().height()*0.25)

        self.add_button = tool_button()
        self.add_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.add_button.setIcon(new_icon('plus'))
        self.add_button.setEnabled(False)

        self.edit_button = tool_button()
        self.edit_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.edit_button.setIcon(new_icon('editing'))
        self.edit_button.setEnabled(False)

        self.delete_button = tool_button()
        self.delete_button.setToolButtonStyle(Qt.ToolButtonIconOnly)

        self.update_button = tool_button()
        self.update_button.setToolButtonStyle(Qt.ToolButtonIconOnly)

        self.view_button = tool_button()
        self.view_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.view_button.setIcon(new_icon('eye'))

        tool_btns_layout = QHBoxLayout()
        tool_btns_layout.setContentsMargins(0, 0, 2, 0)
        tool_btns_layout.addStretch(1)
        tool_btns_layout.addWidget(self.add_button)
        tool_btns_layout.addWidget(self.edit_button)
        tool_btns_layout.addWidget(self.update_button)
        tool_btns_layout.addWidget(self.view_button)

        ## Label List
        self.label_list = QListWidget()

        listview_layout = QVBoxLayout()
        listview_layout.setContentsMargins(0, 0, 0, 0)
        listview_layout.addLayout(tool_btns_layout)
        listview_layout.addWidget(self.label_list, 1)
        self.edit_slice = QWidget()
        self.edit_slice.setLayout(listview_layout)

        ## Pathology Area: Doctor can write down or describe the pathology of Nodule
        self.group_plain_text = QGroupBox("Pathology")
        self.pathology_text = PlainText()
        self.pathology_text.setEnabled(False)
        layout_plain_text = QHBoxLayout(self.group_plain_text)
        layout_plain_text.addWidget(self.pathology_text, 1)


        ##Group Nodule
        self.group_box_lung_nodule = QGroupBox("Lung Nodule")
        group_layout_lung_nodule = QHBoxLayout(self.group_box_lung_nodule)
        group_layout_lung_nodule.addWidget(self.table_analysis, 2)
        group_layout_lung_nodule.addWidget(self.edit_slice,0)

        self.display = Display()
        self.display.wheel_down.connect(self.open_next_image)
        self.display.wheel_up.connect(self.open_prev_image)
        self.display.canvas.newShape.connect(self.new_shape)
        self.display.canvas.selectionChanged.connect(self.shape_selection_changed)
        self.display.canvas.shapeMoved.connect(self.set_dirty)
        self.display.edit_shape.connect(lambda shapes: self.zoomDisplay.update_shape(shapes))
        self.display.canvas.drawingPolygon.connect(self.toggle_drawing_sensitive)
        self.display.canvas.pointSegment.connect(self.segmentation)
        
        ## Slider 
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setObjectName('slider')

        # Control button: preview
        toolbuttonSizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.prevButton = tool_button()
        self.prevButton.setObjectName("prevButton")
        self.prevButton.setToolButtonStyle(Qt.ToolButtonIconOnly | Qt.ToolButtonFollowStyle)
        self.prevButton.setSizePolicy(toolbuttonSizePolicy)
        self.prevButton.setFixedHeight(21)

        self.nextButton = tool_button()
        self.nextButton.setObjectName("nextButton")
        self.nextButton.setToolButtonStyle(Qt.ToolButtonIconOnly | Qt.ToolButtonFollowStyle)
        self.nextButton.setSizePolicy(toolbuttonSizePolicy)
        self.nextButton.setFixedHeight(21)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.prevButton)
        control_layout.addWidget(self.slider, 1)
        control_layout.addWidget(self.nextButton)

        # Zoom Image
        self.zoomDisplay = ZoomDisplay(zoom=300)
        self.zoomDisplay.edit_shape.connect(lambda shapes: self.display.update_shape(shapes))
        self.zoomDisplay.create_shape.connect(self.zoom_create)
        self.zoomDisplay.canvas.pointSegment.connect(self.segmentation)
        self.zoomDisplay.canvas.selectionChanged.connect(self.shape_selection_changed)

        navigateLayout = self.setNavigateArea()
        
        self.action_layout = QGridLayout()
        self.action_layout.addWidget(self.group_box_select, 0, 0, 1, 2)
        self.action_layout.addWidget(self.group_box_infor, 1, 0, 1, 1 )
        self.action_layout.addWidget(group_box_inference, 2, 0, 2, 1)
        self.action_layout.addWidget(self.zoomDisplay, 1, 1, 3, 1)
        self.action_layout.addWidget(self.display, 0, 2, 4, 4)
        self.action_layout.addLayout(control_layout, 4, 2, 1, 4)
        self.action_layout.addWidget(self.group_box_lung_nodule, 5, 0, 1, 2)
        self.action_layout.addWidget(self.group_plain_text, 5, 2, 1, 5)
        self.action_layout.addLayout(navigateLayout, 0, 6, 5, 1)
        self.action_layout.setColumnStretch(1, 1)
        self.action_layout.setColumnStretch(2, 2)

        self.label_coordinates = QLabel("")
        self.label_coordinates.setStyleSheet(const.TEXT_COLOR_DEFAULT)
        self.statusBar().addPermanentWidget(self.label_coordinates)
        windowLayout.addLayout(self.action_layout)
        self.setCentralWidget(self.centralwidget)
        
        self.retranslate_ui()
        QMetaObject.connectSlotsByName(self)

    def retranslate_ui(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", __appname__))
        self.TotalImage.setText(_translate("MainWindow", "/Total Images"))
        self.group_box_lung_nodule.setTitle(_translate("MainWindow", "Total Lung Nodules"))

    def resizeEvent(self, event) -> None:
        self.group_box_lung_nodule.setFixedWidth(int(self.width()*0.48))
        self.table_analysis.setMinimumHeight(int(self.height()*0.15))
        self.tableFile.setFixedHeight(int(self.height()*0.20))
        self.zoomDisplay.setMaximumWidth(int(self.zoomDisplay.height()*1.8))
        win_width = self.width()
        self.group_box_infor.setFixedWidth(int(win_width*0.25))
        if win_width > 1500 and win_width <=1920:
            self.label_list.setFixedWidth(int(self.group_box_lung_nodule.width()*0.20))
            self.edit_slice.setFixedWidth(int(self.group_box_lung_nodule.width()*0.20))
        return super().resizeEvent(event)
        # self.table_analysis.setMinimumSize(int(self.width()/0.333), int(self.height()*0.15))
        # self.load_button.font().setPixelSize(int(self.width()*const.M_FONT/const.DEFAULT_WIN_WIDTH))

    def set_actions(self):
        action = partial(new_action, self)
        quit = action('Quit', self.close,
                      'Ctrl+Q', 'quit', 'quitApp')

        open = action('Open File', self.openDicomDialog,
                      'Ctrl+O', 'open', 'open File Detail')
        open_next_image = action("Next Img", self.open_next_image, 'd', 'next', 'Next Image Detail', enabled = True)

        open_prev_image = action("Prev Img", self.open_prev_image, 'a', 'prev', 'Previous Image Detail', enabled = True)

        close = action('Close File', self.close_file, 'Ctrl+W', 'close', 'closeCurDetail')

        create_mode = action('Create Rectangle', lambda: self.toggle_draw_mode(False, 'rectangle'), 'n', 'plus', 'createBoxDetail', enabled = False)

        create_poly_mode = action('Create Polygon', lambda: self.toggle_draw_mode(False, 'polygon'), 'p', 'square' , None, enabled= False)

        create_point_mode = action('Create Point', lambda: self.toggle_draw_mode(False, 'point'), 'Ctrl+p', 'tag', "Interactive Segmentation", enabled= False)

        edit_mode = action('Edit Box', self.set_edit_mode, 'Ctrl+J', 'editing', 'Edit Box Detail', enabled= False)

        view_mode = action('View Box', self.set_view_mode, 'Ctrl+V', 'eye', 'viewBoxDetail', enabled= False)

        leave_mode = action('View Box', self.set_view_mode, 'Esc', 'eye', 'viewBoxDetail', enabled= True)

        delete = action('Del Box', self.delete_selected_shape, 'Del', 'delete', 'delBoxDetail', enabled = False)

        undo = action("Undo", self.undo_shape_edit, 'Z', None, 'Undo last add and edit shape')

        undoLastPoint = action("Undo Last Point", self.display.canvas.undoLastPoint, 'Ctrl+z', None, "Reomove selected point", enabled=False)

        update_analysis = action('Update Analysis', self.update_analysis_table, None, 'update', 'Update table', enabled = False)

        show_shortcut = action('Help Category', lambda: help_dialog(self).exec_(), None, 'help', 'shortcut')

        show_confirm_status = action('Confirmed Status', self.show_confirmed_status, None, 'Show Confirmed Status', 'status')

        save = action('Save', self.save_all_labels,
                      'Ctrl+S', 'save', 'save Detail', enabled=False)
        open_image_folder = action("Open Image Slices", self.openDicomDialog, None, None, None, enabled= True)
        open_mhd = action("Open MetaImage (.mhd)", self.openFileMetaImage, None, None, None, enabled= True)
        open_dicom = action("Open Dicom File", self.openDicomDialog, None, None, None, enabled= True)
        change_point_size = action("Change Point Size", lambda: change_size(self).exec_(), None, 'change size', 'change size')
        change_auto_refresh_freq = action("Change Auto Refresh Frequency", self.change_auto_refresh_frequency, None, 'change auto refresh', 'change auto refresh')
        
        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.display.zoomWidget)
        self.display.zoomWidget.setWhatsThis(u"Zoom in/out of the image.  Also accessible with " "%s and %s from the canvas." % (format_shortcut("Ctrl+[-+]"), format_shortcut("Ctrl+Wheel")))

        zoom_in = action('Zoom In', partial(self.zoomDisplay.add_zoom, 10),
                         'Ctrl++', 'zoom-in', 'zoominDetail', enabled=False)
        zoom_out = action('Zoom Out', partial(self.zoomDisplay.add_zoom, -10),
                          'Ctrl+-', 'zoom-out', 'zoomoutDetail', enabled=False)
        zoom_org = action('Original Size', partial(self.zoomDisplay.set_zoom, 100),
                          'Ctrl+=', 'zoom', 'originalsizeDetail', enabled=False)
        fit_window = action('Fit Window', self.set_fit_window,
                            'Ctrl+F', 'fit-window', 'fitWinDetail',
                            checkable=True, enabled=False)
        fit_width = action('Fit Width', self.set_fit_width,
                           'Ctrl+Shift+F', 'fit-width', 'fitWidthDetail',
                           checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoom_actions = (self.display.zoomWidget, zoom_in, zoom_out,
                        zoom_org, fit_window, fit_width)

        label_menu = QMenu()
        add_actions(label_menu, (delete, None))
        self.label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(self.pop_label_list_menu)

        self.actions = struct(open= open, close= close, save= save,
                    delete= delete, next=open_next_image, previous= open_prev_image,
                    createMode= create_mode, editMode= edit_mode, viewMode = view_mode,
                    leaveMode = leave_mode,
                    createPolyMode = create_poly_mode,
                    createPointMode = create_point_mode,
                    undoLastPoint = undoLastPoint,
                    undo = undo,
                    zoom= zoom, zoomIn= zoom_in, 
                    zoomOut= zoom_out, zoomOrg= zoom_org,
                    fitWindow= fit_window, fitWidth= fit_width, 
                    zoomActions= zoom_actions, 
                    fileMenuActions= (open, close, quit), 
                    editMenu= (delete, None),
                    onLoadActive= (close, create_mode, edit_mode, leave_mode, view_mode, create_poly_mode),
                    onShapesPresent= (update_analysis, delete),
                    update=update_analysis,
                    help=show_shortcut,
                    status=show_confirm_status,
                    change_point_size = change_point_size,
                    change_auto_refresh_freq = change_auto_refresh_freq,
                    openImage = open_image_folder, openMhd = open_mhd, openDicom = open_dicom)
        
        self.menus = struct(
            file= self.menu('File'),
            edit= self.menu('Edit'),
            view= self.menu('View'),
            help= self.menu('Help'),
            recentFiles= QMenu('Open Recent'),
            labelList= label_menu
        )
        # Auto saving : Enable auto saving if pressing next
        self.auto_saving = QAction('Auto Save Mode', self)
        self.auto_saving.setCheckable(True)
        self.auto_saving.setChecked(True)
        # Benign hidden : Hidden benign nodule
        self.benign_hidden = QAction('Benign Hidden', self)
        self.benign_hidden.setCheckable(True)
        self.benign_hidden.setChecked(False)
        self.benign_hidden.triggered.connect(self.set_benign_hidden)
        # Sync single class mode from PR#106
        self.lastLabel = 'nodule'
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.display_label_option = QAction('Display Label', self)
        self.display_label_option.setShortcut("Ctrl+Shift+P")
        self.display_label_option.setCheckable(True)
        self.display_label_option.setChecked(False)
        self.display_label_option.triggered.connect(self.set_display_label)
        # self.display_label_option.triggered.connect(self.toggle_paint_labels_option)
        # self.change_point_size = QAction('Select Point Size', self)
        # self.change_point_size.setEnabled(True)
        # self.change_point_size.connect(self.change_shape_point_size)

        add_actions(self.menus.file, (open, save, close, quit))
        add_actions(self.menus.edit, (create_mode, create_poly_mode , edit_mode, view_mode, leave_mode))
        add_actions(self.menus.view, (
            self.auto_saving,
            self.benign_hidden,
            self.display_label_option,
            None,
            zoom_in, zoom_out, zoom_org, 
            None,
            fit_window, fit_width,
            change_point_size
        ))
        add_actions(self.menus.help, (show_shortcut, show_confirm_status, change_auto_refresh_freq, None))
        pass

    def actionConnect(self):
        open_menu = QMenu(self)
        open_menu.addAction(self.actions.openDicom)
        open_menu.addAction(self.actions.openImage)
    
        self.prevButton.setDefaultAction(self.actions.previous)
        self.nextButton.setDefaultAction(self.actions.next)
        self.delete_button.setDefaultAction(self.actions.delete)
        self.menus.file.aboutToShow.connect(self.update_file_menu)
        self.update_button.setDefaultAction(self.actions.update)
        
    def build_ui(self):
        "Create connections in User interface"
        # Basic
        self.update_file_menu()
        self.label_list.itemActivated.connect(self.label_selection_changed)
        self.label_list.itemSelectionChanged.connect(self.label_selection_changed)
        # Connect to itemChanged to detect checkbox changes.
        self.label_list.itemChanged.connect(self.label_item_changed)

        Shape.line_color = self.line_color = QColor(DEFAULT_LINE_COLOR)
        Shape.fill_color = self.fill_color = QColor(DEFAULT_FILL_COLOR)
        self.display.canvas.setDrawingColor(self.line_color)
        self.zoomDisplay.canvas.setDrawingColor(self.line_color)
        #Advance
        # self.load_button.clicked.connect(self.openDicomDialog)
        self.display.canvas.currentPostion.connect(self.set_cursor)
        self.display.canvas.zoomPixmap.connect(self.jumpto)

        # self.tableFile.itemSelectionChanged.connect(self.load_from_table)
        self.tableFile.itemDoubleClicked.connect(self.load_from_table)
        self.lineEdit.returnPressed.connect(self.gotoClicked)

        self.edit_button.clicked[bool].connect(self.set_edit_mode)
        self.edit_button.setCheckable(True)
        self.add_button.clicked[bool].connect(self.set_create_mode)
        self.add_button.setCheckable(True)

        self.view_button.clicked[bool].connect(self.set_hidden_mode)
        self.view_button.setCheckable(True)
        self.reset_hidden_mode()

        self.segmentation_button.clicked.connect(self.click_segmentation_btn)
        self.undoSegment_button.clicked.connect(self.undo_shape_edit)
        self.resetSegment_button.clicked.connect(self.reset_segmentation)
        self.finishSegment_button.clicked.connect(self.finish_segment)

        # for talbe analysis
        self.table_analysis.itemClicked.connect(self.load_slice_from_table_analysis)
        self.table_analysis.delete_signal.connect(self.delete_item_analysis_table)
        self.table_analysis.edit_signal.connect(self.edit_item_analysis_table)
        self.table_analysis.color_signal.connect(self.change_color_analysis_table)
        self.table_analysis.selectionModel().selectionChanged.connect(self.on_selectionChange)

        self.slider.valueChanged.connect(self.slider_changed)
        self.log_button.clicked.connect(self.show_log)
        
        self.pathology_text.submit_signal.connect(self.savePlainText)

        self.confirm.clicked.connect(self.save_mask)
        self.save.clicked.connect(self.save_all_labels)

        self.measure.clicked[bool].connect(self.measure_distance)
        self.measure.setCheckable(True)
        self.refresh_button.clicked.connect(self.refresh_patient_table)

    def reset(self):
        self.table_analysis.clear()
        self.table_analysis.setEnabled(False)
        self.label_list.clear()
        self.listFileWidget.clear()
        self.pathology_text.setEnabled(False)
        self.slider.setStyleSheet(const.SLIDER_DEFAULT)
        self.slider.setEnabled(False)

    def refresh_patient_table(self):
        if self.last_loaded_dicom_db_time == None or self.last_loaded_dicom_db_time != os.path.getmtime(self.dicom_db_path):
            with database.DicomDatabaseAPI(self.dicom_db_path) as dbapi:
                self.history, self.loaded_path = refresh(self.history,
                                                        dbapi, 
                                                        path_dicom=const.PATH_DICOM, 
                                                        loaded_path=self.loaded_path)
            # Update loaded time
            self.last_loaded_dicom_db_time = os.path.getmtime(self.dicom_db_path)
            
            self.tableFile.clear()
            self.tableFile._addData(self.history)
            confirmed_counts = len(list(filter(lambda x:x["Confirmed"] != None, self.history)))
            self.tableFile.update_confirm_counts_header(confirmed_counts)
            # Recover the last sort
            sort_column = getattr(self.tableFile, "sortBy", 0)
            sort_reverse = getattr(self.tableFile, "sort_reverse", False)
            self.tableFile.sortByColumn(sort_column, Qt.SortOrder.DescendingOrder if sort_reverse else Qt.SortOrder.AscendingOrder)
            
        # Update Last Modified
        timestamp = get_timestamp()
        self.group_box_select.setTitle("1. Select a Patient (Modified at {})".format(timestamp))
        
    def load_recent(self, path):
        if self.may_continue():
            if os.path.isfile(path):
                if path.endswith(".mhd"):
                    self.importImages(path, loadMHD)
            else:
                self.importImages(path, load_images)

    # def current_item(self):
    #     items = self.label_list.selectedItems()
    #     if items:
    #         return items[0]
    #     return None

    def update_file_menu(self):
        curr_file_path = self.file_path
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recent_files if f !=
                 curr_file_path and os.path.exists(f)]
        for i, f in enumerate(files):
            icon = new_icon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.load_recent, f))
            menu.addAction(action)

    def close_file(self, _value=False):
        if not self.may_continue():
            return
        self.set_clean()
        self.toggle_actions(False)
        self.display.setEnabled(False)
        self.zoomDisplay.setEnabled(False)
        self.pathology_text.setEnabled(False)
        self.save.setEnabled(False)
        self.measure.setEnabled(False)
        self.segmentation_button.setEnabled(False)
        self.display.canvas.clear()
        self.zoomDisplay.canvas.clear()
        self.log_button.setEnabled(False)

    def may_continue(self):
        if not self.save_status:
            return True
        else:
            discard_changes = self.discard_changes_dialog()
            if discard_changes == QMessageBox.No:
                return True
            elif discard_changes == QMessageBox.Yes:
                self.save_all_labels()
                return True
            else:
                return False

    def discard_changes_dialog(self, message = 'The file has been modified, do you want to save the changes?'):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        msg = u'{}'.format(message)
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)            
    
    def closeEvent(self, a0:QCloseEvent):
        if not self.may_continue():
            a0.ignore()
        return super().closeEvent(a0)

    def errorMessage(self, title, message):
        """
        Show Warning Dialog
        """
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def inforMessage(self, title, message):
        """
        Show Infor Dialog
        """
        return QMessageBox.information(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))
    
    def status(self, message, delay=5000):
        """Show information in status bar
        """
        self.statusBar().showMessage(message, delay)

    def show_confirmed_status(self):
        confirmed_status = defaultdict(int)
        for series_info in self.history:
            user_name = series_info['Confirmed_User']
            if user_name != None:
                confirmed_status[user_name] += 1
                
        from libraries.log_dialog import ConfirmedStatusDialog
        dialog = ConfirmedStatusDialog(confirmed_status, parent=self)
        dialog.exec_()
    
    def keyReleaseEvent(self, event:QKeyEvent):
        # if event.key() == Qt.Key_Control:
        #     self.canvas.set_drawing_shape_to_square(False)
        return super().keyReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        # if event.key() == Qt.Key_Control:
        #     self.canvas.set_drawing_shape_to_square(True)
        return super().keyPressEvent(event)

    def no_shapes(self):
        return not self.items_to_shapes

    def set_dirty(self):
        self.dirty = True
        self.save_status = True
        self.actions.save.setEnabled(True)

        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.display.canvas.isShapeRestorable)
        
        # if self._config["auto_save"] or self.actions.saveAuto.isChecked():
        #     label_file = osp.splitext(self.imagePath)[0] + ".json"
        #     if self.output_dir:
        #         label_file_without_path = os.path.basename(label_file)
        #         label_file = os.path.join(self.output_dir, label_file_without_path)
        #     self.saveLabels(label_file)
        #     return
        # self.dirty = True
        # self.actions.save.setEnabled(True)
        # title = __appname__
        # if self.filename is not None:
        #     title = "{} - {}*".format(title, self.filename)
        # self.setWindowTitle(title)

    def set_clean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createPolyMode.setEnabled(True)
    
    def toggle_actions(self, value= True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            if z is not None:
                z.setEnabled(value)
        for action in self.actions.onLoadActive:
            if action is not None:
                action.setEnabled(value)

    def queue_event(self, function):
        QTimer.singleShot(0, function)

    def reset_status(self):
        #Basic
        self.items_to_shapes.clear()
        self.shapes_to_items.clear()
        self.label_list.clear()
        
        self.display.canvas.resetState()
        self.label_coordinates.clear()
        
        self.zoomDisplay.image = QImage()
        # Advance
        self.update_table = False
        self.actions.delete.setEnabled(False)

    def update_recent_file(self):
        self.recent_files = [value.get('Path', "") for value in self.history]

    def undo_shape_edit(self):
        if not self.undoSegment_button.isEnabled():
            return
        self.display.canvas.restoreShape()
        self.label_list.clear()
        self.loadShapes(self.display.shapes())
        self.actions.undo.setEnabled(self.display.canvas.isShapeRestorable)
        self.undoSegment_button.setEnabled(self.display.canvas.isShapeRestorable)
        self.segmentation()

    def toggle_drawing_sensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        self.actions.createMode.setEnabled(drawing)
        self.actions.viewMode.setEnabled(True)
        self.edit_button.setEnabled(not drawing)
        if not drawing:
            # Cancel creation.
            self.display.canvas.setEditing(True)
            self.display.canvas.restoreCursor()

            self.zoomDisplay.canvas.setEditing(True)
            self.zoomDisplay.canvas.restoreCursor()

    def toggle_draw_mode(self, edit=True, createMode = "polygon"):
        # Display
        self.display.canvas.setEditing(edit)
        self.display.canvas.createMode = createMode

        # Zoom Display
        self.zoomDisplay.canvas.setEditing(edit)
        self.zoomDisplay.canvas.createMode = createMode
        self.actions.viewMode.setEnabled(True)
        self.add_button.setEnabled(not edit)
        self.edit_button.setEnabled(edit)
        if edit == True:
            self.actions.createMode.setEnabled(edit)
            self.actions.createPolyMode.setEnabled(True)
            self.actions.createPointMode.setEnabled(True)
        else:
            self.actions.update.setEnabled(True)
            if createMode == "polygon":
                self.actions.createMode.setEnabled(True)
                self.actions.createPolyMode.setEnabled(False)
                self.actions.createPointMode.setEnabled(True)
            elif createMode == "rectangle":
                self.actions.createMode.setEnabled(False)
                self.actions.createPolyMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
            elif createMode == "line":
                self.actions.createMode.setEnabled(True)
                self.actions.createPolyMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
            elif createMode == "point":
                self.actions.createMode.setEnabled(True)
                self.actions.createPolyMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(False)
            elif createMode == "circle":
                self.actions.createMode.setEnabled(True)
                self.actions.createPolyMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
            elif createMode == "linestrip":
                self.actions.createMode.setEnabled(True)
                self.actions.createPolyMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
            else:
                raise ValueError("Unsupported createMode: %s" % createMode)
        self.actions.editMode.setEnabled(not edit)

    def toggle_view_mode(self, view=True):
        self.display.canvas.setViewing()
        self.zoomDisplay.canvas.setViewing()
        # self.actions.createMode.setEnabled(view)
        # self.actions.editMode.setEnabled(view)
        # self.actions.viewMode.setEnabled(not view)
        if (not self.add_button.isChecked()) and (not self.edit_button.isChecked()):
            self.add_button.setEnabled(view)
            self.edit_button.setEnabled(view)
        elif self.add_button.isChecked():
            self.toggle_draw_mode(False, 'polygon')
            self.add_button.setChecked(False)
        elif self.edit_button.isChecked():
            self.toggle_draw_mode(True)
            self.edit_button.setChecked(False)

    def toggle_segment_mode(self, enable: bool):
        self.undoSegment_button.setEnabled(enable)
        self.resetSegment_button.setEnabled(enable)
        self.finishSegment_button.setEnabled(enable)
        self.confirm.setEnabled(False)

    def set_create_mode(self, press=True):
        if press:
            # self.toggle_view_mode(True)
            self.toggle_draw_mode(False, "polygon")
            self.update_table = True
        else:
            self.toggle_view_mode()

    def set_edit_mode(self, press=True):
        if press:
            self.toggle_draw_mode(True, "rectangle")
            self.label_selection_changed()
            self.update_table = True
            self.set_dirty()
            self.edit_on += 1
        else:
            self.edit_on -= 1
            self.toggle_view_mode()

    def reset_hidden_mode(self):
        self.view_button.setIcon(new_icon('eye'))
        self.view_button.setChecked(False)
        self.hidden_on = False

    def set_hidden_mode(self, press=True):
        if press:
            if self.auto_saving.isChecked():
                if self.dirty is True:
                    self.save_label(self.current_slice, self.display.shapes())
            self.view_button.setIcon(new_icon('hidden'))
            self.hidden_on = True
            self.display.canvas.setAllShapeVisible(False)
            self.zoomDisplay.canvas.setAllShapeVisible(False)
        else:
            self.view_button.setIcon(new_icon('eye'))
            self.hidden_on = False
            if self.results_nodule and self.current_slice in self.results_nodule.keys():
                self.load_labels(list(self.results_nodule[self.current_slice]))
            self.display.canvas.setAllShapeVisible(True)
            self.zoomDisplay.canvas.setAllShapeVisible(True)

    def set_benign_hidden(self):
        if self.results_nodule == None or len(self.results_nodule) <= 0:
            return
        self.dirty = True
        self.update_analysis_table()

    def set_display_label(self):
        for shape in self.zoomDisplay.canvas.shapes:
            shape.paint_label = self.display_label_option.isChecked()
        self.zoomDisplay.canvas.update()

    def set_view_mode(self):
        self.toggle_view_mode(True)   

    def set_fit_window(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoom_mode = self.display.FIT_WINDOW if value else self.display.MANUAL_ZOOM
        self.display.adjust_scale()

    def set_fit_width(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoom_mode = self.display.FIT_WIDTH if value else self.display.MANUAL_ZOOM
        self.display.adjust_scale()
    
    """
    Processing Label List
    """
    def pop_label_list_menu(self, point):
        menu = self.menus.labelList
        menu.exec_(self.label_list.mapToGlobal(point))

    # React to canvas signals.
    def label_selection_changed(self):
        if self._no_selection_slot:
            return
        if self.display.canvas.editing() or self.zoomDisplay.canvas.editing():
            selected_shapes = []
            for item in self.label_list.selectedItems():
                selected_shapes.append(self.items_to_shapes[item])
            if selected_shapes:
                self.display.canvas.selectedShapes = self.zoomDisplay.canvas.selectedShapes = selected_shapes

            else:
                self.display.canvas.deSelectShape()
                self.zoomDisplay.canvas.deSelectShape()
        
    
    def shape_selection_changed(self, selected_shapes:list):
        self._no_selection_slot = True
        for shape in self.display.canvas.selectedShapes:
            try:
                self.shapes_to_items[shape].setSelected(True)
            except Exception as e:
                print(e)
        self.label_list.clearSelection()
        self.display.canvas.selectedShapes = selected_shapes
        self.zoomDisplay.canvas.selectedShapes = selected_shapes
        # for shape in self.display.canvas.selectedShapes:
        #     self.shapes_to_items[shape].setSelected(True)
            # shape.selected = True
            # item = self.label_list.findItemByShape(shape)
            # self.label_list.selectItem(item)
            # self.label_list.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        if self.dirty:
            self.actions.update.setEnabled(True)

    def label_item_changed(self, item):
        shape = self.items_to_shapes[item]
        label = item.text()
        f = item.font()
        # f.setStrikeOut(not checked)
        item.setFont(f)
        item.setBackground(shape.line_color)
        shape.checked = True
        self.items_to_shapes[item] = shape
        self.shapes_to_items[shape] = item
        self.set_dirty()
        if shape.group_id is None:
            label_name = shape.label
        else:
            label_name = "{}-{:02d}".format(shape.label.split('-')[0], shape.group_id)
        if label != label_name:
            shape.label = item.text()
            # shape.line_color = generate_color_by_text(shape.label)
            shape.lineColor = const.ColorBBox
            self.set_dirty()
        else:  # User probably changed item visibility
            self.display.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
            self.zoomDisplay.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def loadShapes(self, shapes, replace = True):
        self._no_selection_slot = True
        self.label_list.clearSelection()
        if replace:
            self.label_list.clear()
        for shape in shapes:
            self.add_label(shape)
        
        self._no_selection_slot = False
        self.display.update_shape(shapes, replace=replace)
        self.zoomDisplay.update_shape(shapes, replace)

    def change_shape_point_size(self):
        def point_size_update():
            text = box.currentText()
            self.point_size = int(text)
        Form = QWidget()
        Form.setWindowTitle('oxxo.studio')
        Form.resize(300, 300)

        box = QComboBox(Form)   # 加入下拉選單
        box.addItems(['4','6','8','12'])   # 加入四個選項
        box.setGeometry(10, 10, 200, 30)
        box.currentIndexChanged.connect(point_size_update)
        Form.show()
        
    def add_label(self, shape:Shape):
        shape.paint_label = self.display_label_option.isChecked()
        if shape.shape_type == 'point' or shape.shape_type == 'line':
            return
        if shape.group_id is None:
            label_name = shape.label
        else:
            label_name = "{}-{:02d}".format(shape.label, shape.group_id)
        item = HashableQListWidgetItem(label_name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        # item.setCheckState(Qt.Checked)
        # item.setBackground(generate_color_by_text(shape.label))
        checked = shape.checked
        f = item.font()
        f.setStrikeOut(not checked)
        item.setFont(f)
        item.setBackground(shape.line_color)
        if not self.benign_hidden.isChecked() or shape.category > 0:
            self.items_to_shapes[item] = shape
            self.shapes_to_items[shape] = item
            self.label_list.addItem(item)
        for action in self.actions.onShapesPresent:
            if action is not None:
                action.setEnabled(True)
        self.display.update()
        self.zoomDisplay.update()

    def remove_label(self, shape):
        if shape is None or len(shape) == 0:
            return
        item = self.shapes_to_items[shape]
        del self.shapes_to_items[shape]
        del self.items_to_shapes[item]

    def setDefautShape(self, label, points, shape_type, 
                    category=0, group_id=None, conf=1.0, checked=True, rect= None, mask = None):
        def polygon2rect(ps):
            rect = ps.boundingRect()
            x1 = rect.x()
            y1 = rect.y()
            x2 = x1 + rect.width()
            y2 = y1 + rect.height()
            return (x1,y1,x2,y2)

        shape = Shape(label= label,
                    shape_type = shape_type,
                    group_id = group_id,
                    category = category, 
                    conf = conf,
                    checked = checked,
                    rect = rect,
                    mask = mask,
                    point_size = self.point_size)
        for x, y in points:
            # Ensure the labels are within the bounds of the image. If not, fix them.
            x, y, snapped = self.display.canvas.snapPointToCanvas(x, y)
            if snapped:
                self.set_dirty()

            shape.addPoint(QPointF(x, y))
        shape.close()
        line_color = const.id_color_nodule[category] if group_id is not None else const.id_color_nodule[4]
        fill_color = const.id_color_nodule_fill[category] if group_id is not None else const.id_color_nodule_fill[4]
        
        shape.line_color = QColor(*line_color) if line_color else const.ColorBBox
        shape.fill_color = QColor(*fill_color) if fill_color else const.ColorBBox
            
        if rect is None:
            rect = polygon2rect(shape)
            shape.rect = rect
        if shape_type == 'polygon' and mask is None:
            mask = cv2.fillConvexPoly(np.zeros(self.mImgSize + [3]).astype(np.uint8),
                                        np.array(points, dtype=np.int32),
                                        (255,255,255))
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        return shape

    def load_labels(self, shapes:list):
        self.s = []
        ds = []
        # label, points, line_color, fill_color, conf, checked, category, noduleID
        self.items_to_shapes.clear()
        self.shapes_to_items.clear()
        self.label_list.clear()
        for sDict in shapes:
            if isinstance(sDict, dict):
                points = sDict['points']
                shape = self.setDefautShape(label=sDict['label'],
                                    points = points,
                                    shape_type=sDict['shape_type'],
                                    category=sDict['category'],
                                    group_id=sDict['group_id'],
                                    conf=sDict['conf'],
                                    checked=sDict['checked'],
                                    rect=sDict['rect'],
                                    mask=sDict['mask'] if 'mask' in sDict.keys() else None
                                    )
                if not points:
                    continue
                self.add_label(shape)
                self.s.append(shape)
                if not (self.benign_hidden.isChecked() and sDict['category'] == 0 and sDict['shape_type'] == 'rectangle'):                        
                    ds.append(shape)
        if len(self.s) > 0:
            if self.zoom_x is None and self.zoom_y is None:
                s0 = self.s[0].rect
                x_center = (s0[0] + s0[2])//2
                y_center = (s0[1] + s0[3])//2
                if not self.view_lock:
                    self.zoom_area(x_center, y_center)
        self.display.update_shape(ds)
        self.zoomDisplay.update_shape(ds)
        # In hidden mode, after load labels, set all shapes invisible.
        if self.hidden_on:
            self.display.canvas.setAllShapeVisible(False)
            self.zoomDisplay.canvas.setAllShapeVisible(False)

    def save_label(self, current_slice: int, display_shapes: List[Shape], add_new_shape = False):
        """
        Args:
            current_slice: int
                current slice number
            display_shapes: List[Shape]
                all shapes in current slice
            add_new_shape: bool
                only use in propagate, if cur_slice has nodule already, append new shape to it.
        """
        shapes = []
        nodule_id = []
        
        for shape in display_shapes:
            if shape.shape_type == 'point' or shape.shape_type == 'line':
                continue
            shapes.append(shape.shape2dict())
            nodule_id.append(shape.group_id)
        
        # Hide the benign nodules
        if self.benign_hidden.isChecked():
            for shape in self.s:
                if shape.category == 0 and shape.group_id not in nodule_id:
                    if shape.shape_type == 'point' or shape.shape_type == 'line':
                        continue
                    shapes.append(shape.shape2dict())
        
        if len(shapes) > 0:
            if self.results_nodule is None or len(self.results_nodule) <= 0:
                self.results_nodule = {}
            
            if current_slice not in self.results_nodule:
                self.results_nodule[current_slice] = []
                
            if not add_new_shape:
                self.results_nodule[current_slice] = shapes
            else: # only occur in propagate.
                if current_slice not in self.results_nodule.keys():
                    self.results_nodule[current_slice] = [shapes[0]]
                else:
                    self.results_nodule[current_slice].append(shapes[0])
                
    def save_all_labels(self):
        def saved_notify():
            ok = QMessageBox.Ok
            msg = u'The label is saved'
            return QMessageBox.information(self, u'Notify', msg, ok)
        if self.auto_saving.isChecked():
            if self.dirty is True:
                self.save_label(self.current_slice, self.display.shapes())   
        if self.results_nodule is None or len(self.results_nodule) <= 0:
            self.results_nodule = {}
        if self.label_file is None:
            self.label_file = Label_Save(self.save_dir)
            
        # Update Last Modified
        timestamp = get_timestamp(is_filename=True)
        file_name = f'log_{timestamp}_{self.user_name}'
        self.label_file.save_label_pickle(file_name, self.results_nodule)
        self.save_status = False
        saved_notify()
        self.confirm.setEnabled(True)

    def save_mask(self):
        def saved_notify():
            ok = QMessageBox.Ok
            msg = u'The nodule mask is saved'
            return QMessageBox.information(self, u'Notify', msg, ok)
        nodule_mask = np.zeros([self.img_count] + self.mImgSize).astype(np.uint8)
        blank_mask = np.zeros(self.mImgSize + [3]).astype(np.uint8)
        if self.results_nodule is not None:
            for key, values in self.results_nodule.items():
                for value in values:
                    if value['shape_type'] == 'polygon':
                        mask3 = cv2.fillConvexPoly(blank_mask.copy(),
                                                    np.array(value['points'],dtype=np.int32),
                                                    (255,255,255))
                        mask = cv2.cvtColor(mask3,cv2.COLOR_BGR2GRAY)
                        nodule_mask[key-1] = np.bitwise_or(nodule_mask[key-1],mask.copy())
        mask_filename, split_dirname = database.gen_dicom_file_name_from_path(self.dirname)
        nodule_mask = np.transpose(nodule_mask, (1,2,0))
        
        from libraries.inference import resample_ct_scan
        checked, resampled_mask = resample_ct_scan.resample_mask(self.dirname, 
                                                                np.flip(nodule_mask,-1),
                                                                np.array(self.patient_infor["Spacing"]))
        if checked:
            if self.label_file is None:
                self.label_file = Label_Save(self.save_dir)
            self.label_file.save_mask_npz('raw_' + mask_filename, nodule_mask)
            self.label_file.save_mask_npz(mask_filename, resampled_mask)
            saved_notify()
            with database.DicomDatabaseAPI(self.dicom_db_path) as db_api:
                series_id = db_api.get_series_id_by_folder_info(*[int(d) for d in split_dirname])
                db_api.update_is_relabel(series_id, True)
                db_api.update_relabel_user(series_id, self.user_name)
            
            # Update loaded time
            self.last_loaded_dicom_db_time = os.path.getmtime(self.dicom_db_path)
            
            # Update the confirmed status of current patient
            self.tableFile.update_one_row_confirm_status(self.dirname, self.user_name)
            for i in range(len(self.history)):
                if self.history[i]["Path"] == self.dirname:
                    self.history[i]["Confirmed"] = 'V'
                    self.history[i]["Confirmed_User"] = self.user_name
                    break
            # Update the confirmed counts
            confirmed_counts = len(list(filter(lambda x: x["Confirmed"] == 'V', self.history)))
            self.tableFile.update_confirm_counts_header(confirmed_counts)
        else:
            self.errorMessage("Resample mask failed")
            
    # Callback functions:
    def new_shape(self):
        """Pop-up and give focus to the label editor.
        position MUST be in global coordinates.
        """
        flags = None
        group_id = None
        text = "nodule"
        if text is not None:
            self.prev_label_text = text
            generate_color = const.ColorBBox #generate_color_by_text(text)
            shape = self.display.canvas.setLastLabel(text, flags, generate_color, generate_color)
            shape.group_id = group_id
            self.add_label(shape)
            self.display.canvas.setEditing(False)
            self.zoomDisplay.update_shape([shape], replace=False)
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(True)
            self.set_dirty()
        else:
            self.display.canvas.undoLastLine()
            # self.canvas.reset_all_lines()
            self.display.canvas.shapesBackups.pop()

    def delete_selected_shape(self):
        item = self.label_list.currentItem()
        row = self.label_list.row(item)
        if item is None or row == -1:
            return
        try:
            nid = int(item.data(0).split('-')[1])
        except:
            nid = -1
        check = 0
        if self.results_nodule != None and self.current_slice in self.results_nodule.keys():
            if self.current_slice+1 in self.results_nodule.keys():
                for nodule in self.results_nodule[self.current_slice+1]:
                    if nodule['group_id'] == nid:
                        check = 1
            if self.current_slice-1 in self.results_nodule.keys() and check:
                for nodule in self.results_nodule[self.current_slice-1]:
                    if nodule['group_id'] == nid:
                        return
            # update the nodule in self.results_nodule
            nodule_analysis = self.results_nodule[self.current_slice]
            nodule_analysis = list(filter(lambda x: x['group_id'] != nid, nodule_analysis))
            self.results_nodule[self.current_slice] = nodule_analysis
        self.label_list.takeItem(row)
        shape = self.items_to_shapes[item]

        self.display.canvas.deleteShape(shape)
        if shape.checked == False:
            return
        self.remove_label(shape)
        self.display.canvas.setShapeVisible(shape, False)
        self.zoomDisplay.canvas.setShapeVisible(shape, False)
        self.label_list.repaint()
        self.set_dirty()
        if self.no_shapes():
            for action in self.actions.onShapesPresent:
                if action is not None:
                    action.setEnabled(False)
        
        self.update_analysis_table()
    "Basic Function"
    
    def openFileMetaImage(self, _value=False):
        if not self.may_continue():
            return
        path = os.path.dirname(ustr(self.file_path)) if self.file_path else '.'
        filters = "MetaImage files (*.mhd)"
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
                if not os.path.exists(os.path.splitext(filename)[0] + ".raw"):
                    # self.allEnable(False)
                    return
                self.importImages(filename, loadMHD)

    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.may_continue():
            return
        targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                        '%s - Open Directory Images' % __appname__, const.PATH_DICOM,
                                                        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        self.importImages(targetDirPath, load_images)

    def openDicomDialog(self, _value=False, dirpath=None, silent=False):
        if not self.may_continue():
            return
        self.resizeEvent(None)
        targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                        '%s - Open Directory DICOM' % __appname__, const.PATH_DICOM,
                                                        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        self.importImages(targetDirPath, load_Dicom)

    def importImages(self, dirpath, load_function):

        def errorLoading(self):
            self.patient_infor = None
            self.errorMessage(u'Error opening file',
                                    u"<p>Make sure <i></i> is a valid image file.")
            self.status("Error reading")
            loading.stopAnimation()

        self._reset_params()
        if not dirpath:
            # self.allEnable(False)
            return
        if os.path.isfile(dirpath):
            self.dirname = os.path.dirname(dirpath)
        else:
            self.dirname = dirpath
            
        dirname = os.path.relpath(self.dirname, os.path.abspath(const.PATH_DICOM))
        self.path_log = os.path.join(os.path.abspath(const.PATH_DICOM), dirname)
        self.save_dir = self.path_log
        self.listFileWidget.clear()
        self.table_analysis.clear()
        self.pathology_text.reset_status()
        self.slider.setStyleSheet(const.SLIDER_DEFAULT)
        # self.slider.setEnabled(False)
        try:
            loading = loadingDialog(self)
            loading.setText("Loading File")
            loading.startAnimation()
            QApplication.processEvents() 
            preprocessing = load_function(dirpath)
            # self.patient_infor = preprocessing.get_patient_infor()
            
            objThreading = QThread()
            preprocessing.imageDataNList.connect(self.getImgData)
            preprocessing.imageDataDict.connect(self.getImageDict)
            preprocessing.errorSignal.connect(lambda :errorLoading(self))
            preprocessing.moveToThread(objThreading)
            preprocessing.finished.connect(objThreading.quit)
            objThreading.started.connect(preprocessing.processing)
            objThreading.finished.connect(loading.stopAnimation)
            objThreading.start()
            
            self.setEnabled(False)
            while objThreading.isRunning():
                QApplication.processEvents()
            self.patient_infor = preprocessing.get_patient_infor()
        except: 
            self.errorMessage(u'Error opening file',
                                u"<p>Make sure <i></i> is a valid image file.")
            self.status("Error reading")
            loading.stopAnimation()
        # self.load_file(self.image_data[0], 0)
        self.setEnabled(True)
        if self.patient_infor is not None:
            self.listFileWidget._addData(self.patient_infor)
            self.set_clean()
            self.show_display()
        self.reset_hidden_mode()
        self.group_box_lung_nodule.setTitle("Total Lung Nodules")
        self.confirm.setEnabled(False)

    def getImgData(self, imgData, mImgList, mImgSize):
        """
        Connect from Thread to UI
        """
        self.image_data = imgData
        self.mImgList = mImgList
        self.img_count = len(mImgList)
        self.mImgSize = mImgSize
    
    def getImageDict(self, imageDataDict, spacing):
        self.image_data_dict = imageDataDict
        self.current_slice = list(imageDataDict.keys())[0]
        self.display.canvas.spacing = spacing
        self.zoomDisplay.canvas.spacing = spacing
        self.spacing = spacing

    def show_display(self):
        self.image_data_backup = self.image_data_dict.copy()
        self.data_size = len(self.image_data_dict.items())

        if self.data_size is not None:
            self.TotalImage.setText("/{} images".format(self.data_size))
            list_slice = list(self.image_data_dict.keys())
            self.slider.setMinimum(list_slice[0])
            self.slider.setMaximum(list_slice[-1])
        if self.current_slice is not None:    
            self.load_file(self.image_data_dict[self.current_slice]['data'], 
                           self.image_data_dict[self.current_slice]['path'],
                           self.image_data_dict[self.current_slice]['mode'])
            self.zoom_area(256,256)
        
        self.wheel_scroll = True
        self.keyboard_input = True
        self.actions.previous.setEnabled(True)
        self.actions.next.setEnabled(True)
        # self.inference_button.setEnabled(True)
        self.log_button.setEnabled(True)
        self.segmentation_button.setEnabled(False)
        self.slider.setEnabled(True)
        # self.toggle_segment_mode(True)
        # self.pathology_text.setEnabled(True)
    """
    Processing in Canvas
    """
    def load_file(self, imageData = None, filename = None, mode = False):
        "Use existing data to display on the application"
        zoom_value = [self.zoom_x, self.zoom_y]
        self.reset_status()
        if filename is not None:
            self.status("Loaded %s" % filename)
        status, image = self.display.load_image(imageData)
        if status:
            self.zoomDisplay.load_pixmap(image)
            self.toggle_actions(True)
            index = list(self.image_data_dict.keys()).index(self.current_slice)
            self.lineEdit.setText(str(index + 1))
            if self.results_nodule is not None and self.current_slice in self.results_nodule.keys():
                label_shape = list(self.results_nodule[self.current_slice])
                # self.load_labels(bbox_to_shape(label_shape))
                self.load_labels(label_shape)
            if mode:
                self.toggle_draw_mode(mode)
            else:
                self.toggle_view_mode(not mode)
            self.display.canvas.repaint()
            self.zoomDisplay.canvas.repaint()
        self.zoom_x, self.zoom_y = zoom_value
        
    # def resizeEvent(self, event):
    #     if self.canvas and not self.image.isNull() and self.zoom_mode != self.MANUAL_ZOOM:
    #         self.adjust_scale()
    #     super(MainWindow, self).resizeEvent(event)

    def zoom_area(self, x_origin, y_origin):
        x_zoom = x_origin * self.zoomDisplay.canvas.scale + self.zoomDisplay.canvas.offsetToCenter().x()
        y_zoom = y_origin * self.zoomDisplay.canvas.scale + self.zoomDisplay.canvas.offsetToCenter().y()

        self.zoomDisplay.setScroll(Qt.Horizontal, x_zoom - self.zoomDisplay.scroll_bars[Qt.Horizontal].pageStep()//2)
        self.zoomDisplay.setScroll(Qt.Vertical, y_zoom - self.zoomDisplay.scroll_bars[Qt.Vertical].pageStep()//2)
        pass

    def jumpto(self, x_origin, y_origin, lock=False):
        if lock:
            self.view_lock = lock
        self.zoom_x = x_origin
        self.zoom_y = y_origin
        self.zoom_area(x_origin, y_origin)

    def zoom_create(self, shapes):
        self.display.update_shape(shapes)
        self.set_dirty()

    def set_cursor(self, cursor_x, cursor_y):
        text = "[{}, {}]".format(cursor_x, cursor_y)
        self.cursor.setText(text)

    def show_log(self):
        Log_Dir = os.path.join(self.path_log, 'inference')
        Log_Dialog = FileSystemView(Log_Dir, self)
        Log_Dialog.dict_label_signal.connect(self._get_log_result_nodule)
        
        Log_Dialog.exec_()
        pass
    
    def _get_log_result_nodule(self, dict_nodule):
        self.slider.setStyleSheet(const.SLIDER_DEFAULT)
        self.results_nodule = dict_nodule
        # self.set_dirty()
        self.dirty = False
        self.actions.save.setEnabled(True)
        self.update_analysis_table()
        self.segmentation_button.setEnabled(True)
        self.confirm.setEnabled(True)
        self.current_slice = 1
        self.set_slider()
    """
    Process in navigate area
    """

    def open_next_image(self, _value=False):
        if self.auto_saving.isChecked():
            if self.dirty is True:
                self.save_label(self.current_slice, self.display.shapes())
        if self.image_data_dict is None or self.data_size is None or self.data_size <=0:
            return

        imageData = self.image_data_dict
        if self.current_slice + 1 in imageData.keys():
            self.current_slice += 1
            self.set_slider()

    def open_prev_image(self, _value=False):
        if self.auto_saving.isChecked():
            if self.dirty is True:
                self.save_label(self.current_slice, self.display.shapes())
        if self.image_data_dict is None or self.data_size is None or self.data_size <=0:
            return

        imageData = self.image_data_dict
        if self.current_slice - 1 in imageData.keys():
            self.current_slice -= 1
            self.set_slider()

    def wheelEvent(self, event):
        if self.wheel_scroll:
            y = event.angleDelta().y()
            if self.current_slice in self.image_data_dict.keys():
                if y < 0:
                    self.open_next_image()
                elif y > 0:
                    self.open_prev_image()

    def gotoClicked(self):
        if self.auto_saving.isChecked():
            if self.dirty is True:
                self.save_label(self.current_slice, self.display.shapes())

        image_data = self.image_data_dict
        if self.data_size is not None and self.data_size > 0:
            line = int(self.lineEdit.text()) - 1
            if int(line) > self.data_size:
                num = self.data_size - 1
            elif int(line) <= 1:
                num = 0
            else:
                num = int(line)
            num_slice = list(image_data.keys())[num]
            self.current_slice = num_slice
            self.set_slider()

    """
    Process Slider
    """
    def set_slider(self):
        curr_slice = self.current_slice
        self.slider.setValue(curr_slice)
        pass

    def slider_changed(self):
        value = self.slider.value()
        image_data = self.image_data_dict
        if self.data_size is not None and self.data_size > 0:
            if value in image_data.keys():
                self.current_slice = value
                self.load_file(image_data[self.current_slice]['data'],
                               image_data[self.current_slice]['path'],
                               image_data[self.current_slice]['mode'])
                if self.measure.isChecked():
                    self.measure_distance(True)
        pass

    def update_groove_color(self):
        if self.results_nodule_analysis is not None and len(self.results_nodule_analysis) > 0:
            
            class_color = const.class_color_nodule
            groove_color_range = 'stop:0 black'
            len_dataset = self.data_size
            for item in self.results_nodule_analysis:
                if self.benign_hidden.isChecked() and item['Category'] == 'Benign' and item['data'][0][7]['shape_type'] == 'rectangle':
                    continue
                category = item['Category']
                start = item['Slice_range'][0]
                end = item['Slice_range'][-1]
                color = class_color[category]
                groove_color_range += ', stop:{:.5f} black, stop:{:.5f} {}, stop:{:.5f} {}, \
                    stop:{:.5f} black'.format(((start-1)/len_dataset), (start/len_dataset), \
                    color, (end/len_dataset), color, ((end+1)/len_dataset))
                # groove_color_range += ', start:{} {}, stop:{} {},'.format((start/len_dataset), color, (end/len_dataset), color)
            groove_color_range += ', stop:1 black'
            self.slider.setStyleSheet(
                "QSlider::groove:horizontal {\
                border: 1px solid #999999;\
                height: 20px; \
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, "+groove_color_range+");\
                }"
            )
        else:
            self.slider.setStyleSheet(const.SLIDER_DEFAULT)
            pass

    """
    Plain Text for pathology
    """
    def savePlainText(self, text):
        nodule_infor = self.pathology_text.content
        try:
            index = self.results_nodule_analysis.index(nodule_infor)
            self.results_nodule_analysis[index]['description'] = text
            noduleID = nodule_infor['NoduleID']
            range_slice = nodule_infor['Slice_range']
            for slice in range_slice:
                for ns in self.results_nodule[slice]:
                    if ns['group_id'] == noduleID:
                        ns['description'] = text
        except:
            # self.errorMessage("error",)
            pass

        self.pathology_text.save_flags = False
        pass

    def may_continue_text(self):
        if not self.pathology_text.save_flags:
            return True
        else:
            discard_changes = self.discard_changes_dialog('The pathology has been modified, do you want to save the changes?')
            if discard_changes == QMessageBox.No:
                self.pathology_text.save_flags = False
                return True
            elif discard_changes == QMessageBox.Yes:
                self.savePlainText(self.pathology_text.pathology_textEdit.document().toPlainText())
                return True
            else:
                return False
    """
    Process Table and List
    """
    def search(self, string):
        self.tableFile.setCurrentItem(None)
        if not string:
            return
        matching_items = self.tableFile.findItems(string, Qt.MatchContains)
        if matching_items:
            for item in matching_items:
                item.setSelected(True)
            first_item = matching_items[0]
            self.tableFile.scrollToItem(first_item)

    def load_from_table(self):
        def convert_inference_3d(inference_3d_path):
            uid = self.image_data_dict[self.current_slice]['path'].split('-')[0]

            converted_inference_results = ''

            with open(inference_3d_path, 'r') as f:
                for line in f.readlines():
                    line_split = line.split(' ')

                    line_split[0] = uid + '-' + line_split[0].zfill(4) + ' '

                    for ls in line_split:
                        converted_inference_results += ls

            with open(inference_3d_path.replace('inference_3d', 'inference_FP_reduction'), 'w') as f:
                f.write(converted_inference_results)

        if not self.may_continue():
            return
        items = self.tableFile.selectedItems()
        try:
            if items != []:
                position = items[0].row()
                path = items[-2].text()
                style = items[-1].text()
                self.path_log = path
                if path is None:
                    return
                dir = os.path.dirname(path)
                if os.path.exists(dir):
                    if os.path.isfile(path):
                        if path.endswith(".mhd"):
                            self.importImages(path, loadMHD)
                    else:
                        if style == "DICOM":
                            self.importImages(path, load_Dicom)
                        else:
                            self.importImages(path, load_images)
                    
                    Log = glob.glob(os.path.join(os.path.join(path, 'log'), "log*.log"))

                    if len(Log) > 0:
                        self._get_log_result_nodule(pck.load_dict(Log[-1]))
                    ########## modified by Ben ##########
                    else:
                        if os.path.exists(os.path.join(path, 'inference_FP_reduction.txt')):
                            self.getNoduleAnalysis(os.path.join(path, 'inference_FP_reduction.txt'))
                        elif os.path.exists(os.path.join(path, 'inference_3d.txt')):
                            convert_inference_3d(os.path.join(path, 'inference_3d.txt'))
                            self.getNoduleAnalysis(os.path.join(path, 'inference_FP_reduction.txt'))

                else:
                    self.tableFile._removeRow_index(position)
                self.save.setEnabled(True)
                self.measure.setEnabled(True)
        except:
            self.tableFile._removeRow_index(position)
            self.errorMessage(u'Error opening file',
                                u"<p>Make sure <i></i> is a valid image file.")

    def load_slice_from_table_analysis(self):
        if not self.may_continue_text():
            return
        if self.auto_saving.isChecked():
            if self.dirty is True:
                self.save_label(self.current_slice, self.display.shapes())
        items = self.table_analysis.selectedItems()
        self.view_lock = False
        if items != []:
            item = items[1].text().split('-')
            noduleId = int(items[0].text())
            for nodule_infor in self.results_nodule_analysis:
                if nodule_infor['NoduleID'] == noduleId:
                    self.pathology_text.setText(nodule_infor)
                    data = nodule_infor['data'][0][1:5]
                    x_cen = int((data[0] + data[2])//2)
                    y_cen = int((data[1] + data[3])//2)
                    self.zoom_x = x_cen
                    self.zoom_y = y_cen
                    continue
            curr_slice = int(float(item[0]))
            imageData = self.image_data_dict
            if curr_slice in imageData.keys():
                self.current_slice = curr_slice
                # self.load_file(imageData[self.current_slice]['data'], 
                #                 imageData[self.current_slice]['path'],
                #                 imageData[self.current_slice]['mode'])
                self.jumpto(self.zoom_x, self.zoom_y)
                self.set_slider()
            self.pathology_text.setEnabled(True)
            # pass
    
    def on_selectionChange(self, selected, deselected):
        if not self.may_continue_text():
            return
        if len(selected):
            self.pathology_text.setEnabled(True)
        else:
            self.pathology_text.setEnabled(False)
                
    def update_analysis_table(self):
        if self.auto_saving.isChecked() and self.dirty:
            self.save_label(self.current_slice, self.display.shapes())
            
        self.toggle_view_mode()
        if self.results_nodule is None or len(self.results_nodule) <= 0:
            return
        
        patient_ID = self.patient_infor['PatientID']
        results_nodule = sorted(self.results_nodule.items())
        dict_patient = {patient_ID : dict(results_nodule)}
        patient_tracking = olap.collect3d(dict_patient)
        self.results_nodule_analysis, new_dict_patient = olap.patientAnalysis(patient_tracking, self.spacing)
        self.results_nodule = list(new_dict_patient.values())[0]
        self.table_analysis.clear()
        if self.results_nodule_analysis and len(self.results_nodule_analysis) > 0:
            if self.benign_hidden.isChecked():
                self.table_analysis._addData(list(filter(lambda a: a["Category"] != "Benign" or a["data"][0][7]["shape_type"] == "polygon", self.results_nodule_analysis)))
            else:
                self.table_analysis._addData(self.results_nodule_analysis)
            self.table_analysis.setEnabled(True)
            self.update_groove_color()
        if not self.edit_on:
            self.dirty = False
        self.group_box_lung_nodule.setTitle("Total Lung Nodules:{:5d}".format(self.table_analysis.rowCount()))
        if self.current_slice is not None:    
            self.load_file(self.image_data_dict[self.current_slice]['data'], 
                            self.image_data_dict[self.current_slice]['path'],
                            self.image_data_dict[self.current_slice]['mode'])
    def delete_item_analysis_table(self, id_no:int, list_remove:list, list_analysis:list):
        if len(list_remove) > 0:
            for remove_key in list_remove:
                remove_key = int(remove_key)
                if remove_key in self.results_nodule.keys():
                    if len(self.results_nodule[remove_key]) <=1:
                        del self.results_nodule[remove_key]
                    else:
                        for result in self.results_nodule[remove_key]:
                            if result['group_id'] == id_no:
                                self.results_nodule[remove_key].remove(result)
        
        self.results_nodule_analysis = list_analysis
        self.group_box_lung_nodule.setTitle("Total Lung Nodules:{:5d}".format(self.table_analysis.rowCount()))
        self.update_groove_color()
        self.update()
        self.load_file(self.image_data_dict[self.current_slice]['data'], 
                        self.image_data_dict[self.current_slice]['path'],
                        self.image_data_dict[self.current_slice]['mode'])
        pass

    def edit_item_analysis_table(self, edit_mode, list_slice):
        if edit_mode:
            self.edit_on += 1
            # self.segmentation_button.setEnabled(False)
        else:
            self.edit_on -= 1
            # if self.edit_on == 0:
            #     self.segmentation_button.setEnabled(True)
        for slice in list_slice:
            self.image_data_dict[slice]['mode'] = edit_mode
        self.update()
        if self.image_data_dict[self.current_slice]['mode']:
            self.toggle_draw_mode()
        else:
            self.toggle_view_mode()
        self.set_dirty()

    def change_color_analysis_table(self, index, nodule, noduleIndex):
        self.results_nodule_analysis[index] = nodule
        slice_list = nodule['Slice_range']
        idno = nodule['NoduleID']
        if len(slice_list):
            for key in slice_list:
                key = int(key)
                if key in self.results_nodule.keys():
                    for noduleID in self.results_nodule[key]:
                        if noduleID['group_id'] == idno:
                            noduleID['category'] = noduleIndex
        self.update_groove_color()
        self.update()
        self.load_file(self.image_data_dict[self.current_slice]['data'], 
                        self.image_data_dict[self.current_slice]['path'],
                        self.image_data_dict[self.current_slice]['mode'])

    ########## modified by Ben ##########
    def getNoduleAnalysis(self, pathtxt):
        if pathtxt and os.path.exists(pathtxt):
        # dict_patient, dict_file = olap.merge_overlapping(olap.readtxt(pathtxt))
        
            dict_patient, dict_file = olap.follow_patient(olap.readtxt(pathtxt))
            # if len(list(dict_patient.values())[0]) <= 1:
            #     pathtxt = os.path.join(self.dirname, "inference.txt")
            #     dict_patient, dict_file = olap.follow_patient(olap.readtxt(pathtxt))
            patient_tracking = olap.collect3d(dict_patient)
            self.results_nodule_analysis, new_dict_patient = olap.patientAnalysis(patient_tracking, self.spacing)
            if new_dict_patient:
                self.results_nodule = list(new_dict_patient.values())[0]
                result_txt = olap.dict2txt(new_dict_patient, dict_file)
            else:
                self.results_nodule = {}
                result_txt = ""
                self.inforMessage("Not found", "Not find the nodule")

            #Save result into txt file
            with open(os.path.join(os.path.dirname(pathtxt), "inference_fp.txt"), "w+") as f:
                f.write(result_txt)
                f.close()
            self.result_path = os.path.join(os.path.dirname(pathtxt), "inference_fp.txt")
            # self.show_notif.setText("Detect Successfully")
        else:
            self.errorMessage("Error Detection", "Please run again {}".format(pathtxt))
        if self.results_nodule_analysis and len(self.results_nodule_analysis) > 0:
            self.table_analysis._addData(self.results_nodule_analysis)
            self.table_analysis.setEnabled(True)
            self.group_box_lung_nodule.setTitle("Total Lung Nodules:{:5d}".format(self.table_analysis.rowCount()))
            self.update_groove_color()
        if self.current_slice is not None:    
            self.load_file(self.image_data_dict[self.current_slice]['data'], 
                            self.image_data_dict[self.current_slice]['path'],
                            self.image_data_dict[self.current_slice]['mode'])
        self.save_status = True
        self.actions.save.setEnabled(True)
        self.setEnabled(True)

    """
    Inference
    """
    def inference(self, FPreduction = True):
        from libraries import inferenceThread
        def getDuration(duration_time):
            string_duration = "Processing Time:\t{:.1f} seconds".format(duration_time)
            self.inference_label.setText(string_duration)
            self.status(string_duration)

        def getNoduleAnalysis(pathtxt):
            # with open(pathtxt, 'r') as f:
            #     if len(f.readlines()) <= 1:
            #         pathtxt = os.path.join(self.dirname, 'inference.txt')
            if pathtxt and os.path.exists(pathtxt):
            # dict_patient, dict_file = olap.merge_overlapping(olap.readtxt(pathtxt))
            
                dict_patient, dict_file = olap.follow_patient(olap.readtxt(pathtxt))
                # if len(list(dict_patient.values())[0]) <= 1:
                #     pathtxt = os.path.join(self.dirname, "inference.txt")
                #     dict_patient, dict_file = olap.follow_patient(olap.readtxt(pathtxt))
                patient_tracking = olap.collect3d(dict_patient)
                self.results_nodule_analysis, new_dict_patient = olap.patientAnalysis(patient_tracking, self.spacing)
                if new_dict_patient:
                    self.results_nodule = list(new_dict_patient.values())[0]
                    result_txt = olap.dict2txt(new_dict_patient, dict_file)
                else:
                    self.results_nodule = {}
                    result_txt = ""
                    loading.close()
                    self.inforMessage("Not found", "Not find the nodule")
                

                #Save result into txt file
                with open(os.path.join(os.path.dirname(pathtxt), "inference_fp.txt"), "w+") as f:
                    f.write(result_txt)
                    f.close()
                self.result_path = os.path.join(os.path.dirname(pathtxt), "inference_fp.txt")
                loading.close()
                # self.show_notif.setText("Detect Successfully")
            else:
                loading.close()
                self.errorMessage("Error Detection", "Please run again {}".format(pathtxt))
            if self.results_nodule_analysis and len(self.results_nodule_analysis) > 0:
                self.table_analysis._addData(self.results_nodule_analysis)
                self.table_analysis.setEnabled(True)
                self.group_box_lung_nodule.setTitle("Total Lung Nodules:{:5d}".format(self.table_analysis.rowCount()))
                self.update_groove_color()
            if self.current_slice is not None:    
                self.load_file(self.image_data_dict[self.current_slice]['data'], 
                                self.image_data_dict[self.current_slice]['path'],
                                self.image_data_dict[self.current_slice]['mode'])
            self.save_status = True
            self.actions.save.setEnabled(True)
            self.setEnabled(True)
        if not self.dirname:
            self.errorMessage(u'Not Find Directory',
                                u"<p>Make sure <i></i> {} exists.".format(self.dirname))
            self.status("Error Inference")
            return False
        # self.show_notif.setText('2D Nodule Detection...')
        # loading = loadingDialog(self)
        # loading.setText("Detecting Nodule. Please wait....")
        # loading.startAnimation()
        loading = ProgressDialog(self)
        loading.setText("Detecting Nodule. Please wait....")
        loading.show()
        self.setEnabled(False)
        QApplication.processEvents()
        
        runInfer = inferenceThread(self.dirname)
        objThreading = QThread()
        runInfer.stringOutput.connect(getNoduleAnalysis)
        runInfer.duration.connect(getDuration)
        runInfer.processCount.connect(loading.setValue)
        runInfer.moveToThread(objThreading)
        runInfer.finished.connect(objThreading.quit)
        objThreading.started.connect(runInfer.run)
        objThreading.finished.connect(loading.close)
        objThreading.start()
        while objThreading.isRunning():
            self.setEnabled(False)
            QApplication.processEvents()

        self.setEnabled(True)
        return True

    def click_segmentation_btn(self):
        # Disable all buttons after first click
        self.toggle_draw_mode(False, createMode='point')
        self.actions.createMode.setEnabled(False)
        self.actions.createPolyMode.setEnabled(False)
        self.actions.editMode.setEnabled(False)
        self.actions.delete.setEnabled(False)
        self.actions.update.setEnabled(False)
        
        # Reset progress bar
        self.progress_bar.reset()
        self.progress_bar.setValue(0)
        
        # If the model is not loaded, load it
        if getattr(self, 'interactiveModel', None) is None:
            self.loading = loadingDialog(self)
            self.loading.setText("Loading the Interactive Model. Please wait....")
            self.loading.startAnimation()
            
            load_model_thread = LoadModelThread(self)
            load_model_thread.MODEL_LOADED_SIGNAL.connect(self.end_of_loading_model)
            load_model_thread.start()
            
            self.setEnabled(False)
            while load_model_thread.isRunning():
                QApplication.processEvents()
            
    def end_of_loading_model(self, loaded_results: tuple):
        if loaded_results[0] == None:
            self.errorMessage("Error Loading Model", "Please check again: {}".format(Exception))
        else:
            model, output_keys = loaded_results
            self.interactiveModel, self.output_keys = model, output_keys
            
        self.loading.stopAnimation()
        self.setEnabled(True)

    def segmentation(self):
        from libraries.interactiveSegment import predict
        shapes = self.display.shapes()
        seq_points = []
        
        # Get the points from the shapes
        for shape in shapes:
            if shape.shape_type == 'point':
                x = shape.points[0].x()
                y = shape.points[0].y()
                if shape.positive:
                    point = (x, y, 1)
                else:
                    point = (x, y, 0)
                seq_points.append(point)
        seq_points = np.array(seq_points, dtype = np.int64) # shape = (n, 3)
        
        if self.interactiveModel is not None and len(seq_points) > 0:
            try:
                self.toggle_segment_mode(True)
                last_click_coordinate = seq_points[-1, :2]
                pred = predict(self.interactiveModel, self.image_data_dict[self.current_slice]['data'], seq_points, self.output_keys, if_sis=True, if_cuda=False)
                self.pred_mask = pred
                merge = gene_merge(pred, self.image_data_dict[self.current_slice]['data'])
                
                image = QImage(merge.data, merge.shape[1], merge.shape[0], merge.shape[1]*3,  QImage.Format_RGB888)
                self.display.canvas.loadPixmap(QPixmap.fromImage(image), clear_shapes=False)
                self.zoomDisplay.canvas.loadPixmap(QPixmap.fromImage(image), clear_shapes=False)
                self.zoom_area(last_click_coordinate[0], last_click_coordinate[1])

            except:
                self.errorMessage("Error Predict", "{}".format(Exception))
        else:
            self.load_file(self.image_data_dict[self.current_slice]['data'], 
                            self.image_data_dict[self.current_slice]['path'],
                            self.image_data_dict[self.current_slice]['mode'])
            self.toggle_segment_mode(False)

    def reset_segmentation(self):
        shapes = self.display.shapes()
        
        # Delete points and lines
        self.display.canvas.shapes = [shape for shape in shapes if shape.shape_type != 'point' and shape.shape_type != 'line']
        self.load_file(self.image_data_dict[self.current_slice]['data'], 
                        self.image_data_dict[self.current_slice]['path'],
                        self.image_data_dict[self.current_slice]['mode'])
        self.toggle_segment_mode(False)
        self.click_segmentation_btn()
    
    def finish_segment(self):
        
        def overlap_with_rectangle(pred_mask: np.ndarray, rect: np.ndarray) -> bool:
            pred_mask = pred_mask[rect[1]:rect[3], rect[0]:rect[2]]
            return (pred_mask.sum() > 0)
        def overlap_with_polygon(pred_mask: np.ndarray, mask: np.ndarray) -> bool:
            return (np.logical_and(pred_mask, mask).sum() > 0)
        def fail_segmentation():
            self.errorMessage("Error Segmentation", f"None to save")
            # Delete points
            shapes = self.display.shapes()
            deleted_shape_ids = set()
            for shape_i, shape in enumerate(shapes):
                if shape.group_id == None:
                    deleted_shape_ids.add(shape_i)
            deleted_shape_ids = reversed(sorted(list(deleted_shape_ids)))
            for shape_i in deleted_shape_ids:
                del shapes[shape_i]
            self.loadShapes(shapes, replace=True)
            
            self.load_file(self.image_data_dict[self.current_slice]['data'], 
                            self.image_data_dict[self.current_slice]['path'],
                            self.image_data_dict[self.current_slice]['mode'])
            # Restore the settings        
            self.toggle_segment_mode(False)
            self.confirm.setEnabled(True)
            
        self.display.canvas.setViewing()
        self.zoomDisplay.canvas.setViewing()
        
        # if there is no nodule in the current slice, then return
        polygon, centers = get_polygon((self.pred_mask.astype(np.uint8) * 255))
        if polygon == None or centers == None:
            fail_segmentation()
            return
        
        # Start the loading animation
        loading = loadingDialog(self)
        loading.setWindowTitle("Propagating ...")
        loading.setText("Propagating the segmentation. Please wait....")
        loading.startAnimation()
        self.setEnabled(False)
        try:
            # Check if the segmentation in current slice overlap with the previous segmentation
            # (1) If overlap and the previous annotation is a rectangle, then delete the previous segmentation
            # (2) If overlap and the previous annotation is a polygon, then merge the two masks
            shapes = self.display.shapes()
            deleted_shape_ids = set()
            merged_mask = self.pred_mask.copy()
            for shape_i, shape in enumerate(shapes):
                if shape.group_id == None:
                    deleted_shape_ids.add(shape_i)
                
                elif shape.shape_type == SHAPETYPE.RECTANGLE and overlap_with_rectangle(self.pred_mask, np.array(shape.rect, dtype=np.int32)):
                    deleted_shape_ids.add(shape_i)
                    self.deleted_group_ids_after_propagation.add(shape.group_id)
                    
                elif shape.shape_type == SHAPETYPE.POLYGON:
                    polygon_points = [[point.x(), point.y()] for point in shape.points]
                    polygon_mask = points_to_mask(polygon_points, self.mImgSize + [3])
                    # if overlap, then merge the two masks
                    if overlap_with_polygon(merged_mask, polygon_mask):
                        merged_mask = np.logical_or(merged_mask, polygon_mask).astype(np.uint8)
                        deleted_shape_ids.add(shape_i)
            
            # If there are multiple nodules in the current slice, then split the merged mask into multiple masks
            if len(polygon) > 1 or len(centers) > 1:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(merged_mask.astype(np.uint8), 8)
                masks = []
                for i in range(1, num_labels):
                    nodule_mask = np.zeros_like(merged_mask, dtype=np.uint8)
                    nodule_mask[labels == i] = 1
                    masks.append(nodule_mask)
            else:
                masks = [merged_mask]
            
            is_any_success = False
            for i, mask in enumerate(masks):
                # Update UI
                loading.setText("Propagating the segmentation({}/{})".format(i+1, len(masks)))
                self.progress_bar.reset()
                self.progress_bar.setValue(0)
                
                # Get the polygon and center point of the mask
                polygon, centers = get_polygon(mask.astype(np.uint8) * 255)
                if len(polygon) == 0 or len(centers) == 0:
                    continue
                is_any_success = True
                # Add the new polygon shape to the canvas
                shape = self.setDefautShape(label = 'nodule',
                                            points = polygon[0],
                                            shape_type = SHAPETYPE.POLYGON,
                                            mask = mask.astype(np.uint8))
                shapes.append(shape)
                # Use the points on current slice to propagate the segmentation to the neighboring slices
                center_point = list(centers[0])
                center_point.append(1)
                seq_points = np.array([center_point], dtype=np.int64)
                self.propagate(mask, seq_points)
            
            if is_any_success:
                # Delete the overlapped shapes
                deleted_shape_ids = reversed(sorted(list(deleted_shape_ids)))
                for shape_i in deleted_shape_ids:
                    del shapes[shape_i]
                self.update_analysis_table()
                self.load_file(self.image_data_dict[self.current_slice]['data'], 
                                self.image_data_dict[self.current_slice]['path'],
                                self.image_data_dict[self.current_slice]['mode'])
                # self.loadShapes(shapes, replace=True)
            else:
                fail_segmentation()
        finally:
            self.toggle_segment_mode(False)
            self.confirm.setEnabled(True)
            
            loading.stopAnimation()
            self.setEnabled(True)
        
        if is_any_success:
            self.inforMessage("Propagate", "Propagate Finished")
            
    def clean_canvas(self):
        cur_image = self.image_data_dict[self.current_slice]['data']
        image = QImage(cur_image, cur_image.shape[1], cur_image.shape[0], cur_image.shape[1]*3,  QImage.Format_RGB888)
        self.display.canvas.loadPixmap(QPixmap.fromImage(image), clear_shapes=False)
        self.zoomDisplay.canvas.loadPixmap(QPixmap.fromImage(image), clear_shapes=False)
        self.display.canvas.setViewing()
        self.zoomDisplay.canvas.setViewing()
        self.display.canvas.setAllShapeVisible(False)
        self.zoomDisplay.canvas.setAllShapeVisible(False)

    def measure_distance(self, press: bool):
        if press:
            self.toggle_segment_mode(False)
            if self.results_nodule and self.current_slice in self.results_nodule.keys():
                nodule_mask = np.zeros(self.mImgSize + [3])
                for nodules in self.results_nodule[self.current_slice]:
                    if nodules['shape_type'] == 'rectangle':
                        continue
                    points = nodules['points']
                    nodule_mask += cv2.fillConvexPoly(np.zeros(self.mImgSize + [3]),
                                                      np.array(points ,dtype=np.int32),
                                                      (1,1,1))
                merge = gene_merge(nodule_mask[:,:,0], self.image_data_dict[self.current_slice]['data'])
                image = QImage(merge.data, merge.shape[1], merge.shape[0], merge.shape[1]*3,  QImage.Format_RGB888)
                self.display.canvas.loadPixmap(QPixmap.fromImage(image), clear_shapes=False)
                self.zoomDisplay.canvas.loadPixmap(QPixmap.fromImage(image), clear_shapes=False)
            self.display.canvas.setAllShapeVisible(False)
            self.toggle_draw_mode(False, createMode="line")
            self.zoomDisplay.canvas.setAllShapeVisible(False)
        else:
            if self.results_nodule != None and self.current_slice in self.results_nodule.keys():
                self.load_file(self.image_data_dict[self.current_slice]['data'], 
                               self.image_data_dict[self.current_slice]['path'],
                               self.image_data_dict[self.current_slice]['mode'])
            shapes = self.display.shapes()
            shapes = [shape for shape in shapes if shape.shape_type != 'point' and shape.shape_type != 'line']
            self.loadShapes(shapes, replace=True)
            self.toggle_view_mode(True)
            self.update()

    def propagate(self, pred_mask: np.ndarray, seq_points: np.ndarray):
        """Propagate the segmentation to the neighboring slices.
        
        Args:
            pred_mask: The predicted mask of current slice.
            seq_points: The points of polygon on current slice.
        """
        def numpy_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            """Calculate the IOU of two masks
            """
            intersection = np.logical_and(y_true, y_pred)
            union = np.logical_or(y_true, y_pred)
            iou_score = np.sum(intersection) / np.sum(union)
            return iou_score

        def remove_old_rects() -> None:
            """
            Remove the old rectangles that are overlapped with the propagated mask.
            """
            if len(self.deleted_group_ids_after_propagation) == 0:
                return
            
            for slice_id in self.results_nodule.keys():
                deleted_nodule_ids = []
                for nodule_i, nodule in enumerate(self.results_nodule[slice_id]):
                    if nodule['group_id'] in self.deleted_group_ids_after_propagation and nodule['shape_type'] == SHAPETYPE.RECTANGLE:
                        deleted_nodule_ids.append(nodule_i)
                        
                if len(deleted_nodule_ids) != 0:
                    # Remove from tail to avoid index error
                    deleted_nodule_ids = reversed(deleted_nodule_ids)
                    for i in deleted_nodule_ids:
                        del self.results_nodule[slice_id][i]
                    
            self.deleted_group_ids_after_propagation.clear()

        def combine_overlapped_nodule(slice_id: int, pred_mask: np.ndarray) -> np.ndarray:
            """
            Args:
                slice_id: int
                    The slice id of current slice.
                pred_mask: np.ndarray
                    The predicted mask of current slice.
            """
            deleted_polygon_nodule_ids = []
            
            polygon, centers = get_polygon((pred_mask.astype(np.uint8) * 255))
            deleted_group_ids = set() # only use when is_current_slice is True
            # Exists some nodules on current slice
            if slice_id in self.results_nodule.keys():
                # overlap with exist mask, combine
                slice_img_shape = self.mImgSize + [3] # (512, 512, 3)
                
                for nodule_i, nodule in enumerate(self.results_nodule[slice_id]):
                    if nodule['shape_type'] == SHAPETYPE.RECTANGLE: # overlaps with the existing rectangle, remove rectangle later
                        rect = np.array(nodule['rect'], dtype=np.int32) # (x1, y1, x2, y2)
                        if np.sum(pred_mask[rect[1]:rect[3], rect[0]:rect[2]]) > 0:
                            self.deleted_group_ids_after_propagation.add(nodule['group_id'])
                            deleted_group_ids.add(nodule['group_id'])
                    elif nodule['shape_type'] == SHAPETYPE.POLYGON: # overlaps with the existing polygon, combine them
                        try:
                            mask = points_to_mask(nodule['points'], slice_img_shape)
                            iou = numpy_iou(pred_mask, mask)
                            if iou > 0.0:
                                pred_mask = np.logical_or(pred_mask, mask).astype(np.uint8)
                                deleted_polygon_nodule_ids.append(nodule_i)
                                deleted_group_ids.add(nodule['group_id'])
                        except Exception as e:
                            self.errorMessage("Error Predict", f"{e}")
                            continue
            
            # Because there are some polygons overlap with the predicted mask, we need to remove them and update the polygon and centers
            if len(deleted_polygon_nodule_ids) != 0:
                # Remove from tail to avoid index error
                deleted_polygon_nodule_ids = reversed(deleted_polygon_nodule_ids)
                for i in deleted_polygon_nodule_ids:
                    del self.results_nodule[slice_id][i]
                # Update the polygon and centers
                polygon, centers = get_polygon((pred_mask.astype(np.uint8) * 255))
            
            shapes = []
            for points, center_point in zip(polygon, centers):
                if center_point is None:
                    continue
                points = np.array(points)
                shape = self.setDefautShape(label = 'nodule',
                                            points = points,
                                            shape_type = SHAPETYPE.POLYGON,
                                            mask = pred_mask.astype(np.uint8))
                shapes.append(shape)

            self.save_label(slice_id, shapes, add_new_shape = True)
            seq_points = np.array([list(centers[0]) + [1]], dtype = np.int64)
            return seq_points

        def oneway_propagate(predmask: np.ndarray, 
                             seq_points: np.ndarray, 
                             slice_id: int, 
                             toward: int, 
                             slice_range: List[int]) -> int:
            previous_pred_mask = predmask.copy()
            seq_points = seq_points.copy()
            
            while slice_id in slice_range and len(seq_points) > 0:
                try:
                    QApplication.processEvents()
                    pred_mask = predict(self.interactiveModel,
                                        self.image_data_dict[slice_id]['data'],
                                        seq_points, 
                                        self.output_keys, 
                                        if_sis = True,
                                        if_cuda = False)
                    original_pred_mask = pred_mask.copy()
                    iou = numpy_iou(previous_pred_mask, pred_mask)
                    polygon, centers = get_polygon((pred_mask * 255))
                    if iou < 0.60 or (polygon == None or centers == None):
                        return slice_id
                    
                    # Update the previous predicted mask for next propagation
                    previous_pred_mask = original_pred_mask
                    seq_points = combine_overlapped_nodule(slice_id, pred_mask)

                except Exception as e:
                    self.errorMessage("Error Predict", f"{e}")
                    return slice_id
                slice_id += toward
                self.progress_bar.setValue(min(self.progress_bar.value() + 1, 100))
            return slice_id
        
        from libraries.interactiveSegment import predict
        # Check whether the interactive model exists
        if self.interactiveModel is None:
            self.errorMessage("InteractiveModel is None")
            return
        
        slice_range = range(1, len(self.image_data_dict))
        
        # Combine the predicted mask with the existing polygons
        # Propagate to the upper slices
        upper_num = oneway_propagate(pred_mask, seq_points, self.current_slice - 1, -1, slice_range)
        # Propagate to the lower slices
        lower_num = oneway_propagate(pred_mask, seq_points, self.current_slice + 1, 1, slice_range)
        
        remove_old_rects()
        self.progress_bar.setValue(100)
      
    def change_auto_refresh_frequency(self):
        new_freq, ok = QInputDialog.getInt(self, 'Change Auto Refresh Frequency', 'Please input a number(seconds):', self.auto_refresh_freqency, 30, step=1)
        if ok:
            self.auto_refresh_freqency = new_freq
            self.auto_refresh_timer.setInterval(self.auto_refresh_freqency * 1000) # ms
      
def get_args():
    parser = argparse.ArgumentParser(description='Lung Nodule Detection')
    parser.add_argument('--user_name', type=str, default='default', help='user_name name')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    full_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(full_path))
    app = QApplication(sys.argv)
    app.setStyle("QtCurve")
    
    # Get user name
    args = get_args()
    user_name = args.user_name
    
    # Set up main window
    window = MainWindow(user_name = user_name)
    window.setWindowIcon(QIcon('./sources/detect.ico'))
    window.setWindowTitle(__appname__)
    # window.show()
    window.showMaximized()
    # window.showFullScreen()
    sys.exit(app.exec_())