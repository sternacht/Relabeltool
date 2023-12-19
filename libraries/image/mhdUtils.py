import pickle
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import SimpleITK as sitk
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt

def make_position(v_center, v_diam, spacing):
    """Transform from CT coordinates to matrix
    determine nodule coordinates
    """
    v_diam_z=int(v_diam/spacing[2]+1)
    v_diam_y=int(v_diam/spacing[1]+1)
    v_diam_x=int(v_diam/spacing[0]+1)
    v_diam_z = np.rint(v_diam_z / 2)
    v_diam_y = np.rint(v_diam_y / 2)
    v_diam_x = np.rint(v_diam_x / 2)
    z_min = int(v_center[0] - v_diam_z)
    z_max = int(v_center[0] + v_diam_z + 1)
    x_min = int(v_center[1] - v_diam_x)
    x_max = int(v_center[1] + v_diam_x + 1)
    y_min = int(v_center[2] - v_diam_y)
    y_max = int(v_center[2] + v_diam_y + 1)
    return (x_min, x_max), (y_min, y_max), (z_min, z_max)

def label2Yolo(x_min, x_max, y_min, y_max, width=512, height=512):
    "Convert x1, y1, x2, y2 in label format into yolo format"
    w_yolo = x_max - x_min
    h_yolo = y_max - y_min
    x_center = int(x_min + w_yolo/2 + 1)
    y_center = int(y_min + h_yolo/2 + 1)
    return[float(x_center/width), float(y_center/height), float(w_yolo/width), float(h_yolo/height)]

def read(path):
    if path.endswith(".mhd") and os.path.exists(path):
        return sitk.ReadImage(path)
    return None

def readArray(data, windowMin = -1350, windowMax = 250):
    if data:
        # data = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(
            sitk.Cast(sitk.IntensityWindowing(
                data,
                windowMinimum= windowMin,
                windowMaximum=windowMax,
                outputMinimum=0.0,
                outputMaximum=255.0
            ), sitk.sitkUInt8)
        )
        return array, data.GetOrigin(), data.GetSpacing()
    return None

def writeArray(arr:np.array, pathSave:str, filename, start = 0, end = 10000, extension = ".jpg"):
    dim = arr.shape[0]
    if start < 0:
        start = 0
    if end > dim:
        end = dim
    array = arr[start: end]
    listFileName = [
        os.path.join(pathSave, "IMG-" + filename + "-{0:04d}".format(count+1) + extension) for count in range(start, end)
    ]
    # print("len list: {}".format(len(listFileName)))
    pickle_data = {}
    for count in range(start, end):
        previous = count - 1 if count > 0 else 0
        next = count + 1 if count < len(arr)- 1 else count 
        #print(arr[count].shape, previous, count, next)
        data_arr = np.dstack((arr[previous], arr[count], arr[next]))
        basename = "IMG-" + filename + "-{0:04d}".format(count+1)
        pickle_data[basename] = data_arr
    save_path = os.path.join(pathSave, "image.pickle")
    pickle.dump(pickle_data, open(save_path, "wb"))

    sitk.WriteImage(sitk.GetImageFromArray(array),listFileName)
    return listFileName 

def readNodule(pandas_data, spacing, origin):
    """
    pandas_data: the csv data with pandas format inculde nodule
    spacing: get from data.GetSpacing() function
    origin: get from data.GetOrigin() function
    """
    nodule_coordinate = []
    for nodule_index, cur_row in pandas_data.iterrows():
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]
        center = np.array([node_x, node_y, node_z])
        # nodule center
        v_center = np.rint((center - origin) / spacing)
        v_diam = diam
        # convert x,y,z order v_center to z,y,x order v_center
        v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
        # make_mask(mask_itk, v_center, v_diam,spacing)
        x, y, z = make_position(v_center, v_diam, spacing)
        # print("Nodule coord", x,y,z)
        nodule_coordinate.append(x, y, z)
    return nodule_coordinate

def get_segmented_lungs(im, threshold=-400):
    from skimage.segmentation import clear_border
    from skimage.measure import label,regionprops, perimeter
    from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
    from skimage.filters import roberts, sobel
    from scipy import ndimage as ndi
    import scipy.ndimage
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < threshold

    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)

    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)

    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)

    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    return binary

def extract_main(mask, spacing, vol_limit=[0.68, 8.2]):
    from skimage import measure
    
    voxel_vol = spacing[0]*spacing[1]*spacing[2]

    label = measure.label(mask, connectivity=1)

    properties = measure.regionprops(label)

    for prop in properties:
            if prop.area * voxel_vol < vol_limit[0] * 1e6 or prop.area * voxel_vol > vol_limit[1] * 1e6:
                mask[label == prop.label] = 0
                
    return mask

class loadMHD(QObject):

    imageDataNList = pyqtSignal(list, list)
    imageDataDict = pyqtSignal(dict, tuple)
    finished = pyqtSignal()

    def __init__(self, path_filename):
        super().__init__(parent=None)
        self.path = path_filename
        self.dirname = os.path.dirname(path_filename)
        self.__image = []
        self.data = read(self.path)
        self.infor = self.read_patient_infor()

    def read_patient_infor(self):
        patient_infor = {
            "PatientID": "0010|0020",
            "Name": "0010|0010",
            "Gender": "0010|0040",
            "Date_of_Birth": "0010|0030",
            "Modality": "0008|0060",
            "Date_of_Study": "0008|0020"
        }
        # tags_to_copy = ["0010|0010", # Patient Name
        #         "0010|0020", # Patient ID
        #         "0010|0040", # Gender
        #         "0010|0030", # Patient Birth Date
        #         "0020|000D", # Study Instance UID, for machine consumption
        #         "0020|0010", # Study ID, for human consumption
        #         "0008|0020", # Study Date
        #         "0008|0030", # Study Time
        #         "0008|0050", # Accession Number
        #         "0008|0060"  # Modality
        #     ]
        for key, value in patient_infor.items():
            if self.data.HasMetaDataKey(value):
                patient_infor[key] = self.data.GetMetaData(value)
            else:
                patient_infor[key] = None
        patient_infor["Path"] = self.path
        if patient_infor["PatientID"] is None:
            patient_infor["PatientID"] = os.path.split(os.path.dirname(self.path))[-1]

        patient_infor["Style"] = "MHD"
        return patient_infor

    def get_patient_infor(self):
        return self.infor

    @pyqtSlot()
    def processing(self):
        array, orgin, spacing = readArray(self.data)

        array_size = len(array)
        top = 0
        bottom = array_size

        image_MetaData = array[::-1]
        for image in image_MetaData:
            self.__image.append(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        
        # head_tail_text = os.path.join(self.dirname, "headtail.txt")
        # if not os.path.exists(head_tail_text):
        #     for index, slice in enumerate(image_MetaData):
        #         mask = get_segmented_lungs(slice, 200)
        #         if np.mean(mask) > 0.008:
        #             top = index
        #             break
        #     for index, slice in enumerate(image_MetaData[::-1]):
        #         mask = get_segmented_lungs(slice, 200)
        #         if np.mean(mask) > 0.008:
        #             bottom = array_size - index
        #             break
        #     with open(head_tail_text, 'w+') as write_file:
        #         write_file.write("{} {}".format(top, bottom))
        #         write_file.close()
        # else:
        #     index_txts = []
        #     with open(head_tail_text, "r") as head_tail_txt:
        #         for head_tail in head_tail_txt:
        #             index_txts.extend([int(i) for i in head_tail.split()])
        #         if len(index_txts) > 0:
        #             top = index_txts[0]
        #             bottom = index_txts[1]
        #     head_tail_txt.close()
            # print(index_txts)

        def list2dict(images, filename):
            dictionary = {}
            for index in range(len(images)):
                image_data = images[index]
                dictionary[index + 1 ] = {
                    'path': "IMG-" + filename + "-{0:04d}".format(index+1),
                    'data': image_data,
                    'mode': False
                }
            return dictionary

        foldername = os.path.split(self.dirname)[-1]
        image_dict = list2dict(image_MetaData, foldername)


        self.imageDataDict.emit(image_dict, tuple(spacing[:2]))

        self.__image = self.__image[top: bottom]
        mImgList = writeArray(image_MetaData, self.dirname, foldername, top, bottom)

        # print("Len of Images  ", len(self.__image))
        self.imageDataNList.emit(self.__image, mImgList)
        # self.mImgList.emit(mImgList)
        self.finished.emit()

if __name__ == "__main__":
    path= r"D:\02_BME\000_LUNA16\000_000_raw\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"
    data = read(path)

    # path = r"D:\Master\02_Project\00000744\00000740_4_01.dcm"
    # data = sitk.ReadImage(path)
    # tags_to_copy = ["0010|0010", # Patient Name
    #             "0010|0020", # Patient ID
    #             "0010|0030", # Patient Birth Date
    #             "0020|000D", # Study Instance UID, for machine consumption
    #             "0020|0010", # Study ID, for human consumption
    #             "0008|0020", # Study Date
    #             "0008|0030", # Study Time
    #             "0008|0050", # Accession Number
    #             "0008|0060"  # Modality
    #         ]

    # series_tag_values = [(k, data.GetMetaData(k)) for k in tags_to_copy if data.HasMetaDataKey(k)]
    # print(series_tag_values)
    print(data)
    preprocessing = loadMHD(path)
    # preprocessing.processing()

    # print(data.HasMetaDataKey("0010|0010"))
    # print(data.GetMetaData("0010|0010"))
    # print(data.GetMetaDataKeys())
    # print(data.PatientName())
    infors = preprocessing.data.GetMetaDataKeys()
    for infor in infors:
        print(preprocessing.data.GetMetaData(infor))

