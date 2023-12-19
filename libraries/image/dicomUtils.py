import SimpleITK as sitk
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import os
import pickle
import cv2 as cv

def readArray(data, windowMin = -1350, windowMax = 250):
    if data:
        array = sitk.GetArrayFromImage(
            sitk.Cast(sitk.IntensityWindowing(
                data,
                windowMinimum= windowMin,
                windowMaximum= windowMax,
                outputMinimum= 0.0,
                outputMaximum= 255.0
            ), sitk.sitkUInt8)
        )
        return array
    return None

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


class load_Dicom(QObject):
    imageDataNList = pyqtSignal(list, list, list)
    imageDataDict = pyqtSignal(dict, tuple)
    errorSignal = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, directory, parent = None) -> None:
        super().__init__(parent)
        self.path = os.path.abspath(directory)
        # self.dirname = os.path.abspath(directory) 
        self.reader = sitk.ImageSeriesReader()
        self.__image = []
        self.__imageRGB = []
        self.infor = None
        self.__filename = []
        self.__imagesize = None
        self.image = None


    def read_patient_infor(self):
        patient_infor = {
            "PatientID": "0010|0020",
            "Name": "0010|0010",
            "Gender": "0010|0040",
            "Date_of_Birth": "0010|0030",
            "Modality": "0008|0060",
            "Date_of_Study": "0008|0020",
            "Slice_Thickness":"0018|0050",
            "Pixel_Spacing":"0028|0030",
        }
        for key, value in patient_infor.items():
            if self.reader.HasMetaDataKey(0, value):
                patient_infor[key] = self.reader.GetMetaData(0, value)
            else:
                patient_infor[key] = None
        patient_infor["Path"] = self.path
        if patient_infor["PatientID"] is None:
            patient_infor["PatientID"] = os.path.split(os.path.dirname(self.path))[-1]
        if patient_infor["Date_of_Birth"] is not None:
            date = patient_infor["Date_of_Birth"]
            patient_infor["Date_of_Birth"] = "{}/{}/{}".format(date[:4], date[4:6], date[6:])
        if patient_infor["Date_of_Study"] is not None:
            date = patient_infor["Date_of_Study"]
            patient_infor["Date_of_Study"] = "{}/{}/{}".format(date[:4], date[4:6], date[6:])
        if patient_infor["Name"] is not None:
            patient_infor["Name"] = patient_infor["Name"].replace(" ", "")
        if patient_infor["Gender"] is not None:
            patient_infor["Gender"] = patient_infor["Gender"].replace(" ", "")
        # if patient_infor["Slice_Thickness"] is not None:
        #    data = patient_infor["Slice_Thickness"]
        #    patient_infor["Slice_Thickness"] = str(data)
        # else:
        #    patient_infor["Slice_Thickness"] = str("1")
        # if patient_infor["Pixel_Spacing"] is not None:
        #     data = patient_infor["Pixel_Spacing"].split('\\')
        #     patient_infor["Pixel_Spacing"] = f"{data[0][:4]}x{data[1][:4]}"
        spacing = np.round(self.image.GetSpacing(), decimals=5)
        patient_infor["Pixel_Spacing"] = f"{spacing[1]}x{spacing[0]}"
        patient_infor["Slice_Thickness"] = f"{spacing[2]}"
        patient_infor["Spacing"] = spacing
        patient_infor["Style"] = "DICOM"
        self.infor = patient_infor

    def get_patient_infor(self):
        return self.infor


    @pyqtSlot()
    def processing(self):
        try:
            # image_names = self.reader.GetGDCMSeriesFileNames(os.path.join(self.path, 'Dicom'))
            image_names = self.reader.GetGDCMSeriesFileNames(self.path + '/dicom') # modified by Ben
            self.reader.SetFileNames(image_names)
            self.reader.MetaDataDictionaryArrayUpdateOn()
            self.reader.LoadPrivateTagsOn()
            self.image = self.reader.Execute()

            self.__imagesize = [self.image.GetHeight(),self.image.GetWidth()]
            self.read_patient_infor()

            if self.reader.HasMetaDataKey(0, "0028|1050"):
                window_center = self.reader.GetMetaData(0, "0028|1050").split("\\")[0]
            else:
                window_center = -550
            if self.reader.HasMetaDataKey(0, "0028|1051"):
                window_width = self.reader.GetMetaData(0, "0028|1051").split("\\")[0]
            else:
                window_width = 1600
            # window_center = self.reader.GetMetaData(0, "0028|1050").split("\\")[0]
            # window_width = self.reader.GetMetaData(0, "0028|1051").split("\\")[0]
            min_value = int(window_center) - int(window_width)//2
            max_value = int(window_center) + int(window_width)//2
            image_array = readArray(self.image, windowMin=min_value, windowMax=max_value)

            basenameUID = ["{}-{:04d}".format(self.reader.GetMetaData(index, "0020|000e"), int(self.reader.GetMetaData(index, "0020|0013"))) for index in range(self.image.GetDepth())]

            image_MetaData = image_array[::-1,:,:]
            basenameUID = basenameUID[::-1]

            self.__image = image_MetaData
            self.__filename = basenameUID

            self.__imageRGB = [cv.cvtColor(image, cv.COLOR_GRAY2RGB) for image in self.__image]

            def list2dict(images, image_list):
                dictionary = {}
                for index in range(len(images)):
                    basename = image_list[index]
                    image_data = images[index]
                    dictionary[index + 1] = {
                        'path': basename,
                        'data': image_data, 
                        'mode': False
                    }
                return dictionary
            image_dict = list2dict(self.__imageRGB, self.__filename)
            if self.reader.HasMetaDataKey(0, "0028|0030"):
                spacing = np.round(self.image.GetSpacing(), decimals=5)
                spacing = tuple(spacing)
            else:
                spacing = (0.6,0.6,1)
                print('default spacing', spacing)
            self.imageDataDict.emit(image_dict, spacing)

            def save_pickle(path_save:str, data, list_basename, top, bottom):
                pickle_data = {}
                for index in range(top, bottom):
                    basename = list_basename[index]
                    previous = index -1 if index > 0 else 0
                    next = index + 1 if index < len(data) -1 else index
                    image_data = np.dstack((data[previous], data[index], data[next]))
                    pickle_data[basename] = image_data
                pickle.dump(pickle_data, open(path_save, "wb"))
            save_path = os.path.join(self.path, "image.pickle")
            #save_pickle(save_path, image_MetaData, basenameUID, top, bottom)
            self.imageDataNList.emit(self.__imageRGB, self.__filename, self.__imagesize)
            self.finished.emit()
        except Exception as e:
            print(e)
            self.errorSignal.emit()
            self.finished.emit()
