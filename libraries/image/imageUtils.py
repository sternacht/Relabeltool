import os
from posixpath import basename
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImageReader
import cv2 as cv
from ..ustr import ustr
from .. import utils
from ..image.mhdUtils import get_segmented_lungs
import numpy as np
import pickle

def readPatientInfor(path:str, file = "Patient.txt"):
    patient_infor = {
        "PatientID": None,
        "Name": None,
        "Gender": None,
        "Date_of_Birth": None,
        "Modality": None,
        "Date_of_Study": None,
        "Path": path,
        "Style": "Image"
    }
    path_file = os.path.join(path, file)
    if not os.path.exists(path_file):
        return patient_infor
    with open(path_file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(":")
            if line[0] in patient_infor.keys():   
                patient_infor[line[0]] = line[1]
        if patient_infor['PatientID'] is None:
            patient_infor['PatientID'] = os.path.split(path)[-1] 
    return patient_infor

def gene_merge(pred,img):
    pred_mask= cv.merge([pred*255,pred*255,np.zeros_like(pred)])
    result= np.uint8(np.clip(img*0.7+pred_mask*0.3,0,255))
    return result

class load_images(QObject):
    imageDataNList = pyqtSignal(list, list)
    finished = pyqtSignal()
    imageDataDict = pyqtSignal(dict, tuple)
    errorSignal = pyqtSignal()
    

    def __init__(self, path_folder):
        super().__init__(parent=None)
        self.path = path_folder
        self.dirname = path_folder
        self.__images = []
        self.infor = readPatientInfor(self.path)
        self.spacing = (0.6, 0.6)

    def get_patient_infor(self):
        return self.infor

    @pyqtSlot()
    def processing(self):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []
        basenames = []
        pathImages = []
        gray_images = []

        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    pathImages.append(path)
        utils.natural_sort(pathImages, key=lambda x: x.lower())
        print(pathImages)
        for pathImage in pathImages:
            basenames.append(os.path.splitext(os.path.basename(pathImage))[0])
            image = cv.imread(pathImage)
            images.append(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            gray_images.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        array_size = len(images)
        gray_images = np.array(gray_images)
        top = 0
        bottom = len(images)
        # head_tail_text = os.path.join(self.dirname, "headtail.dat")
        # if not os.path.exists(head_tail_text):
        #     for index, slice in enumerate(gray_images):
        #         mask = get_segmented_lungs(slice, 200)
        #         if np.mean(mask) > 0.008:
        #             top = index
        #             break
        #     for index, slice in enumerate(gray_images[::-1]):
        #         mask = get_segmented_lungs(slice, 200)
        #         if np.mean(mask) > 0.008:
        #             bottom = array_size - index
        #             break

        #     if bottom - top <= array_size//2:
        #         top = 0
        #         bottom = len(images)
        #     with open(head_tail_text, 'w+') as write_file:
        #         write_file.write("{} {}".format(top, bottom))
        #         write_file.close()
        # else:
        #     index_txts = []
        #     with open(head_tail_text, "r+") as head_tail_txt:
        #         for head_tail in head_tail_txt:
        #             index_txts.extend([int(i) for i in head_tail.split()])
        #         if len(index_txts) > 0 and (bottom - top >= array_size//2):
        #             top = index_txts[0]
        #             bottom = index_txts[1]
        #     head_tail_txt.close()
        # self.__images = images[top: bottom]
        # mImgList = basenames[top: bottom]

        self.__images = images
        mImgList = basenames

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
        if len(self.__images) > 0:
            image_dict = list2dict(images, basenames)
            self.imageDataDict.emit(image_dict, self.spacing)

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
            save_pickle(save_path, gray_images, basenames, top, bottom)
            self.imageDataNList.emit(self.__images, mImgList)
            self.finished.emit()
        else:
            self.errorSignal.emit()
            self.finished.emit()







        
    
