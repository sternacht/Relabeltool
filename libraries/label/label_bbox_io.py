import sys
import os
import pickle
import stat

from ..constants import DEFAULT_ENCODING, id_color_nodule, id_color_nodule_fill
from ..save.pickleData import save, save_mask

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING


def bbox_to_shape(list_bbox):
    shapes = []
    for bbox in list_bbox:
        # print(bbox)
        x1, y1, x2, y2, label, conf, checked, category, noduleID = bbox
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        label = 'nodule'
        line_color = id_color_nodule[category]
        fill_color = id_color_nodule_fill[category]
        shapes.append((label, points, line_color, fill_color, conf, checked, category, noduleID))
    return shapes

def shape_to_bbox(points):
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')
    for p in points:
        x = p[0]
        y = p[1]
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)
    if x_min < 1:
        x_min = 1
    if y_min < 1:
        y_min = 1
    return int(x_min), int(y_min), int(x_max), int(y_max)


class Label_Save:

    def __init__(self, directoty:str):
        self.directory = directoty
        self.save_string = ""
        pass

    def save_labels_dict(self, filename, shape_dict:dict, patientId:int= 1):
        """
        Format save file:
        IMG_PatientID_numberSlice x1, y1 , x2, y2, label, conf
        """
        for key, values in shape_dict.items():
            self.save_string += "IMG_{:04d}_{:05d}".format(patientId, int(key))
            for bbox in values:
                x1, y1, x2, y2, label, conf, checked, category, noduleID = bbox
                if checked:
                    bbox_string = " {},{},{},{},{},{}".format(int(x1), int(y1), int(x2), int(y2), int(label), conf)
                    self.save_string += bbox_string
            self.save_string += "\n"
        directory = os.path.join(self.directory, "inference")
        if not os.path.exists(directory):
            os.makedirs(directory)
        path_save = os.path.join(directory, filename + TXT_EXT)
        with open(path_save, 'w+', encoding=ENCODE_METHOD) as out_file:
            out_file.writelines(self.save_string)
            out_file.close()

    def save_label_pickle(self, filename, shape_dict):
        directory = os.path.join(self.directory, "inference")
        if not os.path.exists(directory):
            os.makedirs(directory)
        path_save = os.path.join(directory, filename + ".log")
        save(path_save, shape_dict)

    def save_label_pickle2(self, filename, shape_dict):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        path_save = os.path.join(self.directory, filename + ".log")
        save(path_save, shape_dict)

    def save_mask_npz(self, filename, nodule):
        directory = os.path.join(self.directory, "mask")
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.chmod(directory, os.stat(directory).st_mode | stat.S_IWOTH)
        path_save = os.path.join(directory, filename)
        save_mask(path_save, nodule)


    def load_label_pickle(self, directory_file):
        if os.path.exists(directory_file):
            return pickle.load(open(directory_file, 'rb'))
        return {}
        

        
        




        


