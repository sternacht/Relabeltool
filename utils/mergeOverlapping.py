from collections import OrderedDict
import numpy as np
import os
import glob
from typing import List, Tuple, Dict, Any

def readtxt(path:str):
    """
    Return Dictionary with format
    {
        "File name Image": [list bounding box in (x1, y1, x2, y2, label, conf)]
    }
    """
    bboxes = {}
    with open(path, 'r+') as f:
        for lines in f.readlines():
            lines = lines.strip().split()
            filename = lines[0]
            bbox = []
            for line in lines[1:]:
                line = [float(b) for b in line.split(',')]
                bbox.append(line)
            bboxes[filename] = bbox
        f.close()
    return bboxes

def xywh2xyxy(bb:np.array):
    bb_xyxy = bb[:,:4]
    conf = bb[:,-1]
    bb_xyxy[:,2:] = bb_xyxy[:,:2] + bb_xyxy[:, 2:]
    return bb_xyxy, conf

def xywh2xcyc(bb:np.array):
    bb_xcyc = bb[:,:4]
    conf = bb[:,-1]
    bb_xcyc[:,:2] = bb_xcyc[:, :2] + bb_xcyc[:,2:]/2.0
    return bb_xcyc, conf

def xywh2xyxyc(bb:np.array):
    bb_xyxy = bb[:,:4]
    conf = bb[:,-1]
    bb_xyxy[:,2:] = bb_xyxy[:,:2] + bb_xyxy[:, 2:]
    cen = bb_xyxy[:,:2] + bb_xyxy[:,2:]/2.0
    return bb_xyxy, cen, conf

def expand_filename(filename:str):
    filename = filename.split('-')
    patient = filename[-2][-4:]
    slice = filename[-1][-4:]
    return int(patient), int(slice)

def follow_patient(bb_dict:dict):
    """
    Return: Dictionary with format
    {
        "Patient" : {"num_slice": [list Bounding box],
                    "filename": filename}
    }
    """
    patient_dict = {}
    patient_file = {}
    for filename in bb_dict.keys():
        patient, num_slice = expand_filename(filename)
        bb = bb_dict[filename]
        for b in bb:
            b.append(num_slice)
        if patient not in patient_dict:
            patient_dict[patient] = {}
            patient_file[patient] = {}
        patient_dict[patient][num_slice] = bb
        patient_file[patient][num_slice] = filename
        
    return patient_dict, patient_file

def cal_bbox2ds_iou(bboxes1: np.ndarray, bboxes2: np.ndarray):
    """
    Calculate IoU of list of bounding box.
    Args:
        bboxes1 (numpy.ndarray): 
            First set of bounding boxes, in shape (n, 4), n is the number of bounding boxes, each bounding box has [x_min, y_min, x_max, y_max] format coordinates.
        bboxes2 (numpy.ndarray): 
            Second set of bounding boxes, in shape (m, 4), m is the number of bounding boxes, each bounding box has [x_min, y_min, x_max, y_max] format coordinates.
    Returns:
        iou_matrix (numpy.ndarray): 
            IoU matrix, in shape (n, m), representing the IoU between the first set of bounding boxes and the second set of bounding boxes.
    """
    # Calculate intersection areas
    intersect_min = np.maximum(bboxes1[:, :2].reshape(-1, 1, 2), bboxes2[:, :2])
    intersect_max = np.minimum(bboxes1[:, 2:].reshape(-1, 1, 2), bboxes2[:, 2:])
    intersect_area = np.prod(np.maximum(0, intersect_max - intersect_min), axis=2)

    area_bboxes1 = np.prod(bboxes1[:, 2:] - bboxes1[:, :2], axis=1)
    area_bboxes2 = np.prod(bboxes2[:, 2:] - bboxes2[:, :2], axis=1)

    iou_matrix = intersect_area / (area_bboxes1.reshape(-1, 1) + area_bboxes2 - intersect_area)

    return iou_matrix

def distance_ratio(BB1:np.array, BB2:np.array):
    bboxes1 = BB1[:,:4]
    bboxes2 = BB2[:,:4]
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dc = np.zeros((rows, cols))
    if rows * cols == 0:#
        return dc
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dc = np.zeros((cols, rows))
        exchange = True

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    out_max_xy = np.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = np.minimum(bboxes1[:, :2], bboxes2[:, :2])


    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    
    outer = np.clip((out_max_xy - out_min_xy), a_min=0, a_max=10000)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    dc = (inter_diag) / outer_diag
    # dious = torch.clamp(dious,min=-1.0, max = 1.0)
    if exchange:
        dc = np.transpose(dc)
    # return dious
    return dc

def add_slice_to_dict(merge_dict:dict, patient, find_slice:np.array):
    if patient not in merge_dict:
        merge_dict[patient] = {}
    for bbox in find_slice:
        num_slice = int(bbox[-1])
        bb = bbox[:-1].tolist()
        if num_slice not in merge_dict[patient]:
            merge_dict[patient][num_slice] = []
        merge_dict[patient][num_slice].append(bb)

def remove_duplicate(merge_dict:dict):
    for patient, slice_dict in merge_dict.items():
        for num_slice, bboxes in slice_dict.items():
            # bboxes = np.array(bboxes)
            unique_bb = np.unique(bboxes, axis=0)
            merge_dict[patient][num_slice] = unique_bb.tolist()
        sort_dict = merge_dict[patient]
        sorted_dict = sorted(sort_dict.items())
        merge_dict[patient] = dict(sorted_dict)
    return merge_dict

        

def merge_overlapping(bbox_dict:dict, dc_ratio_thresh = 0.1):
    patient_dict, patient_file = follow_patient(bbox_dict)
    merge_overlapping_dict = {}
    for patient, slices in patient_dict.items():
        for slice, bboxes in slices.items():
            current_bb = bboxes
            find_continues_slice = []
            for i in [-2, -1, 1, 2]:
                if slice + i in slices.keys():
                    find_continues_slice.extend(slices[slice + i])
            if len(find_continues_slice) <=2:
                continue

            current_bb = np.array(current_bb)
            find_continues_slice = np.array(find_continues_slice)
            # compute_distance2(current_bb, find_continues_slice)
            for cur_bb in current_bb:
                cur_bb = np.expand_dims(cur_bb, axis=0)
                dc_bb = distance_ratio(cur_bb, find_continues_slice)
                keep_bb = find_continues_slice[dc_bb < dc_ratio_thresh]
                add_slice_to_dict(merge_overlapping_dict, patient, keep_bb)
    merge_overlapping_dict = remove_duplicate(merge_overlapping_dict)
    return merge_overlapping_dict, patient_file

def add_category_default(merge_dict:dict):
    for patient, slice_dict in merge_dict.items():
        for num_slice, bboxes in slice_dict.items():
            added_box = []
            for box in bboxes:
                box += [1, 0]
                added_box.append(box)
            merge_dict[patient][num_slice] = added_box
    return merge_dict

def dict2txt(merged_dict:dict, patient_file:dict):
    # file_name = os.path.splitext(os.path.basename(file))[0].split('_')[-1]
    string_txt = ""
    for patient, slice_dict in merged_dict.items():
        for num_slice, bboxes in slice_dict.items():
            string_bb = ""
            for bb in bboxes:
                if isinstance(bb, dict):
                    string_bb +=" {},{},{},{},{},{:.5f}".format(bb['rect'][0], bb['rect'][1], bb['rect'][2], bb['rect'][3], 0, bb['conf'])
                else:
                    string_bb += " {},{},{},{},{},{:.5f}".format(bb[0], bb[1], bb[2], bb[3], int(bb[4]), bb[5])
            string_filename = patient_file[patient][num_slice]
            
            string_txt += string_filename + string_bb + "\n"
    return string_txt

class NoduleTracker():
    """Track nodule in slices
    
    Use 2D bounding box in each slice to track 3D bounding box
    """
    def __init__(self, max_num_of_disappeared = 1, iou_threshold = 0.0):
        self.tracked_nodule_bbox2ds = OrderedDict()
        self.tracked_nodule_slice_ids = OrderedDict()
        self.__next_nodule_id = 0
        self.__disappeared = OrderedDict()
        self.__max_num_of_disappeared = max_num_of_disappeared
        self.__iou_threshold = iou_threshold

    def register(self, nodule_bbox2ds: List[Any], slice_i: int) -> None:
        self.tracked_nodule_bbox2ds[self.__next_nodule_id] = nodule_bbox2ds
        self.tracked_nodule_slice_ids[self.__next_nodule_id] = slice_i
        self.__disappeared[self.__next_nodule_id] = 0
        self.__next_nodule_id += 1

    def deregister(self, nodule_id: int) -> None:
        del self.tracked_nodule_bbox2ds[nodule_id]
        del self.tracked_nodule_slice_ids[nodule_id]
        del self.__disappeared[nodule_id]

    def update(self, 
               nodule_bbox2d_infos: List[Any], 
               slice_i: int) -> Tuple[Dict[int, List[Any]], Dict[int, int]]:
        """
        Args:
            nodule_bbox2d_infos: 
                List of bounding box in format [x1, y1, x2, y2, cls, conf, <whole dict>]
                <whole dict> is the whole dictionary of bounding box in format
                {
                    'label': 'nodule',
                    'points': [(x1,y1), (x2, y2)],
                    'conf': conf,
                    'checked': True,
                    'category': category_id,
                    'shape_type': 'rectangle',
                    'group_id': noduleId,
                    'rect': [x1,y1,x2,y2],
                    'description': None
                }
            slice_i: 
                slice number
        Return:
            Tuple of (tracked_nodule_3d, slice_ids), where
            tracked_nodule_3d: 
                Dictionary of bounding box in format
                {
                    nodule_id: [x1, y1, x2, y2, cls, conf, <whole dict>]
                }
            slice_ids:
                Dictionary of slice number in format
                {
                    nodule_id: slice_i
                }
        """
        
        # if there are not any nodules, deregister all objects
        if len(nodule_bbox2d_infos) == 0:
            for nodule_id in list(self.__disappeared.keys()):
                self.__disappeared[nodule_id] += 1
                if self.__disappeared[nodule_id] >= self.__max_num_of_disappeared:
                    self.deregister(nodule_id)
            return self.tracked_nodule_bbox2ds, self.tracked_nodule_slice_ids
        
        # if there is no object, register all rects
        if len(self.tracked_nodule_bbox2ds) == 0:
            for i in range(len(nodule_bbox2d_infos)):
                self.register(nodule_bbox2d_infos[i], slice_i)
        else:
            tracked_nodule_ids = list(self.tracked_nodule_bbox2ds.keys())
            
            nodule_bbox2ds = []
            for bbox_info in nodule_bbox2d_infos:
                nodule_bbox2ds.append(bbox_info[:4])
            tracked_nodule_bbox2ds = []
            for bbox_info in self.tracked_nodule_bbox2ds.values():
                tracked_nodule_bbox2ds.append(bbox_info[:4])
                
            iou_matrix = cal_bbox2ds_iou(np.array(nodule_bbox2ds), np.array(tracked_nodule_bbox2ds))
            iou_matrix = iou_matrix.T
            
            max_value_along_row = np.max(iou_matrix, axis=1)
            
            max_value_along_col = np.max(iou_matrix, axis=0)
            argmax_value_along_col = np.argmax(iou_matrix, axis=0)
            
            # Deregiseter old object
            for i in range(len(max_value_along_row)):
                if max_value_along_row[i] <= self.__iou_threshold:
                    nodule_id = tracked_nodule_ids[i]
                    self.__disappeared[nodule_id] += 1
                    if self.__disappeared[nodule_id] >= self.__max_num_of_disappeared:
                        self.deregister(nodule_id)
            # Register new object or update old object
            for row_i, col_i in zip(argmax_value_along_col, range(len(max_value_along_col))):
                old_objectID = tracked_nodule_ids[row_i]
                if max_value_along_col[col_i] > self.__iou_threshold:
                    self.tracked_nodule_bbox2ds[old_objectID] = nodule_bbox2d_infos[col_i]
                    self.tracked_nodule_slice_ids[old_objectID] = slice_i
                    self.__disappeared[old_objectID] = 0
                else:
                    self.register(nodule_bbox2d_infos[col_i], slice_i)
                    
        return self.tracked_nodule_bbox2ds, self.tracked_nodule_slice_ids
    
def collect3d(patient_merged_dict: Dict[str, Dict[int, List[Dict[str, Any]]]]):
    """Merge 2D bounding boxes into 3D bounding boxes
    Args:
        mergedDict: Dictionary with format
        {
            "PatientID": {
                "num_slice": [list Bounding box]
            }
        }
        Each Bounding box is a dictionary with format
        {
            'label': 'nodule',
            'points': [(x1,y1), (x2, y2)],
            'conf': conf,
            'checked': True,
            'category': category_id,
            'shape_type': 'rectangle',
            'group_id': noduleId,
            'rect': [x1,y1,x2,y2],
            'description': None
        }
    """
    
    patient_tracking = {}
    for patient, slices_merged_dict in patient_merged_dict.items():
        nodule_tracker = NoduleTracker(max_num_of_disappeared = 1)
        tracking = dict()
        for slice_i, bboxes in slices_merged_dict.items():
            merged_bbox2ds = []
            
            for bb in bboxes:
                if bb is None:
                    continue
                if isinstance(bb, dict):
                    merged_bbox2ds.append([*bb['rect'], 0, bb['conf'], bb]) # [x1, y1, x2, y2, cls, conf, <whole dict>]
                elif isinstance(bb, list):
                    merged_bbox2ds.append(bb[:6])
            tracked_nodule_bbox2ds, slice_ids = nodule_tracker.update(merged_bbox2ds, slice_i)
            
            # Add tracked nodule to tracking
            for nodule_id, bboxes in tracked_nodule_bbox2ds.items():
                if nodule_id not in tracking.keys():
                    tracking[nodule_id] = []
                # concat (slice_id, x1, y1, x2, y2, cls, conf, <whole dict>) into a list
                tracking[nodule_id].append([slice_ids[nodule_id]] + list(bboxes)) 
            
            # Check whether there are not any nodule in next 2 slices
            if slice_i + 1 not in slices_merged_dict.keys():
                tracked_nodule_bbox2ds, slice_ids = nodule_tracker.update([], slice_i + 1)
        
        noduleId = {}
        noduleID_count = 1
        for _, values in tracking.items():
            buff = {}
            vs = []
            for value in  values:
                v = list(value[:6])
                vs.append(v)
                buff[tuple(v)] = value 
            vs = np.unique(vs, axis=0).tolist()
            values_list = []
            for v in vs:
                values_list.append(buff[tuple(v)])
            # if len(values_list) > 2:
            noduleId[noduleID_count] = values_list
            noduleID_count += 1
        patient_tracking[patient] = noduleId
    return patient_tracking

def patientAnalysis(patient_tracking:dict, spacing = (0.6, 0.6, 1)):
    """
    Format Dictionary
    {
        PatientID: {
            NoduleID: [[num_slice, coordinates(x1,y1,x2,y2,cls,conf)]]
            }
        }
    }
    Return:
    List: [
        {
            PatientID : PatientID(str),
            NoduleID : NoduleID(int)
            Slice_range: [list]
            Central_Slice : filename(str)
            Diameters: int
            Category: [Benign, Probably Benign, Probably Suspicious, Suspicious]:str
            x1: x1
            y1: y1
            x2: x2
            y2: y2
        }
    ]
    """
    list_category = ['Benign', 'Probably Benign', 'Probably Suspicious', 'Suspicious']
    def category_f(diameter):
        if diameter < 4.0:
            return "Benign", 0
        elif diameter >=4.0 and diameter < 6.0:
            return "Probably Benign", 1
        elif diameter >= 6.0 and diameter <8.0:
            return "Probably Suspicious", 2
        else:
            return "Suspicious", 3
    patient_list = []
    new_patient_dict = {}
    for patient, noduleIds in patient_tracking.items():
        patient_dict = {}
        for noduleId, data in noduleIds.items():
            bboxes = np.array(data)
            slice_list = list(bboxes[:, 0])
            length = (bboxes[:, 3] - bboxes[:,1]) * (bboxes[:, 4] - bboxes[:,2])
            threedV = np.max(length) * len(length) * spacing[0] * spacing[1] * spacing[2]
            diameter = np.round(pow(threedV, 1/3), 4)

            try:
                if bboxes[0][-1]['mode'] == 0:
                    category_name, category_id = category_f(diameter)
                else:
                    category_id = bboxes[0][-1]['category']
                    category_name = list_category[category_id]
            except:
                category_name, category_id = category_f(diameter)
            
            # Find mark type
            mark_type = 'rectangle'
            for nodule_2d_info in bboxes:
                info = nodule_2d_info[-1]
                if info['shape_type'] == 'polygon':
                    mark_type = 'polygon'
                    break
            
            patient_data = {
                'NoduleID': noduleId,
                'Slice_range': slice_list,
                "Diameters" : diameter,
                "Category" : category_name,
                "Mark Type": mark_type,
                'data': bboxes,
                'description': data[0][-1].get('description', None) if len(data[0]) > 7 else None
            }
            patient_list.append(patient_data)
            for d in data:
                if len(d)>7:
                    slice_num = d[0]
                    s = d[-1]
                    s['category'] = category_id
                    s['group_id'] = noduleId
                else:
                    slice_num, x1, y1, x2, y2, cls, conf = d[:7]
                    s = {
                        'label': 'nodule',
                        'points': [(x1,y1), (x2, y2)],
                        'conf': conf,
                        'checked': True,
                        'category': category_id,
                        'shape_type': 'rectangle',
                        'group_id': noduleId,
                        'rect': [x1,y1,x2,y2],
                        'description': None
                    }
                if slice_num not in patient_dict.keys():
                    patient_dict[slice_num] = []
                patient_dict[slice_num].append(s)
        new_patient_dict[patient] = patient_dict
    return patient_list, new_patient_dict            

                
        
if __name__ == "__main__":
    # os.chdir(r"D:\02_BME\NoduleDetection_v4")
    path = r"data\inference.txt"
    dir = os.path.dirname(path)


    # bboxes = readtxt(r"D:\Master\02_Project\NoduleDetection_v4\test.txt")
    bboxes = readtxt(path)
    merged_dict, filedict = merge_overlapping(bboxes)
    patient_tracking = collect3d(merged_dict)
    patient_list, patient_dict = patientAnalysis(patient_tracking)
    print(patient_dict)
    # string_txt= dict2txt(merged_dict, filedict)
    # filename = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
    # path_save = os.path.join(dir,"merged_{}_TONY_2.txt".format(filename))
    # with open(path_save, "w+") as f:
    #     f.writelines(string_txt)
    #     f.close()
    # bb_array = np.array(bboxes['IMG-0558-0262'])
    # bb_xy, conf = xywh2xyxy(bb_array)
    # print(bb_xy)