from collections import OrderedDict
import numpy as np
import os
import glob


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

    # print('==out_min_xy:', out_min_xy)
    # print('==out_max_xy:', out_max_xy)

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
    def __init__(self, maxDisappeared= 1, dcThres = 0.1):
        self.__nextObjectedId = 0
        self.objects = OrderedDict()
        self.slices = OrderedDict()
        self.__disappeared = OrderedDict()
        self.__maxdisappeared = maxDisappeared
        self.__dcThres = dcThres
        pass

    def register(self, rect, slice):
        self.objects[self.__nextObjectedId] = rect
        self.slices[self.__nextObjectedId] = slice
        self.__disappeared[self.__nextObjectedId] = 0
        self.__nextObjectedId += 1
        pass

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.slices[objectID]
        del self.__disappeared[objectID]
        pass

    def update(self, rects, num_slice):
        if len(rects) == 0:
            for objectID in list(self.__disappeared.keys()):
                self.__disappeared[objectID] += 1
                if self.__disappeared[objectID] > self.__maxdisappeared:
                    self.deregister(objectID)
            return self.objects, self.slices
        if len(self.objects) == 0:
            for i in range(0, len(rects)):
                self.register(rects[i],num_slice)
        else:
            objectIDs = list(self.objects.keys())
            objectRects = list(self.objects.values())
            distance_matrix = []
            for rect in rects:
                D = distance_ratio(np.array([rect]), np.array(objectRects))
                distance_matrix.append(D)

            distance_matrix = np.array(distance_matrix).T
            position = np.argwhere(distance_matrix < self.__dcThres)

            usedRows = set()
            usedCols = set()

            for (row, col) in position:
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = rects[col]
                self.slices[objectID] = num_slice
                self.__disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, distance_matrix.shape[0])).difference(usedRows)
            unusedCols = set(range(0, distance_matrix.shape[1])).difference(usedCols)
            if distance_matrix.shape[0] >= distance_matrix.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.__disappeared[objectID] += 1

                    if self.__disappeared[objectID] > self.__maxdisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(rects[col], num_slice)

        return self.objects, self.slices
def colect3d(mergedDict:dict):
    patient_tracking = {}
    for patient, slices_m in mergedDict.items():
        nt = NoduleTracker(maxDisappeared=1)
        tracking = {}
        for slice, bboxes in slices_m.items():
            m_bboxes = []
            for bb in bboxes:
                if bb is None:
                    continue
                if isinstance(bb, dict):
                    m_bboxes.append([*bb['rect'], 0, bb['conf'], bb])
                else:
                    m_bboxes.append(bb[:6])
            objects, slices = nt.update(m_bboxes, slice)
            for (objectID, bboxes) in objects.items():
                if objectID not in tracking.keys():
                    tracking[objectID] = []
                
                tracking[objectID].append([slices[objectID]] + list(bboxes))
            if slice + 1 not in slices_m.keys():
                objects, slices = nt.update([], slice + 1)
            if slice + 2 not in slices_m.keys():
                objects, slices = nt.update([], slice + 2)
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
            patient_data = {
                'NoduleID': noduleId,
                'Slice_range': slice_list,
                "Diameters" : diameter,
                "Category" : category_name,
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
                    # print(slice_num)
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
    patient_tracking = colect3d(merged_dict)
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