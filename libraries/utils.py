import os
import re
import cv2 as cv
import json
import pandas as pd
import numpy as np


def split_file_name_type(ff):
    return os.path.splitext(os.path.basename(ff))[0]

def split_directory_to_list(ff):
    return os.path.normpath(ff).split(os.path.sep)

def normalize_x_y(x,y):
    region_x1=x-50
    region_y1=y-50
    region_x2=x+50
    region_y2=y+50  
    if region_x1<0:
        region_x1=0
        region_x2=100
    if region_y1<0:
        region_y1=0
        region_y2=100
    if region_x2>511:
        region_x2=511
        region_x1=411
    if region_y2>511:
        region_y2=511
        region_y1=411
        
    return region_x1,region_x2,region_y1,region_y2

def compute_iou(bb, gt):
    bb_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    sum_area = bb_area + gt_area
    it = [0, 0, 0, 0]
    it[0] = max(bb[0], gt[0])
    it[1] = max(bb[1], gt[1])
    it[2] = min(bb[2], gt[2])
    it[3] = min(bb[3], gt[3])
    if it[0] >= it[2] or it[1] >= it[3]:
        return 0
    else:
        it_area = (it[2] - it[0]) * (it[3] - it[1])
        return (it_area / (sum_area - it_area))*1.0

def compute_distance(bb, gt):
    it = [0, 0, 0, 0]
    it[0] = min(bb[0], gt[0])
    it[1] = min(bb[1], gt[1])
    it[2] = max(bb[2], gt[2])
    it[3] = max(bb[3], gt[3])
    cc = (it[2] - it[0])**2 + (it[3] - it[1])**2
    bx = (bb[2] + bb[0]) / 2
    by = (bb[3] + bb[1]) / 2
    gx = (gt[2] + gt[0]) / 2
    gy = (gt[3] + gt[1]) / 2
    pp = (bx - gx)**2 + (by - gy)**2
    return pp / cc

def get_segmented_two_lungs(im, plot=False):

    #Load libraries
    from skimage.segmentation import clear_border
    from skimage.measure import label, regionprops
    from skimage.morphology import disk, binary_erosion, binary_closing, binary_opening
    from skimage.filters import roberts
    from scipy import ndimage as ndi

    binary = im < 200
    cleared = clear_border(binary)
    label_image = label(cleared)
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area <= areas[-2] or region.area <= 20000:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    selem = disk(10)
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    return im, binary, get_high_vals

def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)

def drawImage(image, p1, p2, colorNum:int):
    if colorNum == 0:
        color = (0,0,255)
    elif colorNum == 1:
        color = (0,255,255)
    elif colorNum == 2:
        color = (255,255,0)
    elif colorNum == 3:
        color = (255,0,0)
    elif colorNum == 4:
        color = (128,0,128)
    else:
        color = (0,255,0)
    image = cv.rectangle(image, p1, p2, color, 1)
    return image

def read_csv(path:str):
    try:
        return pd.read_csv(path)
    except:
        return None

def write_csv(path:str, data:dict):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def removeVessel(pathInput, pathSave):
    if not os.path.exists(pathInput):
        return False
    BbList = []
    CompareList = []
    _distance = 0
    results = read_csv(pathInput)
    for result in results.values:
        basename = os.path.splitext(os.path.basename(result[0]))[0]
        x1, y1, x2, y2, score, nodule = result[1:7]
        foldername, nameIndex = basename.split('-')[-2:]
        BbList.append([basename, x1, y1, x2, y2, score, nodule, int(foldername), int(nameIndex)])
        CompareList.append([basename, x1, y1, x2, y2, score, nodule, int(foldername), int(nameIndex)])
    remove_vessel_data = {"filename": [],
                            "x1": [],
                            "y1": [],
                            "x2": [],
                            "y2": [],
                            "score":[],
                            "nodule":[]}
    for i in range(len(BbList)):
        flag = False
        canList = []
        for j in range(len(CompareList)):
            if (BbList[i][7] == CompareList[j][7] 
            and abs(BbList[i][8] - CompareList[j][8] <= 2)
            and abs(BbList[i][8] - CompareList[j][8] != 0)):
                canList.append(CompareList[j])
        for j in range(len(canList)):
            bb = (BbList[i][1], BbList[i][2], BbList[i][3], BbList[i][4])
            com = (CompareList[j][1], CompareList[j][2], CompareList[j][3], CompareList[j][4])
            distance = compute_distance(bb, com)
            if distance <= 0.1:
                flag = True
                if _distance < distance:
                    _distance = distance
        if flag:
            remove_vessel_data["filename"].append(BbList[i][0])
            remove_vessel_data["x1"].append(BbList[i][1])
            remove_vessel_data["y1"].append(BbList[i][2])
            remove_vessel_data["x2"].append(BbList[i][3])
            remove_vessel_data["y2"].append(BbList[i][4])
            remove_vessel_data["score"].append(BbList[i][5])
            remove_vessel_data["nodule"].append(BbList[i][6])
    write_csv(pathSave, remove_vessel_data)   

def removeVessel2(pathInput, pathSave):
    if not os.path.exists(pathInput):
        return False
    BbList = []
    CompareList = []
    _distance = 0
    results = read_csv(pathInput)
    for result in results.values:
        basename = os.path.splitext(os.path.basename(result[0]))[0]
        x1, y1, x2, y2, score, nodule = result[1:7]
        foldername, nameIndex = basename.split('-')[-2:]
        BbList.append([basename, x1, y1, x2, y2, score, nodule, int(foldername), int(nameIndex)])
        CompareList.append([basename, x1, y1, x2, y2, score, nodule, int(foldername), int(nameIndex)])
    remove_vessel_data = results.to_dict(orient='list')
    blood_vessel = []
    for i in range(len(BbList)):
        flag = False
        canList = []
        for j in range(len(CompareList)):
            if (BbList[i][7] == CompareList[j][7] 
            and abs(BbList[i][8] - CompareList[j][8] <= 2)
            and abs(BbList[i][8] - CompareList[j][8] != 0)):
                canList.append(CompareList[j])
        for j in range(len(canList)):
            bb = (BbList[i][1], BbList[i][2], BbList[i][3], BbList[i][4])
            com = (CompareList[j][1], CompareList[j][2], CompareList[j][3], CompareList[j][4])
            distance = compute_distance(bb, com)
            if distance <= 0.1:
                flag = True
                if _distance < distance:
                    _distance = distance
        blood_vessel.append(not flag)
    remove_vessel_data["blood vessel"] = blood_vessel
    write_csv(pathSave, remove_vessel_data)    

def defineNodule(x1, y1, x2, y2, score):
    temp_area = (x2 - x1) * (y2 - y1)
    if temp_area < 45 and float(score) >= 0.5:
        return 'Benign'
    elif temp_area < 100 and temp_area >= 45 and float(score) >= 0.1:
        return 'Prob. Benign'
    elif temp_area < 180 and temp_area >= 100 and float(score) >= 0.05:
        return 'Prob. Sus.'
    elif temp_area >= 180 and float(score) >= 0.01:
        return 'Sus.'
                            
def runDarknet(pathDarknet:str, dirname:str, imageList:list, extension = ".jpg"):
    ori_path = os.getcwd()
    os.chdir(pathDarknet)
    new_path = os.getcwd()
    if not os.path.exists(new_path + '/test'):
        os.mkdir(new_path + '/test')
    with open(new_path + '/test/test.txt', 'w') as testing_txt:
        for filename in imageList: 
            test_path = "{}/{}".format(dirname, filename + extension)
            print(test_path)
            testing_txt.writelines(test_path + "\n")
    testing_txt.close()
    # # os.system("chmod +x ./darknet")
    commands = 'darknet detector test init/init_data.data init/init_cfg.cfg init/init_cfg_best_Tony.weights -ext_output -dont_show -thresh 0.38 -out test/result.json < test/test.txt'
    print(commands)
    try:
        os.system(commands)
    except:
        os.chdir(ori_path)
        return False
    
    JsonPath = new_path + '/test/result.json'
    return os.path.abspath(JsonPath)

def runPytorchModel(pathdir:str):
    ori_path = os.getcwd()
    src_input = os.path.join(pathdir, "image.pickle")
    open_conda_env = "activate scaled_yolo"
    run_prog = "python detect2.py --source {} --output {} --save-txt".format(src_input, pathdir)
    close_conda_env = "conda deactivate"
    try: 
        # os.system(open_conda_env)
        
        os.system(open_conda_env + "&&" +run_prog)

        # os.system(close_conda_env)
    except:
        return False
    return os.path.join(pathdir, "inference.txt")

def runOpenvinoModel(pathdir:str):
    ori_path = os.getcwd()
    src_input = os.path.join(pathdir, "image.pickle")
    run_prog = "python inference_openvino_FP_modified.py --source {} --output {} --save-txt".format(src_input, pathdir)
    try: 
        # os.system(open_conda_env)
        
        os.system(run_prog)

        # os.system(close_conda_env)
    except:
        return False
    return os.path.join(pathdir, "inference_FP_reduction.txt")


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,1]
	y1 = boxes[:,2]
	x2 = boxes[:,3]
	y2 = boxes[:,4]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick]

def json2Csv(pathJson, pathcsv):
    if os.path.exists(pathJson):
        with open(pathJson) as jsonFile:
            jsonDatas = json.load(jsonFile)
        csvData = {"filename": [], "x1": [], "y1": [], 
                    "x2": [], "y2": [], "score": [], "nodule": []}
        filenames = []
        for jsonData in jsonDatas:
            if jsonData['objects']:
                filename = os.path.basename(jsonData['filename'])
                filenames.append(filename)
                for obj in range(0, len(jsonData['objects'])):
                    center_x = float(jsonData['objects'][obj]['relative_coordinates']['center_x'])
                    center_y = float(jsonData['objects'][obj]['relative_coordinates']['center_y'])
                    yolo_width = float(jsonData['objects'][obj]['relative_coordinates']['width'])
                    yolo_height = float(jsonData['objects'][obj]['relative_coordinates']['height'])                
                    x1 = round(0.5 * 512 * (2 * center_x - yolo_width))
                    y1 = round(0.5 * 512 * (2 * center_y - yolo_height))
                    x2 = round(0.5 * 512 * (2 * center_x + yolo_width))
                    y2 = round(0.5 * 512 * (2 * center_y + yolo_height))
                    csvData['filename'].append(filename)
                    csvData['x1'].append(x1)
                    csvData['y1'].append(y1)
                    csvData['x2'].append(x2)
                    csvData['y2'].append(y2)
                    csvData['score'].append(round(jsonData['objects'][obj]['confidence'],2))
                    csvData['nodule'].append(np.nan)
        df = pd.DataFrame(csvData)
        for filename in filenames:
            miniData = df[df['filename'] == filename]
            boxes = []
            for index, values in miniData.iterrows():
                boxes.append([index, values[1], values[2], values[3], values[4], values[5]])
            boxes = np.array(boxes)
            nmsbox = non_max_suppression_fast(boxes, 0.5)
            for values in nmsbox:
                nodule = defineNodule(values[1], values[2], values[3], values[4], values[5])
                df.loc[values[0], 'nodule'] = nodule
        df.dropna(inplace=True)
        df.to_csv(pathcsv, index= None)

# def mergeDataFrame(dataframeResult, dataframeGT):
#     if dataframeResult.empty or dataframeGT.empty:
#         return None

#     dataframeResult['filename'] = [split_file_name_type(filename) for filename in dataframeResult['filename']]
#     unique_filenames =(dataframeGT['filename'].append(dataframeResult['filename'])).sort_values().unique()
#     labels = ['nodule']
#     # result = dataframeResult.merge(dataframeGT, how='outer').sort_index(axis=1)
#     evaluation = ObjectDetectMetric(labels)
#     for filename in unique_filenames:
#         result = dataframeResult[dataframeResult['filename'] == filename].values
#         gt = dataframeGT[dataframeGT['filename'] == filename].values
#         prediction = result[:, 1:6]
#         ground_truth = gt[:, 1:]
#         labels_prediction = [len(labels) -1] * len(result)
#         labels_groundtruth = [len(labels) -1] * len(gt)
#         evaluation._update(ground_truth, labels_groundtruth, prediction, labels_prediction, filename)
#     _ = evaluation.get_confusion(0.5, 0.5, 0.6, 0.6)
#     tableGt = evaluation.get_table()
#     table_recall_precision = evaluation.get_table_recall_precision()
#     return tableGt, table_recall_precision



