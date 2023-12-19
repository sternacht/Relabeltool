import argparse
import os
import cv2 as cv
import numpy as np
from pathlib import Path
import pickle
import time
import glob
import math
from utils.general import non_max_suppression
from scipy import ndimage as nd
from openvino.inference_engine import IECore

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, auto_size=32):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.auto_size = auto_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size, auto_size=self.auto_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv.VideoCapture(path)
        self.nframes = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

class LoadPickle:
    def __init__(self, path:str, img_size:tuple = (512, 512), autosize:int = 32):
        p = str(Path(path))
        p = os.path.abspath(p)
        if os.path.isfile(p):
            files = p
        else:
            raise Exception("Error: %s does not exist" % p)
        self.img_size = img_size
        self.autosize = autosize
        try:
            self.data = pickle.load(open(files, 'rb'))
            self.data = list(self.data.items())
        except:
            raise Exception("Error: No load data %s" % p)
        self.nf = len(self.data)
        self.mode = 'images'

        self.index_of_image = {}
        for i in range(len(self.data)):
            path, _ = self.data[i]
            self.index_of_image[path] = i
        pass

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path, img0 = self.data[self.count]
        self.count += 1

        #Padded resize
        img = letterbox(img0, new_shape= self.img_size, auto_size=self.autosize)[0]

        #Convert
        # img = img.transpose(2,0,1)
        img = np.ascontiguousarray(img)
        return path, img, img0, None

    def __len__(self):
        return self.nf

    def __getitem__(self, index):
        path, img = self.data[index]
        return path, img

XML_PATH_DETECTION = r"weights\FP16\best.xml"
BIN_PATH_DETECTION = r"weights\FP16\best.bin"
XML_PATH_FP = r"weights\FP_Reduction_FP16\fp_reduction_saved_model.xml"
BIN_PATH_FP = r"weights\FP_Reduction_FP16\fp_reduction_saved_model.bin"
NUM_REQUESTS = 4

# FOR DETECTION PART
def load_to_IE(num_requests, xml_path = XML_PATH_DETECTION, bin_path = BIN_PATH_DETECTION):
    xml_path = os.path.abspath(xml_path)
    bin_path = os.path.abspath(bin_path)
    if not (os.path.isfile(xml_path) and os.path.isfile(bin_path)):
        raise Exception("Error %s/ %s do not exist" % (xml_path, bin_path))
    ie = IECore()
    # Loading IR files
    net = ie.read_network(model = xml_path, weights= bin_path)

    input_shape = net.inputs['input'].shape
    net_outputs = list(net.outputs.keys())
    # Loading the network to the inference engine
    exec_net = ie.load_network(network= net, device_name="CPU", num_requests=num_requests, config={"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"})
    # exec_net = ie.load_network(network= net, device_name="GPU", num_requests=num_requests)

    return exec_net, input_shape, net_outputs

def do_inference(exec_net, image):
    input_blob = next(iter(exec_net.inputs))
    return exec_net.infer({input_blob: image})

# FOR FP REDUCTION
def load_to_IE_FP(num_requests, xml_path = XML_PATH_FP, bin_path = BIN_PATH_FP):
    xml_path = os.path.abspath(xml_path)
    bin_path = os.path.abspath(bin_path)
    if not (os.path.isfile(xml_path) and os.path.isfile(bin_path)):
        raise Exception("Error %s/ %s do not exist" % (xml_path, bin_path))
    
    ie = IECore()
    # Loading IR files
    net = ie.read_network(model=xml_path, weights=bin_path)

    net_outputs = list(net.outputs.keys())
    # Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU", num_requests=num_requests, config={"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"})
    # exec_net = ie.load_network(network=net, device_name="GPU", num_requests=num_requests)

    return exec_net, net_outputs

# FOR FP REDUCTION
class fp_reduction_preprocess:
    def __init__(self):
        self.large_shape = [80, 80, 30]  # height, width, depth (should be divisible by 2)
        self.medium_shape = [60, 60, 20] # height, width, depth (should be divisible by 2)
        self.small_shape = [40, 40, 10]  # height, width, depth (should be divisible by 2)
        self.final_shape = [40, 40, 10]

    def get_image_3D(self, dataset, image_name):
        image_center_index = dataset.index_of_image[image_name]
        number_of_channels_remain = self.large_shape[2] - 1
        
        image_lower_bound_index = image_center_index - math.ceil(number_of_channels_remain/2)
        image_upper_bound_index = image_center_index + math.floor(number_of_channels_remain/2)

        image_each_channel = []

        for i in range(image_lower_bound_index, image_upper_bound_index+1):
            if i >= 0 and i < len(dataset):
                _, image = dataset[i]
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                image_each_channel.append(image)
            else:
                if i < 0:
                    _, image = dataset[0]
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    image_each_channel.append(image)
                else:
                    image_each_channel.append(np.copy(image_each_channel[-1]))
        
        return np.stack((image_each_channel), axis=-1) # return 3D image

    def get_patch(self, dataset, bbox):
        # Load 3D image
        image = self.get_image_3D(dataset, bbox[0])
        image = image/255

        # Extract the patch based on the bbox center
        x_center = round((float(bbox[1])+float(bbox[3])) / 2)
        y_center = round((float(bbox[2])+float(bbox[4])) / 2)

        large_patch = image[int(max(y_center - (self.large_shape[0]/2), 0)) : int(min(y_center + (self.large_shape[0]/2), image.shape[0])), 
                            int(max(x_center - (self.large_shape[1]/2), 0)) : int(min(x_center + (self.large_shape[1]/2), image.shape[1])),
                            :]
        
        # Add padding if the patch is too small
        if np.any(large_patch.shape[0:2]!=(self.large_shape[0], self.large_shape[1])):
            image_width, image_height, _ = image.shape
            large_patch = np.pad(large_patch, ((max(int(self.large_shape[0]/2)-y_center, 0), int(self.large_shape[0]/2) - min(image_height-y_center, int(self.large_shape[0]/2))),
                                               (max(int(self.large_shape[1]/2)-x_center, 0), int(self.large_shape[1]/2) - min(image_width-x_center, int(self.large_shape[1]/2))),
                                               (0, 0)), mode='constant', constant_values=0.)

        # Crop and resize the patch for different levels of contextual information
        medium_patch = large_patch[round((self.large_shape[0]-self.medium_shape[0])/2) : -round((self.large_shape[0]-self.medium_shape[0])/2),
                                   round((self.large_shape[1]-self.medium_shape[1])/2) : -round((self.large_shape[1]-self.medium_shape[1])/2),
                                   round((self.large_shape[2]-self.medium_shape[2])/2) : -round((self.large_shape[2]-self.medium_shape[2])/2)]

        small_patch = large_patch[round((self.large_shape[0]-self.small_shape[0])/2) : -round((self.large_shape[0]-self.small_shape[0])/2),
                                  round((self.large_shape[1]-self.small_shape[1])/2) : -round((self.large_shape[1]-self.small_shape[1])/2),
                                  round((self.large_shape[2]-self.small_shape[2])/2) : -round((self.large_shape[2]-self.small_shape[2])/2)]

        large_patch_resized = nd.interpolation.zoom(large_patch, zoom=(self.final_shape[0]/self.large_shape[0],
                                                                       self.final_shape[1]/self.large_shape[1],
                                                                       self.final_shape[2]/self.large_shape[2]), 
                                                                 mode='nearest')

        medium_patch_resized = nd.interpolation.zoom(medium_patch, zoom=(self.final_shape[0]/self.medium_shape[0],
                                                                         self.final_shape[1]/self.medium_shape[1],
                                                                         self.final_shape[2]/self.medium_shape[2]), 
                                                                   mode='nearest')
        
        small_patch_resized = nd.interpolation.zoom(small_patch, zoom=(self.final_shape[0]/self.small_shape[0],
                                                                       self.final_shape[1]/self.small_shape[1],
                                                                       self.final_shape[2]/self.small_shape[2]), 
                                                                 mode='nearest')
        
        large_patch_resized = np.transpose(large_patch_resized, (2, 0, 1))
        medium_patch_resized = np.transpose(medium_patch_resized, (2, 0, 1))
        small_patch_resized = np.transpose(small_patch_resized, (2, 0, 1))

        large_patch_resized = np.reshape(large_patch_resized, (1, 10, 40, 40))
        medium_patch_resized = np.reshape(medium_patch_resized, (1, 10, 40, 40))
        small_patch_resized = np.reshape(small_patch_resized, (1, 10, 40, 40))
        
        return large_patch_resized, medium_patch_resized, small_patch_resized

    def calculate_distance(self, bbox1, bbox2):
        x1 = (float(bbox1[1])+float(bbox1[3])) / 2
        y1 = (float(bbox1[2])+float(bbox1[4])) / 2
        x2 = (float(bbox2[1])+float(bbox2[3])) / 2
        y2 = (float(bbox2[2])+float(bbox2[4])) / 2

        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def collect_3D_nodule(self, results):
        series = []
        series_number = []

        for i in range(len(results)):
            if len(series) == 0:
                series.append([results[i]])
                series_number.append([int(results[i][0].split('-')[-1])])
                continue
            
            distances = []
            for j in range(len(series)):
                image_number = int(results[i][0].split('-')[-1])
                if image_number-series_number[j][-1] <= 5:
                    distances.append(self.calculate_distance(series[j][-1], results[i]))
                else:
                    distances.append(1000)
            
            if min(distances) <= 25:
                series[distances.index(min(distances))].append(results[i])
                series_number[distances.index(min(distances))].append(int(results[i][0].split('-')[-1]))
            else:
                series.append([results[i]])
                series_number.append([int(results[i][0].split('-')[-1])])
        
        return series

    def calculate_probability_threshold(self, number_of_bboxes):
        if number_of_bboxes < 3:
            # return 0.4
            # return 0.35
            return 0.3
        elif number_of_bboxes > 20:
            return 0.1
        else:
            # return 77/170 - (3/170)*number_of_bboxes
            # return 26.8/68 - (1/68)*number_of_bboxes
            return 28.5/85 - (1/85)*number_of_bboxes

class Lung_segmentation:
    def __init__(self, dataset_length):
        self.num_img=dataset_length
        self.lung_msk_list=np.zeros((self.num_img,512,512),np.uint8)

    def lung_segment(self,dataset,start_index,end_index,step,state):
        pervious_corr=[[0,0],[0,0]]
        # seg_result=np.zeros((512,512),np.uint8)

        #for index in range(math.floor(self.num_img/2),self.num_img):# half slice number to number
        #for index in range(math.floor(self.num_img/2),0,-1):# half slice number to number
        for index in range(start_index,end_index,step):# half slice number to number
            _, seg_result=dataset[index]
            seg_result=cv.cvtColor(seg_result, cv.COLOR_RGB2GRAY)
            blur=cv.medianBlur(seg_result,11)

            #blur = cv.GaussianBlur(seg_result, (5, 5), 0)
            #blur=cv.medianBlur(blur,21)
            #cv.imshow("blur",blur)
            #cv.waitKey(0)
            #_, binary=cv.threshold(blur, 90, 255, cv.THRESH_BINARY_INV)
            _, binary=cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            
            #cv.imshow('binary',binary)
            kernel = np.ones((5,5), np.uint8)
            #kernel2 = np.ones((3,3), np.uint8)
            dilation = cv.dilate(binary, kernel, iterations = 1)
            erosion = cv.erode(dilation, kernel, iterations = 1)
            nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(erosion, None, None, None, connectivity=8, ltype=None)
            temp=np.zeros((nlabels,1),np.uint8)
            for index2 in range(nlabels):
                temp[index2]=index2
            zipped=np.insert(stats,0,temp[:,0],axis=1)
            zipped=np.insert(zipped,6,centroids[:,0],axis=1)
            zipped=np.insert(zipped,7,centroids[:,1],axis=1)
            if index>=self.num_img*0.8:
                sort_CClabel_result=sorted(zipped,key = lambda t: t[-4]*t[-5],reverse=True)
                #sort_CClabel_result=sorted(zipped,key = lambda t: t[-3],reverse=True)
            else:
                sort_CClabel_result=sorted(zipped,key = lambda t: t[-3],reverse=True)

            lung_candidate=[]
            for index2 in range(len(sort_CClabel_result)):
                #if self.check_imgvalue(labels,sort_CClabel_result[index2],erosion)and sort_CClabel_result[index2][1]>=2 and sort_CClabel_result[index2][2]>=2 and sort_CClabel_result[index2][3]<500 and sort_CClabel_result[index2][4]<500:
                if sort_CClabel_result[index2][0]!=0 and sort_CClabel_result[index2][1]>=2 and sort_CClabel_result[index2][2]>=2 and sort_CClabel_result[index2][1]<=505 and sort_CClabel_result[index2][2]<=505 and sort_CClabel_result[index2][3]<=500 and sort_CClabel_result[index2][4]<=500:
                    lung_candidate.append(sort_CClabel_result[index2])

            whole_msk=labels.copy()
            #print(len(lung_candidate))
            if len(lung_candidate)==0:
                break
            else:
                if lung_candidate[0][3]>=300:#connect together
                    whole_msk=self.create_lungmsk(whole_msk,lung_candidate[0])
                    #print(lung_candidate[0])
                    #print("connect together:"+str(index))
                else: 
                    #print("lung_candidate:"+str(len(lung_candidate)))
                    if len(lung_candidate)>=2:
                        if lung_candidate[0][1]>lung_candidate[1][1]:
                            left_lung=lung_candidate[1]
                            right_lung=lung_candidate[0]
                        else:
                            left_lung=lung_candidate[0]
                            right_lung=lung_candidate[1]
                    

                        if pervious_corr[0][0]==0:
                            pervious_corr[0][0]=left_lung[-2];pervious_corr[0][1]=left_lung[-1]
                            pervious_corr[1][0]=right_lung[-2];pervious_corr[1][1]=right_lung[-1]
                        else:
                            if state:
                                if self.comput_distance(pervious_corr[0],left_lung)<=50 and self.comput_distance(pervious_corr[1],right_lung)<=50:
                                    pervious_corr[0][0]=left_lung[-2];pervious_corr[0][1]=left_lung[-1]
                                    pervious_corr[1][0]=right_lung[-2];pervious_corr[1][1]=right_lung[-1]
                                    #print("Lung split"+str(index))
                                else:
                                    #print("Lung Disappear"+str(index))
                                    break
                            else:
                                if self.comput_distance(pervious_corr[0],left_lung)<=50 or self.comput_distance(pervious_corr[1],right_lung)<=50:
                                    pervious_corr[0][0]=left_lung[-2];pervious_corr[0][1]=left_lung[-1]
                                    pervious_corr[1][0]=right_lung[-2];pervious_corr[1][1]=right_lung[-1]
                                    #print("Lung split"+str(index))
                                else:
                                    #print("Lung Disappear"+str(index))
                                    break
                        
                    
                        left_msk=labels.copy()
                        right_msk=labels.copy()
                        left_msk=self.create_lungmsk(left_msk,left_lung)
                        right_msk=self.create_lungmsk(right_msk,right_lung)
                        whole_msk=cv.bitwise_or(left_msk, right_msk)
                    elif len(lung_candidate)>=1:
                        whole_msk=self.create_lungmsk(whole_msk,lung_candidate[0])
                    else:
                        continue

                whole_msk = whole_msk.astype('uint8')
                kernel = np.ones((7,7), np.uint8)
                whole_msk = cv.dilate(whole_msk, kernel, iterations = 1)
                lung_segment_result=whole_msk
                #lung_segment_result=whole_msk*seg_result
                self.lung_msk_list[index]=lung_segment_result
                #cv.imshow("result",lung_segment_result)
                #cv.waitKey(0)

    def comput_distance(self,corr,corr2):
        distance=((corr[0]-corr2[-2])**2+(corr[1]-corr2[-1])**2)**0.5
        return distance

    def create_lungmsk(self,label,lung_info):
        label_num=lung_info[0]
        label[label!=label_num]=0
        label[label>0]=255
        return label
    
    def calculate_lung_mask_coverage(self, mask_index, bbox_coordinates):
        x1 = int(min(max(0, round(bbox_coordinates[0])), 511))
        y1 = int(min(max(0, round(bbox_coordinates[1])), 511))
        x2 = int(min(max(0, round(bbox_coordinates[2])), 511))
        y2 = int(min(max(0, round(bbox_coordinates[3])), 511))

        mask = self.lung_msk_list[mask_index]

        # DEBUG
        # color_mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        # color_mask = cv.rectangle(color_mask, (x1, y1), (x2, y2), (0, 255, 0))
        # cv.imshow("result", color_mask)
        # cv.waitKey(0)

        total_region = (x2-x1+1) * (y2-y1+1)
        cover_region = 0
        for i in range(y1, y2+1, 1):
            for j in range(x1, x2+1, 1):
                if mask[i][j] == 255:
                    cover_region += 1
        
        return cover_region / total_region

def detect(save_img=False):
    output, source, save_txt = opt.output, opt.source, opt.save_txt
    thresh_conf, thresh_iou = opt.conf_thres, opt.iou_thres
    pickData = source.endswith('.pickle')
    # Detection
    net_1, input_shape, net_outputs_1 = load_to_IE(NUM_REQUESTS)
    image_size = tuple(input_shape[-2:])

    # FP Reduction
    net_fp, net_outputs_fp = load_to_IE_FP(NUM_REQUESTS)

    # Set Dataloader
    if pickData:
        dataset = LoadPickle(source, img_size=image_size, autosize=64)
    else:
        save_img = True
        dataset = LoadImages(source, image_size, auto_size= 64)

    # Run inference
    img = np.zeros(input_shape, dtype=np.float32)
    input_blob = next(iter(net_1.inputs))

    # preprocess input data
    images_preprocess = []
    for path, image, _, _ in dataset:
        img = cv.dnn.blobFromImage(image, 1.0/255.0, image_size, (1,1,1), True)
        images_preprocess.append([path, img])

    # prerequisites for OpenVINO asynchronous inference
    request_id = -1 # index for current inference request
    request_info_buffer = ['' for i in range(NUM_REQUESTS)] # buffer for storing request information
    
    # record results from stage 1
    string_results = ""
    results = []

    print("Detecting Nodule")

    detect_length = len(dataset)
    for path, img in images_preprocess:
        request_id = (request_id+1) % NUM_REQUESTS # current inference request

        # if the current inference request has been occupied
        if request_info_buffer[request_id] != '':
            # wait until the request is finished
            # and get the inference results
            net_1.requests[request_id].wait()
            pred = net_1.requests[request_id].output_blobs[net_outputs_1[-1]].buffer
            pred_path = request_info_buffer[request_id]
            
            # postprocess inference results
            # preds = non_max_suppression(pred, thresh_conf, thresh_iou)
            preds = non_max_suppression(pred, 0.1, 0.1)
            preds = preds[0]
            if preds is not None and len(preds):
                string_results += pred_path
                for *xyxy, conf, cls in preds:
                    string_results += " %g,%g,%g,%g,%g,%g" %(*xyxy, cls, conf)
                    results.append([pred_path, *xyxy, cls, conf])
                string_results += "\n"
            
            # clean the buffer
            request_info_buffer[request_id] = ''

        # send an inference request
        net_1.requests[request_id].async_infer({input_blob: img})
        request_info_buffer[request_id] = path
    
    # retrieve the remaining inference results
    for request_id in range(NUM_REQUESTS):
        # if the inference request hasn't been cleaned
        if request_info_buffer[request_id] != '':
            # wait until the request is finished
            # and get the inference results
            net_1.requests[request_id].wait()
            pred = net_1.requests[request_id].output_blobs[net_outputs_1[-1]].buffer
            pred_path = request_info_buffer[request_id]
            
            # postprocess inference results
            # preds = non_max_suppression(pred, thresh_conf, thresh_iou)
            preds = non_max_suppression(pred, 0.1, 0.1)
            preds = preds[0]
            if preds is not None and len(preds):
                string_results += pred_path
                for *xyxy, conf, cls in preds:
                    string_results += " %g,%g,%g,%g,%g,%g" %(*xyxy, cls, conf)
                    results.append([pred_path, *xyxy, cls, conf])
                string_results += "\n"
            
            # clean the buffer
            request_info_buffer[request_id] = ''
    
    if opt.save_txt:
        print(string_results)
        with open(os.path.join(str(Path(output)), "inference.txt"), "w+") as f:
            f.write(string_results)
    
    print("Reducing FP")
    fp_reduction_preprocessor = fp_reduction_preprocess()
    lung_segmenter = Lung_segmentation(detect_length)

    lung_segmenter.lung_segment(dataset, math.floor(detect_length/2), 0, -1, True) # segment lung region from the middle to the top
    lung_segmenter.lung_segment(dataset, math.floor(detect_length/2), detect_length, 1, False) # segment lung region from the middle to the top
    
    # record results from stage 2
    results_filename = []
    results_bbox = []
    fp_result_bbox_count = 0

    # group results
    result_groups = fp_reduction_preprocessor.collect_3D_nodule(results)

    # preprocess input data
    result_group_length = []
    images_preprocess = []
    for group in result_groups:
        temp = []
        for bbox in group:
            large, medium, small = fp_reduction_preprocessor.get_patch(dataset, bbox)
            temp.append([large, medium, small])
        
        images_preprocess.append(temp)
        result_group_length.append(len(group))

    # prerequisites for OpenVINO asynchronous inference
    request_info_buffer = [[] for i in range(NUM_REQUESTS)]
    for i in range(len(result_groups)):
        # adjust the probability threshold for each group
        threshold = fp_reduction_preprocessor.calculate_probability_threshold(result_group_length[i])

        for j in range(result_group_length[i]):
            request_id = (request_id+1) % NUM_REQUESTS # current inference request
            
            # if the current inference request has been occupied
            if len(request_info_buffer[request_id]) != 0:
                # wait until the request is finished
                # and get the inference results
                net_fp.requests[request_id].wait()
                prediction = net_fp.requests[request_id].output_blobs[net_outputs_fp[-1]].buffer[0][0]
                prediction_bbox_threshold = request_info_buffer[request_id]

                # postprocess inference results
                probability = 0.3*float(prediction_bbox_threshold[6]) + 0.7*prediction
            
                if probability >= prediction_bbox_threshold[7]:
                    if prediction_bbox_threshold[0] not in results_filename:
                        results_filename.append(prediction_bbox_threshold[0])
                        results_bbox.append([])
                    
                    results_index = results_filename.index(prediction_bbox_threshold[0])
                    # results_bbox[results_index].append(bbox[1]+','+bbox[2]+','+bbox[3]+','+bbox[4]+','+bbox[5]+','+bbox[6])
                    results_bbox[results_index].append("{:.3f},{:.3f},{:.3f},{:.3f},{},{:.6f}".format(*prediction_bbox_threshold[1:7]))
                    fp_result_bbox_count += 1

                # clean the buffer
                request_info_buffer[request_id] = []
            
            # send an inference request
            net_fp.requests[request_id].async_infer({'input_1': images_preprocess[i][j][0], 'input_2': images_preprocess[i][j][1], 'input_3': images_preprocess[i][j][2]})
            request_info_buffer[request_id] = result_groups[i][j] + [threshold]

    # retrieve the remaining inference results
    for request_id in range(NUM_REQUESTS):
        # if the inference request hasn't been cleaned
        if len(request_info_buffer[request_id]) != 0:
            # wait until the request is finished
            # and get the inference results
            net_fp.requests[request_id].wait()
            prediction = net_fp.requests[request_id].output_blobs[net_outputs_fp[-1]].buffer[0][0]
            prediction_bbox_threshold = request_info_buffer[request_id]

            # postprocess inference results
            probability = 0.3*float(prediction_bbox_threshold[6]) + 0.7*prediction
        
            if probability >= prediction_bbox_threshold[7]:
                if prediction_bbox_threshold[0] not in results_filename:
                    results_filename.append(prediction_bbox_threshold[0])
                    results_bbox.append([])
                
                results_index = results_filename.index(prediction_bbox_threshold[0])
                # results_bbox[results_index].append(bbox[1]+','+bbox[2]+','+bbox[3]+','+bbox[4]+','+bbox[5]+','+bbox[6])
                results_bbox[results_index].append("{:.3f},{:.3f},{:.3f},{:.3f},{},{:.6f}".format(*prediction_bbox_threshold[1:7]))
                fp_result_bbox_count += 1

            # clean the buffer
            request_info_buffer[request_id] = []

    string_results = ''
    for i in range(len(results_filename)):
        string_results += (results_filename[i] + ' ')

        for bbox in results_bbox[i]:
            string_results += (bbox + ' ')
        
        string_results += '\n'
    
    if opt.save_txt:
        print(string_results)
        with open(os.path.join(str(Path(output)), "inference_FP_reduction.txt"), "w+") as f:
            f.write(string_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=r"D:\02_BME\data\NCKUH\0003\image.pickle", help= "Source")
    parser.add_argument('--output', type=str, default=r"D:\02_BME\data\NCKUH\0003", help= "Output")
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    detect()

