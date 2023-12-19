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
import openvino.inference_engine.constants

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

# FOR DETECTION PART
def load_to_IE(xml_path = XML_PATH_DETECTION, bin_path = BIN_PATH_DETECTION):
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
    exec_net = ie.load_network(network= net, device_name="CPU")
    # exec_net = ie.load_network(network= net, device_name="GPU")

    return exec_net, input_shape, net_outputs

def do_inference(exec_net, image):
    input_blob = next(iter(exec_net.inputs))
    return exec_net.infer({input_blob: image})

# FOR FP REDUCTION
def load_to_IE_FP(xml_path = XML_PATH_FP, bin_path = BIN_PATH_FP):
    xml_path = os.path.abspath(xml_path)
    bin_path = os.path.abspath(bin_path)
    if not (os.path.isfile(xml_path) and os.path.isfile(bin_path)):
        raise Exception("Error %s/ %s do not exist" % (xml_path, bin_path))
    
    ie = IECore()
    # Loading IR files
    # net = IENetwork(model = xml_path, weights= bin_path)
    net = ie.read_network(
        model=xml_path, 
        weights=bin_path
    )

    net_outputs = list(net.outputs.keys())
    # Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU")
    # exec_net = ie.load_network(network=net, device_name="GPU")

    return exec_net, net_outputs

def get_image_3D(dataset, image_name, number_of_channels):
    image_center_index = dataset.index_of_image[image_name]
    number_of_channels_remain = number_of_channels - 1
    
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

def get_patch(dataset, bbox):
    large_shape = [80, 80, 30]  # height, width, depth (should be divisible by 2)
    medium_shape = [60, 60, 20] # height, width, depth (should be divisible by 2)
    small_shape = [40, 40, 10]  # height, width, depth (should be divisible by 2)
    final_shape = [40, 40, 10]

    # Load 3D image
    image = get_image_3D(dataset, bbox[0], large_shape[2])
    image = image/255

    # Extract the patch based on the bbox center
    x_center = round((float(bbox[1])+float(bbox[3])) / 2)
    y_center = round((float(bbox[2])+float(bbox[4])) / 2)

    large_patch = image[int(max(y_center - (large_shape[0]/2), 0)) : int(min(y_center + (large_shape[0]/2), 512)), 
                        int(max(x_center - (large_shape[1]/2), 0)) : int(min(x_center + (large_shape[1]/2), 512)),
                        :]
    
    # Add padding if the patch is too small
    if np.any(large_patch.shape[0:2]!=(large_shape[0], large_shape[1])):
        image_width, image_height, _ = image.shape
        large_patch = np.pad(large_patch, ((max(int(large_shape[0]/2)-y_center, 0), int(large_shape[0]/2) - min(image_height-y_center, int(large_shape[0]/2))),
                                           (max(int(large_shape[1]/2)-x_center, 0), int(large_shape[1]/2) - min(image_width-x_center, int(large_shape[1]/2))),
                                           (0, 0)), mode='constant', constant_values=0.)

    # Crop and resize the patch for different levels of contextual information
    medium_patch = large_patch[round((large_shape[0]-medium_shape[0])/2) : -round((large_shape[0]-medium_shape[0])/2),
                                round((large_shape[1]-medium_shape[1])/2) : -round((large_shape[1]-medium_shape[1])/2),
                                round((large_shape[2]-medium_shape[2])/2) : -round((large_shape[2]-medium_shape[2])/2)]

    small_patch = large_patch[round((large_shape[0]-small_shape[0])/2) : -round((large_shape[0]-small_shape[0])/2),
                                round((large_shape[1]-small_shape[1])/2) : -round((large_shape[1]-small_shape[1])/2),
                                round((large_shape[2]-small_shape[2])/2) : -round((large_shape[2]-small_shape[2])/2)]

    large_patch_resized = nd.interpolation.zoom(large_patch, zoom=(final_shape[0]/large_shape[0],
                                                                    final_shape[1]/large_shape[1],
                                                                    final_shape[2]/large_shape[2]), 
                                                                mode='nearest')

    medium_patch_resized = nd.interpolation.zoom(medium_patch, zoom=(final_shape[0]/medium_shape[0],
                                                                    final_shape[1]/medium_shape[1],
                                                                    final_shape[2]/medium_shape[2]), 
                                                                mode='nearest')
	
    small_patch_resized = nd.interpolation.zoom(small_patch, zoom=(final_shape[0]/small_shape[0],
                                                                    final_shape[1]/small_shape[1],
                                                                    final_shape[2]/small_shape[2]), 
                                                                mode='nearest')
    
    large_patch_resized = np.transpose(large_patch_resized, (2, 0, 1))
    medium_patch_resized = np.transpose(medium_patch_resized, (2, 0, 1))
    small_patch_resized = np.transpose(small_patch_resized, (2, 0, 1))

    large_patch_resized = np.reshape(large_patch_resized, (1, 10, 40, 40))
    medium_patch_resized = np.reshape(medium_patch_resized, (1, 10, 40, 40))
    small_patch_resized = np.reshape(small_patch_resized, (1, 10, 40, 40))
    
    return large_patch_resized, medium_patch_resized, small_patch_resized

def detect(save_img=False):
    output, source, save_txt = opt.output, opt.source, opt.save_txt
    thresh_conf, thresh_iou = opt.conf_thres, opt.iou_thres
    pickData = source.endswith('.pickle')
    # Detection
    net_1, input_shape, net_outputs_1 = load_to_IE()
    image_size = tuple(input_shape[-2:])

    # FP Reduction
    net_fp, net_outputs_fp = load_to_IE_FP()

    # Set Dataloader
    if pickData:
        dataset = LoadPickle(source, img_size=image_size, autosize=64)
    else:
        save_img = True
        dataset = LoadImages(source, image_size, auto_size= 64)

    # Run inference
    t0 = time.time()
    img = np.zeros(input_shape, dtype=np.float32)
    _ = do_inference(net_1, img)

    string_results = ""
    results = []
    print("Detecting Nodule")
    for path, image, im0s, vid_cap in dataset:
        img = cv.dnn.blobFromImage(image, 1.0/255.0, image_size, (1,1,1), True)
        pred = do_inference(net_1, img)
        key = net_outputs_1[-1]
        pred = pred[key]
        
        # preds = non_max_suppression(pred, thresh_conf, thresh_iou)
        preds = non_max_suppression(pred, 0.1, 0.1)
        preds = preds[0]
        if preds is not None and len(preds):
            string_results += path
            for *xyxy, conf, cls in preds:
                string_results += " %g,%g,%g,%g,%g,%g" %(*xyxy, cls, conf)
                results.append([path, *xyxy, cls, conf])
            string_results += "\n"
    
    if opt.save_txt:
        print(string_results)
        with open(os.path.join(str(Path(output)), "inference.txt"), "w+") as f:
            f.write(string_results)
    print("Reducing FP")
    number_of_bbox = len(results)
    results_filename = []
    results_bbox = []
    finish_count = 0

    for bbox in results:
        large, medium, small = get_patch(dataset, bbox)

        prediction = net_fp.infer({'input_1': large, 'input_2': medium, 'input_3': small})
        # prediction = do_inference(net, large, medium, small)
        prediction = prediction[net_outputs_fp[-1]][0][0]

        probability = 0.2*float(bbox[6]) + 0.8*prediction
    
        if probability >= 0.3:
            if bbox[0] not in results_filename:
                results_filename.append(bbox[0])
                results_bbox.append([])
            
            results_index = results_filename.index(bbox[0])
            # results_bbox[results_index].append(bbox[1]+','+bbox[2]+','+bbox[3]+','+bbox[4]+','+bbox[5]+','+bbox[6])
            results_bbox[results_index].append("{:.3f},{:.3f},{:.3f},{:.3f},{},{:.6f}".format(*bbox[1:]))
        
        finish_count += 1
        print(str(finish_count) + ' / ' + str(number_of_bbox), end='\r')

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

