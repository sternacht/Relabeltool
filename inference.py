import argparse
import os
import platform
import shutil
import cv2 as cv
import numpy as np
from pathlib import Path
import time
import glob
import xml.etree.cElementTree as et
from lxml import etree
import codecs

import torch
from utils.general import non_max_suppression
from openvino.inference_engine import IECore, IENetwork
import utils.mergeOverlapping as olap

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']

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

def get_patient_info(path:str):
    filename = os.path.splitext(os.path.basename(path))[0]
    patient_id, no_slice = filename.split("-")[-2:]
    return int(patient_id), int(no_slice), path

def get_filename(path:str):
    filename = os.path.splitext(os.path.basename(path))[0]
    return filename

def get_patient_dict(list_patient_filename:list):
    dict_patient = {}
    for path in list_patient_filename:
        patient_id, no_slice, path = get_patient_info(path)
        if patient_id not in dict_patient.keys():
            dict_patient[patient_id] = {no_slice : path}
        else:
            dict_patient[patient_id][no_slice] = path
    return dict_patient

class LoadFolder:
    def __init__(self, path:str, img_size:tuple = (512, 512), auto_size:int = 32, extension = ".jpg"):
        p = str(Path(path))
        p = os.path.abspath(p)
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        self.patient_info = get_patient_dict(images)
        self.img_size = img_size
        self.auto_size = auto_size

        self.nf = len(images)
        self.files = images
        self.mode = "images"
        assert self.nf > 0, "No images in %s. Supported formats are:\nimages: %s" % (p, img_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        self.count += 1
        img0 = cv.imread(path)
        assert img0 is not None, "Image Not Found " + path 
        if len(img0) < 3:
            patient_id, no_slice, path = get_patient_info(path)
            if no_slice -1 in self.patient_info[patient_id].keys():
                path_previous = self.patient_info[patient_id][no_slice -1]
            else:
                path_previous = path

            if no_slice +1 in self.patient_info[patient_id].keys():
                path_next = self.patient_info[patient_id][no_slice +1]
            else:
                path_next = path
            img_previous = cv.imread(path_previous, 0)
            img_next = cv.imread(path_next, 0)
            img0 = np.dstack((img_previous, img0, img_next))
        
        img = letterbox(img0, new_shape=self.img_size, auto_size=self.auto_size)[0]

        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, None

def save_xml(path, filename, image_size, preds, path_save, database_src = 'LUNA'):
    tree = et.Element("result")

    filename_et = et.SubElement(tree, "filename")
    filename_et.text = os.path.basename(path)

    source = et.SubElement(tree, 'source')
    database = et.SubElement(source, 'database')
    database.text = database_src

    size_part = et.SubElement(tree, 'size')
    width = et.SubElement(size_part, 'width')
    height = et.SubElement(size_part, 'height')
    depth = et.SubElement(size_part, 'depth')
    width.text = str(image_size[1])
    height.text = str(image_size[0])
    if len(image_size) == 3:
        depth.text = str(image_size[2])
    else:
        depth.text = '1'

    segmented = et.SubElement(tree, 'segmented')
    segmented.text = '0'

    for xmin, ymin, xmax, ymax, conf, cls in preds:
        object_item = et.SubElement(tree, 'object')
        name = et.SubElement(object_item, 'name')
        name.text = str(cls)
        pose = et.SubElement(object_item, 'pose')
        pose.text = "Unspecified"
        
        confident = et.SubElement(object_item, "confident")
        confident.text = "{:.4f}".format(conf)

        bnd_box = et.SubElement(object_item, 'bndbox')
        x_min = et.SubElement(bnd_box, 'xmin')
        x_min.text = str(xmin)
        y_min = et.SubElement(bnd_box, 'ymin')
        y_min.text = str(ymin)
        x_max = et.SubElement(bnd_box, 'xmax')
        x_max.text = str(xmax)
        y_max = et.SubElement(bnd_box, 'ymax')
        y_max.text = str(ymax)

    out_file = None
    out_file = codecs.open(os.path.join(path_save, filename + ".xml"), 'w', encoding='utf-8')
    def prettify(elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = et.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding='utf-8').replace("  ".encode(), "\t".encode())

    prettify_result = prettify(tree)
    out_file.write(prettify_result.decode('utf8'))
    out_file.close()

def save_result_txt(path_save:str, dict_patient:dict, dict_file:dict):
    path_save = str(Path(path_save))
    for patient, infor in dict_patient.items():
        string = ""
        for no_slice, bb in infor.items():
            filename = dict_file[patient][no_slice]
            string += filename
            for *xyxy, conf, cls in bb:
                string += " %g,%g,%g,%g,%g,%g" %(*xyxy, cls, conf)
            string +="\n"
        with open(os.path.join(path_save, "patient_{:04d}.txt".format(patient)), 'w+') as f:
            f.writelines(string)
            f.close()
    
XML_PATH = r"weights\FP16\best.xml"
BIN_PATH = r"weights\FP16\best.bin"

def load_to_IE(xml_path = XML_PATH, bin_path = BIN_PATH):
    xml_path = os.path.abspath(xml_path)
    bin_path = os.path.abspath(bin_path)
    if not (os.path.isfile(xml_path) and os.path.isfile(bin_path)):
        raise Exception("Error %s/ %s do not exist" % (xml_path, bin_path))
    ie = IECore()
    # Loading IR files
    net = IENetwork(model = xml_path, weights= bin_path)

    input_shape = net.inputs['input'].shape
    net_outputs = list(net.outputs.keys())
    # Loading the network to the inference engine
    exec_net = ie.load_network(network= net, device_name="CPU")
    # exec_net = ie.load_network(network= net, device_name="GPU")

    return exec_net, input_shape, net_outputs

def do_inference(exec_net, image):
    input_blob = next(iter(exec_net.inputs))
    return exec_net.infer({input_blob: image})

def detect(save_img=False):
    output, source, save_txt = opt.output, opt.source, opt.save_txt

    conf_thresh, iou_thresh = opt.conf_thres, opt.iou_thres

    net, input_shape, net_outputs = load_to_IE()
    image_size = tuple(input_shape[-2:])

    # Set Dataloader
    
    save_img = True
    dataset = LoadFolder(source, image_size, auto_size= 64)

    # Run inference
    t0 = time.time()
    img = np.zeros(input_shape, dtype=np.float32)
    _ = do_inference(net, img)

    string_results = ""
    bbox_results = {}

    for path, image, im0s, vid_cap in dataset:
        img = cv.dnn.blobFromImage(image, 1.0/255.0, image_size, (1,1,1), True)
        pred = do_inference(net, img)
        key = net_outputs[-1]
        pred = pred[key]
        pred = torch.from_numpy(pred)
    
        preds = non_max_suppression(pred, conf_thresh, iou_thresh, classes=0, agnostic=False)
        preds = preds[0].cpu().detach().numpy()
        
        if preds is not None and len(preds):
            filename = get_filename(path)
            bbox_results[filename] = preds
    dict_patient, dict_file = olap.merge_overlapping(bbox_results)
    if True:
        save_result_txt(output, dict_patient, dict_file)

    if True:
        for patient, infor in dict_patient.items():
            for no_slice, preds in infor.items():
                filename = dict_file[patient][no_slice]
                save_xml(filename, filename, (512,512,3), preds, str(Path(output)))


    


    #         string_results += path
    #         txt_result = ""
    #         for *xyxy, conf, cls in preds:
    #             string_results += " %g,%g,%g,%g,%g,%g" %(*xyxy, cls, conf)
    #             txt_result += " %g,%g,%g,%g,%g,%g\n" %(*xyxy, cls, conf)
    #         string_results += "\n"
    #         filename = get_filename(path)
    #         with open(os.path.join(str(Path(output)), filename + ".txt"), "w+") as f:
    #             f.write(txt_result)
    #             f.close()

    #         # if opt.save_xml:
    #         if True:
    #             save_xml(str(Path(path)), filename, image.shape, preds, str(Path(output)))
    
    # # if opt.save_txt:
    # if True:
    #     # print(string_results)
    #     with open(os.path.join(str(Path(output)), "inference.txt"), "w+") as f:
    #         f.write(string_results)
    #         f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=r"D:\02_BME\000_LUNA16\004_Gray24bit_RGB_All\test", help= "Source")
    parser.add_argument('--output', type=str, default=r"D:\02_BME\000_LUNA16\004_Gray24bit_RGB_All", help= "Output")
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-xml', action='store_true', help='save results to *.xml')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    detect()

