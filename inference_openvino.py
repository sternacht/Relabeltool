import argparse
import os
import platform
import shutil
import cv2 as cv
import numpy as np
from pathlib import Path
import pickle
import time
import glob
from utils.general import non_max_suppression
from openvino.inference_engine import IECore, IENetwork

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
    pickData = source.endswith('.pickle')

    net, input_shape, net_outputs = load_to_IE()
    image_size = tuple(input_shape[-2:])

    # Set Dataloader
    if pickData:
        dataset = LoadPickle(source, img_size=image_size, autosize=64)
    else:
        save_img = True
        dataset = LoadImages(source, image_size, auto_size= 64)

    # Run inference
    t0 = time.time()
    img = np.zeros(input_shape, dtype=np.float32)
    _ = do_inference(net, img)

    string_results = ""

    for path, image, im0s, vid_cap in dataset:
        img = cv.dnn.blobFromImage(image, 1.0/255.0, image_size, (1,1,1), True)
        pred = do_inference(net, img)
        key = net_outputs[-1]
        pred = pred[key]
        # pred = torch.from_numpy(pred)
    
        # preds = non_max_suppression(pred, 0.1, 0.1, classes=0, agnostic=False)
        # preds = preds[0].cpu().detach().numpy()
        preds = non_max_suppression(pred, 0.4, 0.1)
        preds = preds[0]
        if preds is not None and len(preds):
            string_results += path
            for *xyxy, conf, cls in preds:
                string_results += " %g,%g,%g,%g,%g,%g" %(*xyxy, cls, conf)
            string_results += "\n"
    
    if opt.save_txt:
        print(string_results)
        with open(os.path.join(str(Path(output)), "inference.txt"), "w+") as f:
            f.write(string_results)


        # box = pred[:4]
        # conf =pred[4]
        # if any(i > image_size[0] for i in box):
        #     continue
        # if save_img:
        #     image = cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=r"D:\Project\PyTorch_YOLOv4_2\image.pickle", help= "Source")
    parser.add_argument('--output', type=str, default=r"D:\Project\PyTorch_YOLOv4_2", help= "Output")
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    detect()

