import os
import math
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import cv2 as cv
from pathlib import Path
import time
try:
    import inference_openvino_FP_preprocess_parallel as infer
except:
    import libraries.inference.inference_openvino_FP_preprocess_parallel as infer

class inferenceThread(QObject):
    processCount = pyqtSignal(int)
    duration = pyqtSignal(float)
    stringOutput = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, directory:str, parent=None) -> None:
        super().__init__(parent)
        self.dir = directory
        self.src_input = os.path.join(os.path.abspath(directory), "image.pickle")
        self.num_requests = 4
        self.initial_model()
        self.output = directory
        
    def initial_model(self):
        if not os.path.exists(self.src_input):
            return False
        self.net_1, self.input_shape, self.net_outputs_1 = infer.load_to_IE(self.num_requests)
        self.image_size = tuple(self.input_shape[-2:])

        # FP Reduction
        self.net_fp, self.net_outputs_fp = infer.load_to_IE_FP(self.num_requests)

        # Set Dataloader
        if self.src_input.endswith('.pickle'):
            self.dataset = infer.LoadPickle(self.src_input, img_size=self.image_size, autosize=64)
        else:
            save_img = True
            self.dataset = infer.LoadImages(self.src_input, self.image_size, auto_size= 64)

    @pyqtSlot()
    def run(self):
        # img = np.zeros(self.input_shape, dtype=np.float32)
        # _ = infer.do_inference(self.net_1, img)
        print("Detecting Nodule")
        
        net_input_blob_1 = next(iter(self.net_1.inputs))

        # preprocess input data
        images_preprocess = []
        for path, image, _, _ in self.dataset:
            img = cv.dnn.blobFromImage(image, 1.0/255.0, self.image_size, (1,1,1), True)
            images_preprocess.append([path, img])
        
        # prerequisites for OpenVINO asynchronous inference
        request_id = -1 # index for current inference request
        request_info_buffer = ['' for i in range(self.num_requests)] # buffer for storing request information

        # record results from stage 1
        string_results = ""
        results = []

        # record start time
        t1 = time.time()

        detect_length = len(self.dataset)
        detect_count = 0
        for path, img in images_preprocess:
            request_id = (request_id+1) % self.num_requests # current inference request

            # if the current inference request has been occupied
            if request_info_buffer[request_id] != '':
                # wait until the request is finished and get the inference results
                self.net_1.requests[request_id].wait()
                pred = self.net_1.requests[request_id].output_blobs[self.net_outputs_1[-1]].buffer
                pred_path = request_info_buffer[request_id]

                # postprocess inference results
                # preds = non_max_suppression(pred, thresh_conf, thresh_iou)
                preds = infer.non_max_suppression(pred, 0.1, 0.1)
                preds = preds[0]
                if preds is not None and len(preds):
                    string_results += pred_path
                    for *xyxy, conf, cls in preds:
                        string_results += " %g,%g,%g,%g,%g,%g" %(*xyxy, cls, conf)
                        results.append([pred_path, *xyxy, cls, conf])
                    string_results += "\n"
                
                # clean the buffer
                request_info_buffer[request_id] = ''

                detect_count += 1
                self.processCount.emit(int((detect_count/detect_length) * 80))
            
            # send an inference request
            self.net_1.requests[request_id].async_infer({net_input_blob_1: img})
            request_info_buffer[request_id] = path
        
        # retrieve the remaining inference results
        for request_id in range(self.num_requests):
            # if the inference request hasn't been cleaned
            if request_info_buffer[request_id] != '':
                # wait until the request is finished and get the inference results
                self.net_1.requests[request_id].wait()
                pred = self.net_1.requests[request_id].output_blobs[self.net_outputs_1[-1]].buffer
                pred_path = request_info_buffer[request_id]

                # postprocess inference results
                # preds = non_max_suppression(pred, thresh_conf, thresh_iou)
                preds = infer.non_max_suppression(pred, 0.1, 0.1)
                preds = preds[0]
                if preds is not None and len(preds):
                    string_results += pred_path
                    for *xyxy, conf, cls in preds:
                        string_results += " %g,%g,%g,%g,%g,%g" %(*xyxy, cls, conf)
                        results.append([pred_path, *xyxy, cls, conf])
                    string_results += "\n"
                
                # clean the buffer
                request_info_buffer[request_id] = ''

                detect_count += 1
                self.processCount.emit(int((detect_count/detect_length) * 80))

        print(string_results)
        with open(os.path.join(str(Path(self.output)), "inference.txt"), "w+") as f:
            f.write(string_results)
            f.close()
        
        print("Reducing FP")
        fp_reduction_preprocessor = infer.fp_reduction_preprocess()
        # lung_segmenter = infer.Lung_segmentation(detect_length)

        # lung_segmenter.lung_segment(self.dataset,math.floor(detect_length/2),0,-1,True) # segment lung region from the middle to the top
        # lung_segmenter.lung_segment(self.dataset,math.floor(detect_length/2),detect_length,1,False) # segment lung region from the middle to the bottom

        # record results from stage 2
        fp_results_filename = []
        fp_results_bbox = []
        fp_result_bbox_count = 0

        # group results
        result_groups = fp_reduction_preprocessor.collect_3D_nodule(results)

        # preprocess input data
        images_preprocess = []
        result_group_length = []
        # result_coverage = []
        for group in result_groups:
            temp_images = []
            # temp_coverages = []
            for bbox in group:
                large, medium, small = fp_reduction_preprocessor.get_patch(self.dataset, bbox)
                temp_images.append([large, medium, small])

                # print(bbox[0])
                # mask_index = self.dataset.index_of_image[bbox[0]]
                # coverage = lung_segmenter.calculate_lung_mask_coverage(mask_index, bbox[1:5])
                # print(coverage)
                # temp_coverages.append(coverage)
            
            images_preprocess.append(temp_images)
            result_group_length.append(len(group))
            # result_coverage.append(temp_coverages)
        
        # prerequisites for OpenVINO asynchronous inference
        request_info_buffer = [[] for i in range(self.num_requests)]

        fp_length = len(result_groups)
        fp_count = 0
        for i in range(len(result_groups)):
            # adjust the probability threshold for each group
            threshold = fp_reduction_preprocessor.calculate_probability_threshold(result_group_length[i])

            for j in range(result_group_length[i]):
                request_id = (request_id+1) % self.num_requests # current inference request

                # if the current inference request has been occupied
                if len(request_info_buffer[request_id]) != 0:
                    # wait until the request is finished and get the inference results
                    self.net_fp.requests[request_id].wait()
                    prediction = self.net_fp.requests[request_id].output_blobs[self.net_outputs_fp[-1]].buffer[0][0]
                    prediction_bbox_threshold_coverage = request_info_buffer[request_id]

                    # postprocess inference results
                    probability = 0.3*float(prediction_bbox_threshold_coverage[6]) + 0.7*prediction

                    # DEBUG
                    # print(prediction_bbox_threshold_coverage[0:7] + [round(prediction, 6), round(probability, 6)])
                
                    if probability >= prediction_bbox_threshold_coverage[7]:
                        # check if the bbox is in the lung region

                        # if prediction_bbox_threshold_coverage[8] >= 0.1:
                        # record positive bboxes
                        if prediction_bbox_threshold_coverage[0] not in fp_results_filename:
                            fp_results_filename.append(prediction_bbox_threshold_coverage[0])
                            fp_results_bbox.append([])
                        
                        fp_result_index = fp_results_filename.index(prediction_bbox_threshold_coverage[0])
                        fp_results_bbox[fp_result_index].append("{:.3f},{:.3f},{:.3f},{:.3f},{},{:.6f}".format(*prediction_bbox_threshold_coverage[1:7]))
                        
                        fp_result_bbox_count += 1
                    
                    # clean the buffer
                    request_info_buffer[request_id] = []
                
                # send an inference request
                self.net_fp.requests[request_id].async_infer({'input_1': images_preprocess[i][j][0], 'input_2': images_preprocess[i][j][1], 'input_3': images_preprocess[i][j][2]})
                # request_info_buffer[request_id] = result_groups[i][j] + [threshold, result_coverage[i][j]]
                request_info_buffer[request_id] = result_groups[i][j] + [threshold]
            
            fp_count += 1
            self.processCount.emit(int(80 + (fp_count/fp_length) *20))
        
        # retrieve the remaining inference results
        for request_id in range(self.num_requests):
            # if the inference request hasn't been cleaned
            if len(request_info_buffer[request_id]) != 0:
                # wait until the request is finished and get the inference results
                self.net_fp.requests[request_id].wait()
                prediction = self.net_fp.requests[request_id].output_blobs[self.net_outputs_fp[-1]].buffer[0][0]
                prediction_bbox_threshold_coverage = request_info_buffer[request_id]
                
                # postprocess inference results
                probability = 0.3*float(prediction_bbox_threshold_coverage[6]) + 0.7*prediction

                # DEBUG
                # print(prediction_bbox_threshold_coverage[0:7] + [round(prediction, 6), round(probability, 6)])
            
                if probability >= prediction_bbox_threshold_coverage[7]:
                    # check if the bbox is in the lung region

                    # if prediction_bbox_threshold_coverage[8] >= 0.1:
                    # record positive bboxes
                    if prediction_bbox_threshold_coverage[0] not in fp_results_filename:
                        fp_results_filename.append(prediction_bbox_threshold_coverage[0])
                        fp_results_bbox.append([])
                    
                    fp_result_index = fp_results_filename.index(prediction_bbox_threshold_coverage[0])
                    fp_results_bbox[fp_result_index].append("{:.3f},{:.3f},{:.3f},{:.3f},{},{:.6f}".format(*prediction_bbox_threshold_coverage[1:7]))
                    
                    fp_result_bbox_count += 1
                
                # clean the buffer
                request_info_buffer[request_id] = []

        if fp_result_bbox_count <= 1:
            fp_results_filename = []
            fp_results_bbox = []
            fp_result_bbox_count = 0

            # for i in range(len(result_groups)):
            #     for j in range(result_group_length[i]):
            #         if result_coverage[i][j] >= 0.2:
            #             if result_groups[i][j][0] not in fp_results_filename:
            #                 fp_results_filename.append(result_groups[i][j][0])
            #                 fp_results_bbox.append([])
                        
            #             fp_result_index = fp_results_filename.index(result_groups[i][j][0])
            #             fp_results_bbox[fp_result_index].append("{:.3f},{:.3f},{:.3f},{:.3f},{},{:.6f}".format(*result_groups[i][j][1:7]))

            lung_segmenter = infer.Lung_segmentation(detect_length)

            lung_segmenter.lung_segment(self.dataset,math.floor(detect_length/2),0,-1,True) # segment lung region from the middle to the top
            lung_segmenter.lung_segment(self.dataset,math.floor(detect_length/2),detect_length,1,False) # segment lung region from the middle to the bottom

            for bbox in results:
                mask_index = self.dataset.index_of_image[bbox[0]]
                coverage = lung_segmenter.calculate_lung_mask_coverage(mask_index, bbox[1:5])

                if coverage > 0.2:
                    if bbox[0] not in fp_results_filename:
                        fp_results_filename.append(bbox[0])
                        fp_results_bbox.append([])
                    
                    fp_result_index = fp_results_filename.index(bbox[0])
                    fp_results_bbox[fp_result_index].append("{:.3f},{:.3f},{:.3f},{:.3f},{},{:.6f}".format(*bbox[1:7]))

        string_results = ''
        for i in range(len(fp_results_filename)):
            string_results += (fp_results_filename[i] + ' ')

            for bbox in fp_results_bbox[i]:
                string_results += (bbox + ' ')
            
            string_results += '\n'
        
        print(string_results)
        with open(os.path.join(str(Path(self.output)), "inference_FP_reduction.txt"), "w+") as f:
            f.write(string_results)
            f.close()

        t2 = time.time()
        duration = t2 -t1
        self.duration.emit(duration)
        self.stringOutput.emit(os.path.join(str(Path(self.output)), "inference_FP_reduction.txt"))
        self.finished.emit()