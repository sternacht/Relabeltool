import sys
import os
import numpy as np
import pandas as pd
from sklearn.utils.linear_assignment_ import linear_assignment

class ObjectDetectMetric():
    def __init__(self,
                names_class:list, 
                check_class_first=True):
        """
        Intial parameter to analysis
        Input:
            names_class: a label list. for example: ['nodule']
            check_class_first: a boolean. to calculate epsilon with first class
        """
        self.names_class = names_class
        self.check_class_first = check_class_first
        self.number_classes = len(names_class)
        self.number_groundtruth_all = [0] * self.number_classes
        self.number_prediction_all = [0] * self.number_classes
        self.infos_all = [ [] for _ in range(self.number_classes + 1)]
        self._is_sorted = False
        self.table_dict = {'filename': [],
                            'score': [],
                            'prediction': [],
                            'ground truth': []}

    @staticmethod
    def _get_IOUs(bboxes1_xyxy,bboxes2_xyxy):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            boxes1 (Array[N, 4])
            boxes2 (Array[M, 4])
        Returns:
            iou (Array[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        This implementation is taken from the above link and changed so that it only uses numpy..
        """
        if bboxes1_xyxy.shape[0] == 0:
            return np.zeros((len(bboxes1_xyxy),len(bboxes2_xyxy)))
        if bboxes2_xyxy.shape[0] == 0:
            return np.zeros((len(bboxes1_xyxy),len(bboxes2_xyxy)))
        bboxes1 = (bboxes1_xyxy).reshape((-1,1,4))
        bboxes2 = (bboxes2_xyxy).reshape((1,-1,4))
        x_overlap = np.minimum(bboxes1[:,:,2],bboxes2[:,:,2]) - np.maximum(bboxes1[:,:,0],bboxes2[:,:,0]) # right-left
        y_overlap = np.minimum(bboxes1[:,:,3],bboxes2[:,:,3]) - np.maximum(bboxes1[:,:,1],bboxes2[:,:,1]) # bottom-top
        area_i = np.where( (x_overlap>0) & (y_overlap>0) , x_overlap*y_overlap, 0.)
        area1 = ( bboxes1[:,:,2]-bboxes1[:,:,0] )*( bboxes1[:,:,3]-bboxes1[:,:,1] )
        area2 = ( bboxes2[:,:,2]-bboxes2[:,:,0] )*( bboxes2[:,:,3]-bboxes2[:,:,1] )
        return area_i/(area1+area2-area_i)

    @staticmethod
    def _get_IOGTs(bboxes1_xyxy,bboxes2_xyxy):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            boxes1 (Array[N, 4])
            boxes2 (Array[M, 4])
        Returns:
            iou (Array[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        This implementation is taken from the above link and changed so that it only uses numpy..
        """
        if bboxes1_xyxy.shape[0] == 0:
            return np.zeros((len(bboxes1_xyxy),len(bboxes2_xyxy)))
        if bboxes2_xyxy.shape[0] == 0:
            return np.zeros((len(bboxes1_xyxy),len(bboxes2_xyxy)))
        bboxes1 = (bboxes1_xyxy).reshape((-1,1,4))
        bboxes2 = (bboxes2_xyxy).reshape((1,-1,4))
        x_overlap = np.minimum(bboxes1[:,:,2],bboxes2[:,:,2]) - np.maximum(bboxes1[:,:,0],bboxes2[:,:,0]) # right-left
        y_overlap = np.minimum(bboxes1[:,:,3],bboxes2[:,:,3]) - np.maximum(bboxes1[:,:,1],bboxes2[:,:,1]) # bottom-top
        area_i = np.where( (x_overlap>0) & (y_overlap>0) , x_overlap*y_overlap, 0.)
        area1 = ( bboxes1[:,:,2]-bboxes1[:,:,0] )*( bboxes1[:,:,3]-bboxes1[:,:,1] )
        return area_i/(area1)
    
    @staticmethod
    def _get_IOPs(bboxes1_xyxy,bboxes2_xyxy):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            boxes1 (Array[N, 4])
            boxes2 (Array[M, 4])
        Returns:
            iou (Array[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        This implementation is taken from the above link and changed so that it only uses numpy..
        """
        if bboxes1_xyxy.shape[0] == 0:
            return np.zeros((len(bboxes1_xyxy),len(bboxes2_xyxy)))
        if bboxes2_xyxy.shape[0] == 0:
            return np.zeros((len(bboxes1_xyxy),len(bboxes2_xyxy)))
        bboxes1 = (bboxes1_xyxy).reshape((-1,1,4))
        bboxes2 = (bboxes2_xyxy).reshape((1,-1,4))
        x_overlap = np.minimum(bboxes1[:,:,2],bboxes2[:,:,2]) - np.maximum(bboxes1[:,:,0],bboxes2[:,:,0]) # right-left
        y_overlap = np.minimum(bboxes1[:,:,3],bboxes2[:,:,3]) - np.maximum(bboxes1[:,:,1],bboxes2[:,:,1]) # bottom-top
        area_i = np.where( (x_overlap>0) & (y_overlap>0) , x_overlap*y_overlap, 0.)
        area2 = ( bboxes2[:,:,2]-bboxes2[:,:,0] )*( bboxes2[:,:,3]-bboxes2[:,:,1] )
        return area_i/(area2)

    def appendTable(self, prediction, ground_truth, score, filename):
        self.table_dict['filename'].append(filename)
        self.table_dict['prediction'].append(prediction)
        self.table_dict['ground truth'].append(ground_truth)
        self.table_dict['score'].append(score)
        
    def _update(self, ground_truth, labels_ground_truth, prediction, labels_prediction, filename):
        """
        Find and match bounding box to ground truth.
        Update detected nodule and non-detected nodule.  
        Arguments:
            ground_truth (Array[N, 4])
            labels_ground_truth (Array[N,1])
            prediction (Array[M, 5])
            labels_prediction (Array[N,1])
        """
        self._is_sorted = False
        labels_groundtruth = np.array(labels_ground_truth, dtype=np.uint16)
        bboxes_groundtruth = np.array(ground_truth, dtype=np.float32)
        if prediction.shape[0] != 0:
            labels_prediction = np.array(labels_prediction[:], dtype=np.uint16)
            bboxes_prediction = np.array(prediction[:, :4], dtype=np.float32)
            scores_prediction = np.array(prediction[:, -1], dtype=np.float32)
        else:
            labels_prediction = np.array(prediction, dtype=np.uint16)
            bboxes_prediction = np.array(prediction, dtype=np.float32)
            scores_prediction = np.array(prediction, dtype=np.float32)

        number_groundtruth = len(bboxes_groundtruth)
        number_prediction  = len(bboxes_prediction)
        for label in labels_groundtruth:
            self.number_groundtruth_all[label] += 1
        for label in labels_prediction:
            self.number_prediction_all[label] += 1

        matrix_IOU = self._get_IOUs(bboxes_groundtruth,bboxes_prediction)
        matrix_IOP = self._get_IOPs(bboxes_groundtruth,bboxes_prediction)
        matrix_IOGT = self._get_IOGTs(bboxes_groundtruth, bboxes_prediction)
        same = labels_groundtruth.reshape((-1,1))==labels_prediction.reshape((1,-1))
        if self.check_class_first:
            matrix_IOU *= same
            matrix_IOP *= same
            matrix_IOGT *= same
        else:
            matrix_IOU[same] *= (1+1e-06)
            matrix_IOP[same] *= (1+1e-06)
            matrix_IOGT[same] *= (1+1e-06)

        # if matrix_IOU.size != 0:
        #     ind = np.argwhere(matrix_IOU==np.amax(matrix_IOU,1, keepdims=True))
        #     self.matched = np.array(list(map(list, ind)))
        # else:
        #     self.matched = np.empty((0,2), dtype=np.int32)

        self.matched = linear_assignment(-matrix_IOU)

        self.unmatched_groundtruth = list(set(range(number_groundtruth))-set(self.matched[:,0]))
        self.unmatched_prediction  = list(set(range(number_prediction ))-set(self.matched[:,1]))
        for n,(i,j) in reversed(list(enumerate(self.matched))):
            if matrix_IOU[i,j] == 0:
                self.unmatched_groundtruth.append(i)
                self.unmatched_prediction.append(j)
                self.matched = np.delete(self.matched,n,0)
            else:
                self.infos_all[ labels_prediction[j] ].append([labels_groundtruth[i],
                                                               scores_prediction[j],
                                                               matrix_IOU[i,j], 
                                                               matrix_IOP[i,j], 
                                                               matrix_IOGT[i,j]])
                self.appendTable(bboxes_prediction[j], 
                                bboxes_groundtruth[i], 
                                scores_prediction[j], 
                                filename= filename)

        for i in self.unmatched_groundtruth:
            self.infos_all[ -1 ].append( [labels_groundtruth[i],0,0,0,0] )
            self.appendTable(None, 
                            bboxes_groundtruth[i], 
                            None, 
                            filename)

        for j in self.unmatched_prediction:
            self.infos_all[ labels_prediction[j] ].append( [-1,scores_prediction[j],0, 0, 0])
            self.appendTable(bboxes_prediction[j], 
                            None, 
                            scores_prediction[j], 
                            filename)

    def _sort(self):
        self.IOUs_all   = [ [] for _ in range(self.number_classes) ]
        self.scores_all = [ [] for _ in range(self.number_classes) ]
        for no_class in range(self.number_classes):
            infos = np.array( self.infos_all[no_class] ).reshape((-1,5))
            matched = ( infos[:,0] == no_class )
            unmatched = np.logical_not(matched)
            self.scores_all[no_class]  = list(infos[matched,1])
            self.IOUs_all[  no_class]  = list(infos[matched,2])
            self.scores_all[no_class] += list(infos[unmatched,1])
            self.IOUs_all[  no_class] += [0]*sum(unmatched)
            if len(self.IOUs_all[no_class]) != 0:
                self.IOUs_all[  no_class],\
                self.scores_all[no_class] = zip(*sorted(zip(self.IOUs_all[no_class],
                                                            self.scores_all[no_class]),
                                                        key=lambda x:x[1]+x[0]*1e-10, reverse=True))
                #for i in range(len(self.IOUs_all[no_class])):
                #    print("%7.5f    %7.5f"%(self.IOUs_all[no_class][i],self.scores_all[no_class][i]))
                #print("--")
        self._is_sorted = True

    def get_confusion(self, threshold_confidence, 
                    threshold_IOU,
                    threshold_IOP,
                    threshold_IOGT, 
                    conclude=False):
        
        eps = 1e-50
        IOUs_avg = []
        matrix_confusion = np.zeros((self.number_classes+1,self.number_classes+1))
        for no_class_prediction in range(self.number_classes):
            infos = np.array(self.infos_all[no_class_prediction])
            infos = infos.reshape((-1,5))
            above_IOU = ( infos[:,2]>=threshold_IOU )
            above_IOP = ( infos[:,3]>=threshold_IOP)
            above_IOGT = (infos[:,4]>=threshold_IOGT)
            above_total = []
            for iou, iop, iogt in zip(above_IOU, above_IOP, above_IOGT):
                if iou:
                    above_total.append(iou)
                else:
                    if iop:
                        above_total.append(iop)
                    else:
                        above_total.append(iogt)
            above_IOU = np.array(above_total)
            same = ( infos[:,0]==no_class_prediction )
            IOUs_avg.append( np.sum(infos[:,2]*same)/(np.sum(same)+eps) ) # if matched
            below_IOU = np.logical_not(above_IOU)

            matrix_confusion[-1,no_class_prediction] += np.sum(below_IOU)
            for no_class_groundtruth in range(self.number_classes):
                matched_class = ( infos[:,0] == no_class_groundtruth )
                matrix_confusion[no_class_groundtruth,no_class_prediction] += np.sum(matched_class & above_IOU)
                matrix_confusion[no_class_groundtruth,-1]                  += np.sum(matched_class & below_IOU)

        infos = np.array(self.infos_all[-1])
        if len(infos):
            for no_class_groundtruth in range(self.number_classes):
                matrix_confusion[no_class_groundtruth,-1] += np.sum( infos[:,0]==no_class_groundtruth )

        # print the results
        fields = self.names_class+["none"]
        length_name = max([len(str(s)) for s in fields]+[5])
        spacing = "- "*max((int(7+((length_name+3)*(self.number_classes+3))/2)),
                           length_name+33)
        content = ""
        content += spacing+"\nConfusion Matrix\n"+spacing+"\n"
        content += ("thresh_confidence: %f"%threshold_confidence).rstrip("0")+"\n"
        content += ("thresh_IOU       : %f"%threshold_IOU).rstrip("0")+"\n"
        content += ("thresh_IOP       : %f"%threshold_IOP).rstrip("0")+"\n"
        content += ("thresh_IOGT      : %f"%threshold_IOGT).rstrip("0")+"\n"
        matrix_confusion = np.uint32(matrix_confusion)
        content2 = " "*(length_name+3+12)
        for j in range(self.number_classes+1):
            content2 += "[%*s] "%(length_name,fields[j])
        content2 += "[%*s] \n"%(length_name,"total")
        content2 += "%*sPrediction\n"%(int(12+(len(content2)-10)/2),"")
        content += content2
        content3 = ""
        for i in range(self.number_classes+1):
            content3 = "Groundtruth " if i==int((self.number_classes+1)/2) else " "*12
            content3 += "[%*s] "%(length_name,fields[i])
            for j in range(self.number_classes+1):
                if i==j==self.number_classes:
                    break
                content3 += "%*d "%(length_name+2,matrix_confusion[i,j])
            if i < self.number_classes:
                content3 += "%*d "%(length_name+2,self.number_groundtruth_all[i])
            content += content3+"\n"
        content += " "*12+"[%*s] "%(length_name,"total")
        for j in range(self.number_classes):
            content += "%*d "%(length_name+2,self.number_prediction_all[j])
            #content += "%*d "%(length_name+2,sum(matrix_confusion[:,j]))
        content += "\n"+spacing+"\n"
        self.table_Analysis = []
        for no_class,name in enumerate(self.names_class):
            precision = matrix_confusion[no_class,no_class]/(self.number_prediction_all[ no_class]+eps)
            recall    = matrix_confusion[no_class,no_class]/(self.number_groundtruth_all[no_class]+eps)
            f1_score = (2 * recall *100 * precision)/(recall+ precision)
            content += ("[%*s]   recall: %6.2f %%"
                              "     precision: %6.2f %%"
                              "     f1-score: %6.2f %%"
                             "     avg IOU: %6.2f %%\n")%(\
                    length_name,name,
                    1e2*recall,
                    1e2*precision,
                    f1_score,
                    1e2*IOUs_avg[no_class])
            infor = [name, "{}/{} = {:6.2f} %".format(matrix_confusion[no_class, no_class], 
                                                (self.number_groundtruth_all[ no_class]+eps), 
                                                1e2*recall),
                    "{}/{} = {:6.2f} %".format(matrix_confusion[no_class, no_class], 
                                                (self.number_prediction_all[ no_class]+eps), 
                                                1e2*precision),
                    "{:6.2f}".format(f1_score)]
            self.table_Analysis.append(infor)
        content += spacing
        if conclude:
            print(content)
        else:
            return content
        
    def get_table(self):
        "Return dictionary of prediction and ground truth"
        return pd.DataFrame(self.table_dict)

    def get_table_recall_precision(self):
        return pd.DataFrame(self.table_Analysis, columns=['labels', 'recall (%)', 'precision (%)', 'f1-score'])
