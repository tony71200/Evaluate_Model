import sys
import os
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from libraries import utils
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

now = datetime.now()

class BME_FROC():

    def __init__(self, dir:str, 
                folder_GroundTruth:str, 
                folder_ModelName:str, 
                name_class:list, 
                check_class_first = True):
        self.dir = dir
        self.folder_GroundTruth = folder_GroundTruth
        self.folder_ModelName = folder_ModelName
        self.name_class = name_class
        self.check_class_first = check_class_first

        self.number_classes = len(name_class)

        # self.number_groundtruth_all = [0] * self.number_classes
        # self.number_prediction_all = [0] * self.number_classes
        # self.infos_all = [ [] for _ in range(self.number_classes + 1)]
        self._is_sorted = False

        path_gt = os.path.join(self.dir, self.folder_GroundTruth, "GroundTruth.csv")
        self.ground_truth = self._load_groundtruth(path_gt)
        path_pred = utils.select_file(os.path.join(self.dir, self.folder_ModelName))
        self.prediction = self._load_prediction(path_pred)
        self.save_result = os.path.join(dir, folder_ModelName, "003_Result")

        self.number_of_threshold = 100
        self.range_threshold = [0.01, 1.0]


    def clear(self):
        self.number_groundtruth_all = [0] * self.number_classes
        self.number_prediction_all = [0] * self.number_classes
        self.infos_all = [ [] for _ in range(self.number_classes + 1)]
        self._is_sorted = False

    @staticmethod
    def _load_groundtruth(path_groundTruth):
        if os.path.exists(path_groundTruth):
            dict_gt = {}
            gt_df = utils.readcsv(path_groundTruth)
            for _, row in gt_df.iterrows():
                filename = row['filename']
                bb = [0, row['x1'], row['y1'], row['x2'], row['y2']]
                if not filename in dict_gt.keys():
                    dict_gt[filename] = [bb]
                else:
                    dict_gt[filename].append(bb)
            return dict_gt
        else:
            print("Not find GroundTruth.csv")
            sys.exit()

    @staticmethod
    def _load_prediction(path_prediction):
        if os.path.exists(path_prediction):
            dict_pred = {}
            gt_df = utils.readjson(path_prediction)
            for _, row in gt_df.iterrows():
                if len(row['objects']) == 0:
                    continue 
                filename = utils.get_filename(row['filename'])
                for obj in row['objects']:
                    label = obj['class_id']
                    x_center = obj['relative_coordinates']['center_x']
                    y_center = obj['relative_coordinates']['center_y']
                    w_yolo = obj['relative_coordinates']['width']
                    h_yolo = obj['relative_coordinates']['height']
                    score = obj['confidence']
                    x1, y1, x2, y2 = utils.yolo2xy((x_center, y_center, w_yolo, h_yolo))
                    bb = [label, x1, y1, x2, y2, score]
                    if not filename in dict_pred.keys():
                        dict_pred[filename] = [bb]
                    else:
                        dict_pred[filename].append(bb)
            return dict_pred
        else:
            print("Not find {}".format(utils.get_basename(path_prediction)))
            sys.exit()

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

    def _processNMS(self, thresh_confident, thresh_iou = 0.5):
        prediction = self.prediction.copy()
        for key, value in self.prediction.items():
            array_pred = np.array(value)
            array_pred = array_pred[array_pred[:, 5] > thresh_confident]
            array_pred_1 = array_pred[:,1:5].copy()
            array_compare = array_pred_1.copy()
            all_IOU = self._get_IOUs(array_pred_1, array_compare)
            want_idx = np.where(all_IOU > thresh_iou)
            delete_value = []
            for idx_pred, idx_compare in zip(want_idx[0], want_idx[1]):
                if idx_pred != idx_compare:
                    if array_pred[idx_pred, 5] >= array_pred[idx_compare, 5]:
                        delete_value.append(idx_compare)
                    else:
                        delete_value.append(idx_pred)
            prediction[key] = array_pred.tolist()
            if len(delete_value) != 0:
                somelist = [i for j, i in enumerate(prediction[key]) if j not in np.unique(delete_value)]
                prediction[key] = somelist
        return prediction

    def _update_per_slice(self, ground_truth, prediction):
        """
        Find and match bounding box to ground truth.
        Update detected nodule and non-detected nodule.  
        Arguments:
            ground_truth (Array[N, 5])
            prediction (Array[M, 6])
        """
        self._is_sorted = False
        number_groundtruth_all = [0] * self.number_classes
        number_prediction_all = [0] * self.number_classes
        infos_all = [ [] for _ in range(self.number_classes + 1)]

        labels_groundtruth = np.array(ground_truth[:,0], dtype=np.uint16)
        bboxes_groundtruth = np.array(ground_truth[:,1:], dtype=np.float32)
        if prediction.shape[0] != 0:
            labels_prediction = np.array(prediction[:, 0], dtype=np.uint16)
            bboxes_prediction = np.array(prediction[:, 1:5], dtype=np.float32)
            scores_prediction = np.array(prediction[:, -1], dtype=np.float32)
        else:
            labels_prediction = np.array(prediction, dtype=np.uint16)
            bboxes_prediction = np.array(prediction, dtype=np.float32)
            scores_prediction = np.array(prediction, dtype=np.float32)

        number_groundtruth = len(bboxes_groundtruth)
        number_prediction  = len(bboxes_prediction)
        for label in labels_groundtruth:
            number_groundtruth_all[label] += 1
        for label in labels_prediction:
            number_prediction_all[label] += 1

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

        matched = linear_assignment(-matrix_IOU)
        unmatched_groundtruth = list(set(range(number_groundtruth))-set(matched[:,0]))
        unmatched_prediction  = list(set(range(number_prediction ))-set(matched[:,1]))
        for n,(i,j) in reversed(list(enumerate(matched))):
            if matrix_IOU[i,j] == 0:
                unmatched_groundtruth.append(i)
                unmatched_prediction.append(j)
                matched = np.delete(matched,n,0)
            else:
                infos_all[ labels_prediction[j] ].append([labels_groundtruth[i],
                                                               scores_prediction[j],
                                                               matrix_IOU[i,j], 
                                                               matrix_IOP[i,j], 
                                                               matrix_IOGT[i,j]])

        for i in unmatched_groundtruth:
            infos_all[ -1 ].append( [labels_groundtruth[i],0,0,0,0] )

        for j in unmatched_prediction:
            infos_all[ labels_prediction[j] ].append( [-1,scores_prediction[j],0, 0, 0])
        return infos_all, number_groundtruth_all, number_prediction_all


    def calculate_TPNFP_per_slice(self, infos_all, number_groundtruth_all,
                                number_prediction_all,
                                threshold_IoU, 
                                threshold_IoP,
                                threshold_IoGT):
        eps = 1e-50
        FP_list_prob_map = [0 * len(self.name_class)]
        sensitivity_list_prob_map = [0 * len(self.name_class)]
        matrix_confusion = np.zeros((self.number_classes+1,self.number_classes+1))
        for no_class_prediction in range(self.number_classes):
            infos = np.array(infos_all[no_class_prediction])
            infos = infos.reshape((-1,5))
            above_IOU = ( infos[:,2]>=threshold_IoU )
            above_IOP = ( infos[:,3]>=threshold_IoP)
            above_IOGT = (infos[:,4]>=threshold_IoGT)
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
            below_IOU = np.logical_not(above_IOU)

            matrix_confusion[-1,no_class_prediction] += np.sum(below_IOU)
            for no_class_groundtruth in range(self.number_classes):
                matched_class = ( infos[:,0] == no_class_groundtruth )
                if len(matched_class) == 0:
                    # print("matched_class empty")
                    continue
                matrix_confusion[no_class_groundtruth,no_class_prediction] += np.sum(matched_class & above_IOU)
                matrix_confusion[no_class_groundtruth,-1]                  += np.sum(matched_class & below_IOU)
        for no_class,name in enumerate(self.name_class):
            sensitivity = matrix_confusion[no_class,no_class]/(number_groundtruth_all[no_class]+eps)
            FP = number_prediction_all[no_class] - matrix_confusion[no_class, no_class]
            sensitivity_list_prob_map[no_class] = sensitivity
            FP_list_prob_map[no_class] = FP
        return sensitivity_list_prob_map, FP_list_prob_map

            

    def __process_update(self, thresh_confident):
        prediction = self._processNMS(thresh_confident)
        sensitivity_list = [[] * len(self.name_class)]
        FP_list = [[] * len(self.name_class)]
        # path_csv = os.path.join(self.dir, self.folder_ModelName, '003_Result', str(thresh_confident), 'result_conf_matrix_{}.csv'.format(now.strftime("%Y%m%d")))
        # self.convert_csv(prediction, path_csv)
        for key in self.ground_truth.keys():
            gt = self.ground_truth[key]
            if key in prediction.keys():
                bb = prediction[key]
            else:
                bb = []
            infor_all, number_gt, number_pred = self._update_per_slice(np.array(gt), np.array(bb))
            sensitivity_list_prob, FP_list_prob = self.calculate_TPNFP_per_slice(infor_all, number_gt, number_pred, 0.5,0.7,0.7)
            for no_class,name in enumerate(self.name_class):
                sensitivity_list[no_class].append(sensitivity_list_prob[no_class])
                FP_list[no_class].append(FP_list_prob[no_class])
        return np.mean(sensitivity_list, axis= 1), np.mean(FP_list, axis= 1)
    
    def calculate_FROC(self):
        sensitivity_list_threshold = [[] * len(self.name_class)]
        FPavg_list_threshold = [[] * len(self.name_class)]    
        threshold_list = (np.linspace(self.range_threshold[0], self.range_threshold[1], self.number_of_threshold)).tolist()
        for threshold in threshold_list:
            mean_sensitivity, mean_FP = self.__process_update(threshold)
            for no_class, _ in enumerate(self.name_class):
                sensitivity_list_threshold[no_class].append(mean_sensitivity[no_class])
                FPavg_list_threshold[no_class].append(mean_FP[no_class])
        return sensitivity_list_threshold, FPavg_list_threshold, threshold_list

    def plotFROC(self,x,y,save_path = None,threshold_list=None):
        plt.figure()
        plt.xlabel('FPavg/slice(image)')
        plt.ylabel('Sensitivity')
        plt.grid(True, color ='k')
        for no_class, _ in enumerate(self.name_class):
            plt.plot(x[no_class],y[no_class], 'o-', label = self.name_class[no_class]) 
            
        #annotate thresholds
        if threshold_list != None:
            #round thresholds
            threshold_list = [ '%.2f' % elem for elem in threshold_list ]            
            xy_buffer = None
            for i, xy in enumerate(zip(x, y)):
                if xy != xy_buffer:                                    
                    plt.annotate(str(threshold_list[i]), xy=xy, textcoords='data')
                    xy_buffer = xy
        plt.title("FROC")
        plt.legend(loc=2)
        plt.savefig(save_path)
        plt.show()




if __name__ == "__main__":
    directory = r'LUNA16\003_LUNA_train_test'
    GT_folder = "000_GroundTruth"
    # Model_folder = "001_ShanModel"
    Model_folder = "001_Model_20210718"
    eval = BME_FROC(directory, GT_folder, Model_folder,
                            ["nodule"])
    sensitivity_list, Fpavg_list, threshold_list = eval.calculate_FROC()
    eval.plotFROC(Fpavg_list, sensitivity_list, os.path.join(eval.save_result, 'FROC.png'))
