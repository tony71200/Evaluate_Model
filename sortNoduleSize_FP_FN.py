import sys
import numpy as np
import glob
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from datetime import datetime
from libraries import utils
now = datetime.now()
    
def drawBB(image:Image, annotations, color = "green"):
    """
    Annotation is a list of annotaion with format x1, y1, x2, y2
    """
    draw_image = ImageDraw.Draw(image)
    for annotation in annotations:
        pt1 = (annotation[0], annotation[1])
        pt2 = (annotation[2], annotation[3])
        draw_image.rectangle([pt1, pt2], outline=color)

    return image 

def readImage(path:str):
    return Image.open(path)

def writeImage(image:Image, path:str):
    return image.save(path, "JPEG")

def resize_image(image:Image, size, resample = Image.BICUBIC):
    return image.resize(size, resample = resample)

def crop_image(image: Image, bb):
    left, top, right, bottom = bb
    left = left -20
    top = top - 20
    right = right + 20
    bottom = bottom +20
    im1 = image.crop((left, top, right, bottom))
    return im1

def get_concate_h(img1, img2, color = (0,0,0)):
    position = (img1.size[0] + img2.size[0], max(img1.size[1], img2.size[1]))
    dst = Image.new('RGB', position, color)
    dst.paste(img1, (0,0))
    dst.paste(img2, (img1.size[0], 0))
    return dst

def get_concat_inside_image(img1: Image, img2: Image, bb:list):
    """
    bb: bounding box of nodule
    it includes x1, y1, x2, y2
    """
    w1, h1 = img1.size
    w2, h2 = img2.size
    w = w1 if w1 > w2 else w2
    h = h1 if h1 > h2 else h2

    dst = Image.new('RGB', (w, h), (0,0,0))
    dst.paste(img1, (0, 0))
    
    delta_w = w - w2
    delta_h = h - h2
    if bb[2] > delta_w:
        pt2 = (0, delta_h)
    else:
        pt2 = (delta_w, 0)

    dst.paste(img2, pt2)
    return dst

def make_new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def bySize(bb):
    bounding_box = np.array(bb, dtype=float)
    w, h = bounding_box[2:] - bounding_box[:2]
    temp_area = w * h
    if temp_area < 100:
        return 'Benign'
    elif temp_area < 178 and temp_area >= 100 :
        return 'Prob. Benign'
    elif temp_area < 625 and temp_area >= 178:
        return 'Prob. Sus.'
    elif temp_area >= 625:
        return 'Sus.'

def bySize2(bb):
    bounding_box = np.array(bb, dtype=float)
    w, h = bounding_box[2:] - bounding_box[:2]
    temp_area = w * h
    if temp_area < 178:
        return 'small'
    elif temp_area < 625 and temp_area >= 178:
        return 'medium'
    elif temp_area >= 625:
        return 'large'

class ObjectDetectMetric_bySize():
    def __init__(self, dir:str, path_image:str,folder_GroundTruth:str, 
                folder_ModelName:str,
                list_folder = ['subset{}'.format(i) for i in range(10)], 
                check_class_first=True,
                extension = '*.jpg'):
        self.dir = dir
        self.folder_GrouthTruth = folder_GroundTruth
        self.folder_ModelName = folder_ModelName
        self.check_class_first = check_class_first
        self.names_class = []
        
        self._is_sorted = False

        path_gt = os.path.join(self.dir, self.folder_GrouthTruth, "GroundTruth.csv")
        self.ground_truth = self._load_groundtruth(path_gt)
        path_pred = utils.select_file(os.path.join(self.dir, self.folder_ModelName))
        self.prediction = self._load_prediction(path_pred)
        self.save_result = os.path.join(dir, folder_ModelName, "003_Result")

        self.number_classes = len(self.names_class)
        self.number_groundtruth_all = [0] * self.number_classes
        self.number_prediction_all = [0] * self.number_classes
        self.infos_all = [ [] for _ in range(self.number_classes + 1)]
        self.False_negative = {'filename': [], 'bounding_box': []}
        self.False_positive = {'filename': [], 'bounding_box': []}

        self.list_path_image = []
        for subset in list_folder:
            self.list_path_image.extend(glob.glob(os.path.join(path_image, subset, extension)))

        # filter_object = filter(lambda path: "IMG-0001-0001" in path, self.list_path_image)
        # print(list(filter_object))


    def clear(self):
        self.number_groundtruth_all = [0] * self.number_classes
        self.number_prediction_all = [0] * self.number_classes
        self.infos_all = [ [] for _ in range(self.number_classes + 1)]
        self._is_sorted = False
        
    def _load_groundtruth(self, path_groundTruth):
        if os.path.exists(path_groundTruth):
            dict_gt = {}
            gt_df = utils.readcsv(path_groundTruth)
            for _, row in gt_df.iterrows():
                filename = row['filename']
                type_nodule = bySize([row['x1'], row['y1'], row['x2'], row['y2']])
                if not (type_nodule in self.names_class):
                    self.names_class.append(type_nodule)
                bb = [self.names_class.index(type_nodule), row['x1'], row['y1'], row['x2'], row['y2'], 1.0]
                if not filename in dict_gt.keys():
                    dict_gt[filename] = [bb]
                else:
                    dict_gt[filename].append(bb)

                if not (type_nodule in self.names_class):
                    self.names_class.append(type_nodule)
            return dict_gt
        else:
            print("Not find GroundTruth.csv")
            sys.exit()

    def _load_prediction(self, path_prediction):
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
                    type_nodule = bySize([x1,y1, x2, y2])
                    if not (type_nodule in self.names_class):
                        self.names_class.append(type_nodule)
                    bb = [self.names_class.index(type_nodule), x1, y1, x2, y2, score]
                    if not filename in dict_pred.keys():
                        dict_pred[filename] = [bb]
                    else:
                        dict_pred[filename].append(bb)

                    if not (type_nodule in self.names_class):
                        self.names_class.append(type_nodule)
            return dict_pred
        else:
            print("Not find {}".format(utils.get_basename(path_prediction)))
            sys.exit()

    @staticmethod
    def _get_IOUs(bboxes1_xyxy,bboxes2_xyxy, get='iou'):
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
        if get.lower() == 'iou':
            return area_i/(area1+area2-area_i)
        elif get.lower() == 'iop':
            return area_i/(area2)
        elif get.lower() == 'iogt':
            return area_i/(area1)
        else:
            print("Wrong option: only fill 'iou', 'iop' or 'iogt'")
            sys.exit()

    @staticmethod
    def convert_csv(dictionary:dict, pathsave):
        dictionary_fn = {'filename': [],
                    'bounding box': []}

        for key in dictionary.keys():
            for bounding_box in dictionary[key]:
                dictionary_fn["filename"].append(key)
                dictionary_fn["bounding box"].append(bounding_box)
        df = pd.DataFrame(dictionary_fn, index=None)
        df.to_csv(pathsave)

    @staticmethod
    def appendTable(table:dict, filename:str, bb):
        table['filename'].append(filename)
        table['bounding_box'].append(bb)
        return table
        
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
        # number_object = 0
        # for key in prediction.keys():
        #     number_object += len(prediction[key])
        # print(number_object)
        return prediction
    
    def _update(self, ground_truth, prediction, filename:str, confident_threshold, threshold_iou = 0.5, threshold_iop= 0.7, threshold_iogt = 0.7):
        """
        Find and match bounding box to ground truth.
        Update detected nodule and non-detected nodule.  
        Arguments:
            ground_truth (Array[N, 5])
            prediction (Array[M, 6])
            filename: string
        """
        self._is_sorted = False
        labels_groundtruth = np.array(ground_truth[:,0], dtype=np.uint16)
        bboxes_groundtruth = np.array(ground_truth[:,1:5], dtype=np.float32)
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
            self.number_groundtruth_all[label] += 1
        for label in labels_prediction:
            self.number_prediction_all[label] += 1

        matrix_IOU = self._get_IOUs(bboxes_groundtruth,bboxes_prediction)
        matrix_IOP = self._get_IOUs(bboxes_groundtruth,bboxes_prediction, get='iop')
        matrix_IOGT = self._get_IOUs(bboxes_groundtruth, bboxes_prediction, get='iogt')
        same = labels_groundtruth.reshape((-1,1))==labels_prediction.reshape((1,-1))
        if self.check_class_first:
            matrix_IOU *= same
            matrix_IOP *= same
            matrix_IOGT *= same
        else:
            matrix_IOU[same] *= (1+1e-06)
            matrix_IOP[same] *= (1+1e-06)
            matrix_IOGT[same] *= (1+1e-06)

        # ious = matrix_IOU > threshold_iou
        # iops = matrix_IOP > threshold_iop
        # iogts = matrix_IOGT > threshold_iogt

        # ious = ious.tolist()[0]
        # iops = iops.tolist()[0]
        # iogts = iogts.tolist()[0]
        # total_threshold = []
        # for index, (iou, iop, iogt) in enumerate(zip(ious, iops, iogts)):
        #     if iou:
        #         total_threshold.append(matrix_IOU[index])
        #     else:
        #         if iop:
        #             total_threshold.append(matrix_IOP[index])
        #         else:
        #             total_threshold.append(matrix_IOGT[index])
        # total_threshold = np.array(total_threshold)

        if matrix_IOU.size != 0:
            ind = np.argwhere(matrix_IOU==np.amax(matrix_IOU,1, keepdims=True))
            self.matched = np.array(list(map(list, ind)))
        else:
            self.matched = np.empty((0,2), dtype=np.int32)

        # if total_threshold.size != 0:
        #     ind = np.argwhere(total_threshold==np.amax(total_threshold,1, keepdims=True))
        #     self.matched = np.array(list(map(list, ind)))
        # else:
        #     self.matched = np.empty((0,2), dtype=np.int32)
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

        for i in self.unmatched_groundtruth:
            self.infos_all[ -1 ].append( [labels_groundtruth[i],0,0,0,0] )
            # self.appendTable(self.False_negative, filename, bboxes_groundtruth[i])
            path_save = os.path.join(self.save_result, str(confident_threshold), 'result_image', 'FN')
            make_new_folder(path_save)
            self.draw_image(filename, bboxes_groundtruth[i], bboxes_prediction, path_save, i, 'red', 'blue')
            
            

        for j in self.unmatched_prediction:
            self.infos_all[ labels_prediction[j] ].append( [-1,scores_prediction[j],0, 0, 0])
            # self.appendTable(self.False_positive, filename, bboxes_prediction[j])
            path_save = os.path.join(self.save_result, str(confident_threshold), 'result_image', 'FP')
            make_new_folder(path_save)
            self.draw_image(filename, bboxes_prediction[j], bboxes_groundtruth,  path_save, j, 'blue', 'red')

    def draw_image(self, filename:str, bounding_box:list, list_bb_gt_pred:list, path_save:str, index = 0, color1 = 'blue', color2 = 'red'):
        filter_object = filter(lambda path: filename in path, self.list_path_image)
        path_image = (list(filter_object))[0]
        image = readImage(path_image)
        image = drawBB(image, [bounding_box], color1)
        if len(list_bb_gt_pred) != 0:
            image = drawBB(image, list_bb_gt_pred, color2)
        crop = crop_image(image, bounding_box)
        crop = resize_image(crop, (170,170))
        crop = ImageOps.expand(crop, border = 5, fill="#000")
        # concat_image = get_concate_h(image, crop)
        concat_image = get_concat_inside_image(image, crop, bounding_box)
        folder = bySize2(bounding_box)
        path_folder = os.path.join(path_save, folder)
        make_new_folder(path_folder)
        save_filename = os.path.join(path_folder, filename + "_{}.jpg".format(index))
        writeImage(concat_image, save_filename)
        


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
        for no_class,name in enumerate(self.names_class):
            precision = matrix_confusion[no_class,no_class]/(self.number_prediction_all[ no_class]+eps)
            recall    = matrix_confusion[no_class,no_class]/(self.number_groundtruth_all[no_class]+eps)
            f1_score = (2 * recall *100 * precision)/(recall+ precision)
            content += ("[%*s]   recall: %d/ %d: %6.2f %%"
                              "     precision: %d/ %d: %6.2f %%"
                              "     f1-score: %6.2f %%"
                             "     avg IOU: %6.2f %%\n")%(\
                    length_name,name,
                    matrix_confusion[no_class,no_class], (self.number_groundtruth_all[no_class]+eps), 1e2*recall,
                    matrix_confusion[no_class,no_class], (self.number_prediction_all[ no_class]+eps), 1e2*precision,
                    f1_score,
                    1e2*IOUs_avg[no_class])
        content += spacing
        if conclude:
            print(content)
        else:
            return content
    
    def processAupdate(self, thresh_confident):
        prediction = self._processNMS(thresh_confident)
        path_csv = os.path.join(self.dir, self.folder_ModelName, '003_Result', str(thresh_confident), 'result_conf_matrix_size{}.csv'.format(now.strftime("%Y%m%d")))
        self.convert_csv(prediction, path_csv)
        for key in self.ground_truth.keys():
            gt = self.ground_truth[key]
            if key in prediction.keys():
                bb = prediction[key]
            else:
                bb = []
            self._update(np.array(gt), np.array(bb), key, thresh_confident)

if __name__ == "__main__":
    directory = r'D:\02_BME\003_evaluation_Model\LUNA16\001_LUNA_all_Gray24bit'
    path_image_folder = r'D:\02_BME\000_LUNA16\004_Gray24bit_CAD_All'
    GT_folder = "000_GroundTruth"
    Model_folder = "001_ShanModel"
    confident_list = [0.1,0.5]
    eval = ObjectDetectMetric_bySize(directory, path_image_folder, GT_folder, Model_folder)
    for conf in confident_list:
        if not os.path.exists(os.path.join(eval.save_result, str(conf))):
            os.makedirs(os.path.join(eval.save_result, str(conf)))
        eval.processAupdate(conf)
        result = eval.get_confusion(conf,0.5,0.7,0.7, False)
        print(result)
        with open(os.path.join(eval.save_result, str(conf), "confusion_maxtrix_size_{}.txt".format(now.strftime("%Y%m%d"))), 'w+') as writeLine:
            writeLine.writelines(result)
            writeLine.close()
        eval.clear()
