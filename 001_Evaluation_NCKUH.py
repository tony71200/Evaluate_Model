import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

now = datetime.now()

from libraries import utils

class Evaluation_NCKUH():

    def __init__(self, directory:str, evaluate_folder:str, 
                list_folder:list,
                w_wo = [True, True, True, False, False]):
        self.dir = directory
        self.folder = evaluate_folder
        self.list_folder = list_folder
        self.w_wo_or_not = w_wo
        self.filename_list = []

        self.true_positive = [[0, 0]] * len(list_folder)
        self.false_positive = [[0, 0]] * len(list_folder)
        self.ground_truth = [[0, 0]] * len(list_folder)
        self.bb_data = self.json2csv()
        self.result_folder = os.path.join(self.dir, self.folder, '003_Result')
        

    def json2csv(self, save = False, pathSave="BeforeNMS.csv"):
        pathjson = utils.select_file(os.path.join(self.dir, self.folder))
        if pathjson == None:
            sys.exit()
        json = utils.readjson(pathjson)
        objectDict = {'filename': [], 
                    'x1': [], 
                    'y1': [],
                    'x2': [],
                    'y2': [],
                    'score': [],
                    'label': [],
                    'folder': [] }
        for _, values in json.iterrows():
            filename = utils.get_filename(values['filename'])
            self.filename_list.append(filename)
            if values['objects'] == []:
                continue
            folder = os.path.split(os.path.dirname(values['filename']))[-1]
            for obj in values['objects']:
                label = obj['class_id']
                x_center = obj['relative_coordinates']['center_x']
                y_center = obj['relative_coordinates']['center_y']
                w_yolo = obj['relative_coordinates']['width']
                h_yolo = obj['relative_coordinates']['height']
                score = obj['confidence']
                x1, y1, x2, y2 = utils.yolo2xy((x_center, y_center, w_yolo, h_yolo))
                objectDict['filename'].append(filename)
                objectDict['x1'].append(x1)
                objectDict['y1'].append(y1)
                objectDict['x2'].append(x2)
                objectDict['y2'].append(y2)
                objectDict['score'].append(score)
                objectDict['label'].append(label)
                objectDict['folder'].append(folder)
            
        df = pd.DataFrame(objectDict)
        if save:
            df.to_csv(pathSave)
        return df

    def processNMS(self, dataFrame, confidence, save = False, pathSave="result.csv"):
        if dataFrame.empty:
            print("Error Input Data Frame ")
            return None
        conf_df = dataFrame[dataFrame['score'] > confidence]
        compare_df = conf_df.copy()
        delete_index = []
        for index, loadrow in conf_df.iterrows():
            filename = loadrow['filename']
            mini_df = compare_df[compare_df['filename'] == filename]
            bb = (loadrow['x1'], loadrow['y1'], loadrow['x2'], loadrow['y2'])
            for _, compareRow in mini_df.iterrows():
                compare_bb = (compareRow['x1'], compareRow['y1'], compareRow['x2'], compareRow['y2'])
                if utils.compute_iou(bb, compare_bb) > utils.IOU_THRESHOLD:
                    if loadrow['score'] >= compareRow['score']:
                        pass
                    else:
                        # conf_df['filename'][index] = np.nan
                        # conf_df.loc[index, 'filename'] = None
                        delete_index.append(index)
        conf_df.drop(delete_index, axis = 0)
        if save:
            conf_df.to_csv(pathSave, index = False)
        return conf_df

    def processNMS2(self, dataFrame, confidence, save = False, pathSave="result.csv"):
        if dataFrame.empty:
            print("Error Input Data Frame ")
            return None
        conf_df = dataFrame[dataFrame['score'] > confidence]
        for filename in self.filename_list:
            mini_df = conf_df[conf_df['filename'] == filename]
            boxes = []
            for index, values in mini_df.iterrows():
                boxes.append([index, values[1], values[2], values[3], values[4], values[5]])
            boxes = np.array(boxes)
            nmsbox = utils.non_max_suppression_fast(boxes, 0.5)
            for values in nmsbox:
                nodule = utils.defineNodule(values[1], values[2], values[3], values[4], values[5])
                conf_df.loc[values[0], 'nodule'] = nodule
        conf_df.dropna(inplace = True)
        if save:
            conf_df.to_csv(pathSave, index = False)
        return conf_df

    @staticmethod
    def calculateConfusionMatrixValues(bb_data, gt_data):
        TruePositive = 0
        FalsePositive = 0
        for _, bbLines in bb_data.iterrows():
            inResult = False
            isTp = False
            bb = (bbLines["x1"], bbLines["y1"], bbLines["x2"], bbLines["y2"])
            # print(bbLines['filename'])
            filename = utils.get_filename(bbLines['filename']) + ".jpg"
            mini_gtLines = gt_data[gt_data['filename'] == filename]
            for _, GtLine in mini_gtLines.iterrows():
                inResult = True
                # gt = (int(GtLine[2]), int(GtLine[3]), int(GtLine[4]), int(GtLine[5]))
                gt = (int(GtLine["x1"]), int(GtLine["y1"]), int(GtLine["x2"]), int(GtLine["y2"]))
                iou = utils.compute_iou(bb, gt)
                if iou > utils.IOU_THRESHOLD:
                    TruePositive += 1
                    isTp = True
                else:
                    iop = utils.compute_iop(bb, gt)
                    if iop > utils.IOP_THRESHOLD:
                        TruePositive += 1
                        isTp = True
                    else:
                        iogt = utils.compute_iogt(bb, gt)
                        if iogt > utils.IOGT_THRESHOLD:
                            TruePositive += 1
                            isTp = True
            if isTp == False:
                if inResult == True:
                    FalsePositive +=1
        return TruePositive, FalsePositive

    @staticmethod
    def readGroundTruth(path:str):
        if path:
            return utils.readcsv(path)
        else:
            print("Not find GroundTruth.csv")
            sys.exit()

    def process(self, conf):
        for index, test_folder in enumerate(self.list_folder):
            bb_data = self.bb_data[self.bb_data['folder'] == test_folder]
            bb_data = self.processNMS2(bb_data, conf, save = True, 
                                    pathSave= os.path.join(self.result_folder, str(conf), "result_{}.csv".format(test_folder)))
            if self.w_wo_or_not[index]:
                path_gt_w = os.path.join(self.dir, "000_GroundTruth", "ggo_{}_w.csv".format(test_folder))
                path_gt_wo = os.path.join(self.dir, "000_GroundTruth", "ggo_{}_wo.csv".format(test_folder))

                gt_data_w = self.readGroundTruth(path_gt_w)
                gt_data_wo = self.readGroundTruth(path_gt_wo)

                self.ground_truth[index] = [len(gt_data_w), len(gt_data_wo)]

                tp_w, fp_w = self.calculateConfusionMatrixValues(bb_data, gt_data_w)

                tp_wo, fp_wo = self.calculateConfusionMatrixValues(bb_data, gt_data_wo)

                self.true_positive[index] = [tp_w, tp_wo]
                self.false_positive[index] = [fp_w, fp_wo]
            else:
                path_gt = os.path.join(self.dir, "000_GroundTruth", "ggo_{}.csv".format(test_folder))

                gt_data = self.readGroundTruth(path_gt)

                self.ground_truth[index] = [len(gt_data), 0]

                tp, fp = self.calculateConfusionMatrixValues(bb_data, gt_data)

                self.true_positive[index] = [tp, 0]
                self.false_positive[index] = [fp, 0]

    @staticmethod
    def calculateRnP(tp, fp, gt):
        recall = (tp/gt)*100 if gt !=0 else 0
        precision = (tp/ (tp + fp)) *100 if (tp + fp) != 0 else 0
        f1_score = (2 * recall * precision)/ (recall + precision) if (recall + precision) != 0 else 0

        # Recall
        result = "Recall = True Positive/ Ground Truth = {}/{} = {:.2f}\n".format(tp, gt, recall)
        
        # Precision
        result += "Precision = True Positive/ Total Object = {}/{} = {:.2f}\n".format(tp, (tp + fp), precision)

        # F1 Score
        result += "F1 score ={:.2f}".format(f1_score)
        return result

    def calculateRecallNPrecision_eachGroup(self):
        result = ""
        for index, test_folder in enumerate(self.list_folder):
            result += '\t'+ test_folder + '\n'
            if self.w_wo_or_not[index]:
                tp_w, tp_wo = self.true_positive[index]
                fp_w, fp_wo = self.false_positive[index]
                gt_w, gt_wo = self.ground_truth[index]
                result += "With pathlothogy\n"
                result += self.calculateRnP(tp_w, fp_w, gt_w) + "\n"
                result += "Without pathlothogy\n"
                result += self.calculateRnP(tp_wo, fp_wo, gt_wo) + "\n"
                result += "Total\n"
                result += self.calculateRnP(tp_w + tp_wo, fp_w + fp_wo, gt_w+ gt_wo) + "\n"
            else:
                tp_wo, _ = self.true_positive[index]
                fp_wo, _ = self.false_positive[index]
                gt_wo, _ = self.ground_truth[index]
                result += "Without pathlothogy\n"
                result += self.calculateRnP(tp_wo, fp_wo, gt_wo) + "\n"
        return result

    def calculateRecallNPrecision(self):
        result = ""
        wo_w_folder = [1,2]

        result += 'Data_Ln 2D_w_patho: \n'
        tp_w = np.sum(np.array([self.true_positive[index][0] for index in wo_w_folder]))
        fp_w = np.sum(np.array([self.false_positive[index][0] for index in wo_w_folder]))
        gt_w = np.sum(np.array([self.ground_truth[index][0] for index in wo_w_folder]))
        result += self.calculateRnP(tp_w, fp_w, gt_w) + '\n'

        result += 'Data_Ln 2D_wo_patho: \n'
        tp_wo = np.sum(np.array([self.true_positive[index][1] for index in wo_w_folder]))
        fp_wo = np.sum(np.array([self.false_positive[index][1] for index in wo_w_folder]))
        gt_wo = np.sum(np.array([self.ground_truth[index][1] for index in wo_w_folder]))
        result += self.calculateRnP(tp_wo, fp_wo, gt_wo) + '\n'

        result += 'Data_Me 2D_wo_patho: \n'
        tp_wo = self.true_positive[3][0]
        fp_wo = self.false_positive[3][0]
        gt_wo = self.ground_truth[3][0]
        result += self.calculateRnP(tp_wo, fp_wo, gt_wo) + '\n'

        return result

    def calculateRecallNPrecision_assign(self, list_assign = ['Test_B', 'Test_C', 'Test_Me']):
        result = ""
        true_positive = np.sum(np.array(self.true_positive), axis=1)
        false_positive = np.sum(np.array(self.false_positive), axis=1)
        ground_truth = np.sum(np.array(self.ground_truth), axis=1)

        assign = [self.list_folder.index(i) for i in list_assign]

        result += 'Total Data_Ln 2D \n'
        tp_w = np.sum(np.array([true_positive[index] for index in assign]))
        fp_w = np.sum(np.array([false_positive[index] for index in assign]))
        gt_w = np.sum(np.array([ground_truth[index]for index in assign]))
        result += self.calculateRnP(tp_w, fp_w, gt_w) + '\n'

        return result

    
if __name__ == "__main__":
    path = r"D:\02_BME\003_evaluation_Model\NCKUH"
    # evaluate_folder = "001_ShanModel"
    # evaluate_folder = "002_TonyModel"
    # evaluate_folder = "003_LUNA16Training"
    evaluate_folder = "005_TonyModel_2021_0716"
    list_folder = ['Test_A', 'Test_B', 'Test_C', 'Test_Me', 'Test_S']
    list_conf = [0.1, 0.5]
    evaluation = Evaluation_NCKUH(path, evaluate_folder, list_folder)
    for conf in list_conf:
        print("Confidence Threshold: {}".format(conf))
        path_conf_folder = os.path.join(evaluation.result_folder, str(conf))
        if not os.path.exists(path_conf_folder):
            os.makedirs(path_conf_folder)
        evaluation.process(conf)
        result = evaluation.calculateRecallNPrecision_eachGroup()
        result += "\n\n"
        result += evaluation.calculateRecallNPrecision()

        result += "\n\n"
        result += evaluation.calculateRecallNPrecision_assign()
        print(result)
        with open(os.path.join(path_conf_folder, 'result_{}.txt'.format(now.strftime("%m_%d_%Y"))), 'w+') as writeFile:
            writeFile.writelines(result)
            writeFile.close()
        
    # print(evaluation.calculateRecallNPrecision_eachGroup())

