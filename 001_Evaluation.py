import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

now = datetime.now()

from libraries import utils

class Evaluation():

    def __init__(self, directory, gt_folder, test_folder):
        self.directory = directory
        self.gt_folder = gt_folder
        self.test_folder = test_folder
        gt_path = os.path.join(self.directory, self.gt_folder, 'groundTruth.csv')
        if os.path.exists(gt_path):
            self.gt_path = gt_path
        else:
            self.gt_path = None
        self.readGroundTruth()

        self.result_folder = os.path.join(self.directory, self.test_folder, '003_Result')
        self.TruePositive = 0
        self.FalsePositive = 0

    def clearValue(self):
        self.TruePositive = 0
        self.FalsePositive = 0


    def json2csv(self, pathjson:str, save = False, pathSave="BeforeNMS.csv"):
        if not os.path.exists(pathjson):
            print('Not find json file')
            return sys.exit()
        json = utils.readjson(pathjson)
        objectDict = {'filename': [], 
                    'x1': [], 
                    'y1': [],
                    'x2': [],
                    'y2': [],
                    'score': [],
                    'label': [] }
        for _, values in json.iterrows():
            if values['objects'] == []:
                continue
            filename = utils.get_filename(values['filename'])
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

    def readGroundTruth(self):
        if self.gt_path:
            self.gt_df = utils.readcsv(self.gt_path)
        else:
            print("Not find GroundTruth.csv")
            sys.exit()

    def calculateConfusionMatrixValues(self, bb_data):
        for _, bbLines in bb_data.iterrows():
            inResult = False
            isTp = False
            bb = (bbLines["x1"], bbLines["y1"], bbLines["x2"], bbLines["y2"])
            # print(bbLines['filename'])
            filename = utils.get_filename(bbLines['filename'])
            mini_gtLines = self.gt_df[self.gt_df['filename'] == filename]
            for _, GtLine in mini_gtLines.iterrows():
                inResult = True
                # gt = (int(GtLine[2]), int(GtLine[3]), int(GtLine[4]), int(GtLine[5]))
                gt = (int(GtLine[1]), int(GtLine[2]), int(GtLine[3]), int(GtLine[4]))
                iou = utils.compute_iou(bb, gt)
                if iou > utils.IOU_THRESHOLD:
                    self.TruePositive += 1
                    isTp = True
                else:
                    iop = utils.compute_iop(bb, gt)
                    if iop > utils.IOP_THRESHOLD:
                        self.TruePositive += 1
                        isTp = True
                    else:
                        iogt = utils.compute_iogt(bb, gt)
                        if iogt > utils.IOGT_THRESHOLD:
                            self.TruePositive += 1
                            isTp = True
            if isTp == False:
                if inResult == True:
                    self.FalsePositive +=1

    def calculateRecallNPrecision(self):
        Gt = len(self.gt_df)
        # Recall
        self.recall = (self.TruePositive/ Gt) * 100

        # Precision
        self.TotalObject = self.TruePositive + self.FalsePositive
        if self.TotalObject == 0:
            self.precision = 0
        else:
            self.precision = (self.TruePositive / self.TotalObject) * 100
        
        # F1 score
        self.f1 = (2 * self.recall * self.precision) / (self.recall + self.precision)

    def writetxt(self, path):
        TP = self.TruePositive
        # FP = self.FalsePositive
        GT = len(self.gt_df)
        TotalObject = self.TotalObject
        recall = self.recall
        precision = self.precision
        f1 = self.f1

        # Recall
        result = "Recall = True Positive/ Ground Truth = {}/{} = {}\n".format(TP, GT, recall)
        
        # Precision
        result += "Precision = True Positive/ Total Object = {}/{} = {}\n".format(TP, TotalObject, precision)

        # F1 Score
        result += "F1 score ={}".format(f1)
        print(result)
        with open(path, 'w') as txt:
            txt.writelines(result)
            txt.close()
 
if __name__ == "__main__":
    directory = r'D:\02_BME\003_evaluation_Model\LUNA16\003_LUNA_train_test'
    GT_folder = "000_GroundTruth"
    Model_folder = "001_Model_20210718"
    eval = Evaluation(directory, GT_folder, Model_folder)
    json_path = utils.select_file(os.path.join(directory, Model_folder))
    jsonData = eval.json2csv(pathjson= json_path)
    CONF_THRESH = [0.1, 0.5]
    print(Model_folder)
    for conf in CONF_THRESH:
        print("Confidence Threshold: {}".format(conf))
        path_conf_folder = os.path.join(eval.result_folder, str(conf))
        if not os.path.exists(path_conf_folder):
            os.makedirs(path_conf_folder)
        bb_data = eval.processNMS(jsonData, conf, True, os.path.join(path_conf_folder, 'result.csv'))
        eval.calculateConfusionMatrixValues(bb_data)
        eval.calculateRecallNPrecision()
        eval.writetxt(os.path.join(path_conf_folder, 'result_{}.txt'.format(now.strftime("%Y%m%d"))))
        eval.clearValue()

    print("Done")

        



    