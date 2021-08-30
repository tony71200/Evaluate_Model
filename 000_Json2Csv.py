import json
import csv

# Path = 'D:/ModelResult/'
Path = ""
#ModelName = '001_Original'
#ModelName = '002_2xinteration'
#ModelName = '003_LateralCsp'
#ModelName = '004_Lateralc1c2'
# ModelName = '006_Test'
#ModelName = '005_Lateralc1c2Sppx2'
#ModelName = '006_yolov4_Origin'
# ModelName = '007_yolov4_4P'
# ModelName = '009_EfficientD4'
ModelName = '010_EfficientDet_D1'
ConfScore = ['0.05', '0.01']
TestName = ['A', 'B', 'C', 'Me']
iou_thres = 0.3

def compute_iou(bb, gt):
    # bb[0]: x1, bb[1]: y1, bb[2]: x2, bb[3]: y2
    # gt[0]: x1, gt[1]: y1, gt[2]: x2, gt[3]: y2

    # computing area of each rectangles
    bb_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    # computing the sum_area
    sum_area = bb_area + gt_area

    # find the each edge of intersect rectangle
    it = [0, 0, 0, 0]
    it[0] = max(bb[0], gt[0])
    it[1] = max(bb[1], gt[1])
    it[2] = min(bb[2], gt[2])
    it[3] = min(bb[3], gt[3])

    # judge if there is an intersect
    if it[0] >= it[2] or it[1] >= it[3]:
        return 0
    else:
        it_area = (it[2] - it[0]) * (it[3] - it[1])
        return (it_area / (sum_area - it_area))*1.0

for cs in range(0, 1):
    for tn in range(0, 4):
        JsonPath = Path + ModelName + '/003_Result/' + ConfScore[cs] + '/ggo_test_' + TestName[tn] + '.json'
        CsvPath = Path + ModelName + '/003_Result/' + ConfScore[cs] + '/ggo_test_' + TestName[tn] + '_beforedoublenms.csv'
        with open(JsonPath) as JsonFile:
            JsonDatas = json.load(JsonFile)
        with open(CsvPath, 'w', newline='') as CsvFile:
            CsvWriter = csv.writer(CsvFile)
            CsvWriter.writerow(['filename', 'x1', 'y1', 'x2', 'y2', 'score'])
            for JsonData in JsonDatas:
                Index = []
                if JsonData['objects']:
                    #Index.append(str(JsonData['frame_id']))
                    Index.append(str(JsonData['filename']))
                    for obj in range(0, len(JsonData['objects'])):
                        #Index.append(str(JsonData['objects'][obj]['class_id']))
                        #Index.append(str(JsonData['objects'][obj]['name']))
                        center_x = float(JsonData['objects'][obj]['relative_coordinates']['center_x'])
                        center_y = float(JsonData['objects'][obj]['relative_coordinates']['center_y'])
                        yolo_width = float(JsonData['objects'][obj]['relative_coordinates']['width'])
                        yolo_height = float(JsonData['objects'][obj]['relative_coordinates']['height'])                
                        x1 = str(round(0.5 * 512 * (2 * center_x - yolo_width)))
                        y1 = str(round(0.5 * 512 * (2 * center_y - yolo_height)))
                        x2 = str(round(0.5 * 512 * (2 * center_x + yolo_width)))
                        y2 = str(round(0.5 * 512 * (2 * center_y + yolo_height)))                
                        Index.append(x1)
                        Index.append(y1)
                        Index.append(x2)
                        Index.append(y2)
                        Index.append(str(JsonData['objects'][obj]['confidence']))
                        CsvWriter.writerow(Index)
                        Index = []
                        #Index.append(str(JsonData['frame_id']))
                        Index.append(str(JsonData['filename']))
        CsvFile.close()
for cs in range(0, 1):
    for tn in range(0, 4):
        LoadPath = Path + ModelName + '/003_Result/' + ConfScore[cs] + '/ggo_test_' + TestName[tn] + '_beforedoublenms.csv'
        SavePath = Path + ModelName + '/003_Result/' + ConfScore[cs] + '/ggo_test_' + TestName[tn] + '.csv'
        print(SavePath)
        LoadList = []
        SaveList = []
        CompareList = []
        with open(LoadPath, 'r', newline='') as LoadFile:
            LoadRows = csv.reader(LoadFile)
            for LoadRow in LoadRows:
                if LoadRow[0][0] != 'f':
                    LoadList.append(LoadRow)
                    CompareList.append(LoadRow)
        with open(SavePath, 'w+', newline='') as SaveFile:
            SaveWriter = csv.writer(SaveFile)
            SaveWriter.writerow(['filename', 'x1', 'y1', 'x2', 'y2', 'score'])
            for LoadLine in LoadList:
                for CompareLine in CompareList:
                    if LoadLine[0] == CompareLine[0]:
                        bb = (int(LoadLine[1]), int(LoadLine[2]), int(LoadLine[3]), int(LoadLine[4]))
                        gt = (int(CompareLine[1]), int(CompareLine[2]), int(CompareLine[3]), int(CompareLine[4]))
                        iou = compute_iou(bb, gt)
                        if iou > iou_thres:
                            if float(LoadLine[5]) >= float(CompareLine[5]):
                                pass
                            else:
                                LoadLine[0] = 'failed'
            for LoadLine in LoadList:
                if LoadLine[0] != 'failed':
                    SaveWriter.writerow(LoadLine)
            CsvFile.close()

