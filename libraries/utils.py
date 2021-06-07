import pandas as pd
import os
import cv2 as cv


def readcsv(path:str):
    return pd.read_csv(path)

def writecsv(dictionary:dict, path:str):
    try:
        df = pd.DataFrame(dictionary)
        df.to_csv(path)
    except:
        print("Dont save. Please check again")

def readjson(path:str):
    return pd.read_json(path)

def yolo2xy(bb:tuple, width = 512, height = 512):
    x1 = round(0.5 * width * (2 * bb[0] - bb[2]))
    y1 = round(0.5 * height * (2 * bb[1] - bb[3]))
    x2 = round(0.5 * width * (2 * bb[0] + bb[2]))
    y2 = round(0.5 * height * (2 * bb[1] + bb[3]))
    return x1, y1, x2, y2

def drawRect(image, bb:tuple, color = (0,255,0)):
    pt1 = (bb[0], bb[1])
    pt2 = (bb[2], bb[3])
    return cv.rectangle(image, pt1, pt2, color, thickness= 1, lineType=cv.LINE_AA)

def readImage(path:str):
    return cv.imread(path)

def writeImage(path:str, image):
    return cv.imwrite(path, image)

def get_filename(path:str):
    return os.path.splitext(os.path.basename(path))[0]

def get_dirname(path:str):
    return os.path.dirname(path)

def get_basename(path:str):
    return os.path.basename(path)

IOU_THRESHOLD = 0.5
IOP_THRESHOLD = 0.7
IOGT_THRESHOLD = 0.7

def compute_iou(bb:tuple, gt:tuple):
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

def compute_iop(bb:tuple, gt:tuple):
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
        return (it_area / bb_area)*1.0

def compute_iogt(bb:tuple, gt:tuple):
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
        return (it_area / gt_area)*1.0

if __name__ == "__main__":
    path = r"D:\Master\02_Project\LUNA16\evaluationModel\LUNA16\resultAllSubset.json"
    print(readjson(path))

