import pandas as pd
import os
import cv2 as cv
import numpy as np

def readcsv(path:str):
    return pd.read_csv(path)

def writecsv(dictionary:dict, path:str):
    try:
        df = pd.DataFrame(dictionary)
        df.to_csv(path)
    except:
        print("Dont save. Please check again")

def readjson(path:str):
    return pd.read_json(open(path, "r", encoding="utf8"))
    # try:
    #     return pd.read_json(open(path, "r", encoding="utf8"))
    # except:
    #     with open(path) as JsonFile:
    #         line = JsonFile.readlines()
    #         print(line)

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

def compute_iou_iop_iogt(bb:tuple, gt:tuple):
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
        return (it_area / (sum_area - it_area))*1.0, (it_area / bb_area)*1.0, (it_area / gt_area)*1.0

def make_ground_truth_table(path_folder:str, subset_list:list, extension = "*.txt"):
    from glob import glob
    import pandas as pd
    groundtruth = {'filename': [],
                    'x1': [],
                    'y1': [],
                    'x2': [],
                    'y2': [],
                    'subset':[]}
    for subset in subset_list:
        for path_txt in glob(os.path.join(path_folder, subset, extension)):
            if get_basename(path_txt) == "classes.txt":
                continue
            filename = get_filename(path_txt)
            with open(path_txt, 'r') as anno:
                for line in anno.readlines():
                    line = line.strip().split(" ")
                    if len(line) == 0:
                        continue
                    label = line[0]
                    bb = (float(line[1]), float(line[2]), float(line[3]), float(line[4]))
                    x1, y1, x2, y2 = yolo2xy(bb)
                    groundtruth['filename'].append(filename)
                    groundtruth['x1'].append(x1)
                    groundtruth['y1'].append(y1)
                    groundtruth['x2'].append(x2)
                    groundtruth['y2'].append(y2)
                    groundtruth['subset'].append(subset)
                    #groundtruth['nodule'].append(label)
        
    df = pd.DataFrame(groundtruth)
    df.to_csv(os.path.join(path_folder, 'GroundTruth.csv'), index=None)

def select_file(root_path:str):
    import tkinter as tk
    from tkinter import filedialog as fd
    root = tk.Tk()
    root.withdraw()
    filetypes = (
        ('JSON files', '*.json'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a JSON file',
        initialdir= root_path,
        filetypes=filetypes)
    root.destroy()
    return filename

def non_max_suppression_fast(boxes:list, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,1]
	y1 = boxes[:,2]
	x2 = boxes[:,3]
	y2 = boxes[:,4]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick]

def defineNodule(x1, y1, x2, y2, score):
    temp_area = (x2 - x1) * (y2 - y1)
    if temp_area < 45 and float(score) >= 0.5:
        return 'Benign'
    elif temp_area < 100 and temp_area >= 45 and float(score) >= 0.1:
        return 'Prob. Benign'
    elif temp_area < 180 and temp_area >= 100 and float(score) >= 0.05:
        return 'Prob. Sus.'
    elif temp_area >= 180 and float(score) >= 0.01:
        return 'Sus.'
if __name__ == "__main__":
    # path = r"D:\BME\000_LUNA16\000_002_Gray_2D_CAD_Ellip\Nodule"
    # subset_list = ['subset{}'.format(i) for i in range(10)]
    # make_ground_truth_table(path, subset_list)
    path = r"D:\yolov4\darknet\build\darknet\x64\data"
    subset_list = ["luna_test"]
    make_ground_truth_table(path, subset_list)

