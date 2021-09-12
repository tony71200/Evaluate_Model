import os
from glob import glob
import pandas as pd
if __name__ == '__main__':
    import utils
  
def read_json(path:str, isPytorch=True):
    if os.path.exists(path):
        json = pd.read_json(open(path, "r", encoding="utf8"))
        pred_dict = {}
        if isPytorch:
            for _, value in json.iterrows():
                filename = value['image_id']
                category = value['category_id']
                x1, y1, w, h = value['bbox']
                score = value['score']
                if not filename in pred_dict.keys():
                    pred_dict[filename] = [(x1, y1, x1+w, y1 + h, category, score)]
                else:
                    pred_dict[filename].append((x1, y1, x1 +w, y1+h, category, score))
        else:
            for _, row in json.iterrows():
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
                    bb = [x1, y1, x2, y2, label, score]
                    if not filename in pred_dict.keys():
                        pred_dict[filename] = [bb]
                    else:
                        pred_dict[filename].append(bb)
        return pred_dict

def convert2txt(pred_dict:dict):
    string = ""
    for filename in pred_dict.keys():
        string += filename
        for value in pred_dict[filename]:
            string += " {},{},{},{},{},{}".format(value[0], value[1], value[2], value[3], value [4], value[5])

        string += '\n'
    return string

def xywh2xyxy(bb, width = 512, height= 512):
    x_cen, y_cen, w_yolo, h_yolo = bb * (width, height, width, height)
    x1 = x_cen - w_yolo/2
    y1 = y_cen - h_yolo/2
    x2 = x_cen + w_yolo/2
    y2 = y_cen + h_yolo/2
    return x1, y1, x2, y2

def read_txt(path:str, extension = ".txt"):
    list_anno = []
    with open(path, 'r') as file_txt:
        for line in file_txt.readlines:
            line = line.strip().split()
            label = line[0]
            bb = (float(line[1]), float(line[2]), float(line[3]), float(line[4]))
            x1, y1, x2, y2 = xywh2xyxy(bb)
            list_anno.append((label, x1, y1, x2, y2))
        file_txt.close()
    return list_anno

def read_ground_truth_txt(path:str, ext=".txt"):
    with open(path, 'r+') as gt_file:
        groundtruth_dict = {}
        for line in gt_file.readlines():
            line = line.strip().split()
            filename = line[0]
            pred_bbs = []
            for pred_bb in line[1:]:
                pred_bb = pred_bb.split(',')
                pred_bbs.append((int(pred_bb[4]), float(pred_bb[0]), float(pred_bb[1]), float(pred_bb[2]), float(pred_bb[3])))
            groundtruth_dict[filename] = pred_bbs

        gt_file.close()
        return groundtruth_dict

def read_pred_txt(path:str, ext=".txt"):
    with open(path, 'r+') as pred_file:
        pred_dict = {}
        for line in pred_file.readlines():
            line = line.strip().split()
            filename = line[0]
            pred_bbs = []
            for pred_bb in line[1:]:
                pred_bb = pred_bb.split(',')
                pred_bbs.append((0, float(pred_bb[0]), float(pred_bb[1]), float(pred_bb[2]), float(pred_bb[3]), float(pred_bb[5])))
            pred_dict[filename] = pred_bbs

        pred_file.close()
        return pred_dict

def split_ln_me(gt_dict:dict, pred_dict:dict, save_file = "pred_Ln.txt"):
    s = ""
    for pred_filename in pred_dict.keys():
        if pred_filename in gt_dict.keys():
            s += pred_filename
            for pred_bb in pred_dict[pred_filename]:
                s += " {},{},{},{},{},{}".format(pred_bb[0], pred_bb[1], pred_bb[2], pred_bb[3], pred_bb[4], pred_bb[5])

            s += "\n"
    with open(save_file, 'w+') as write_file:
        write_file.writelines(s)
        write_file.close()


if __name__ == "__main__":
#     string = convert2txt(read_json(r"D:/PyTorch_YOLOv4/detections_val2017__results.json"))
#     with open("predict_result.txt", 'w+') as file_txt:
#         file_txt.writelines(string)
#         file_txt.close()
    path_gt = r"D:\02_BME\003_evaluation_Model\NCKUH-New\000_GroundTruth\Test_Me.txt"
    path_pred = r"D:\02_BME\003_evaluation_Model\NCKUH-New\005_modified_yolov4_aspp_darknet_20210910\Test_ncku_rgb_20210910.json"
    path_save = r"D:\02_BME\003_evaluation_Model\NCKUH-New\005_modified_yolov4_aspp_darknet_20210910\pred_Me.txt"
    gt_dict = read_ground_truth_txt(path_gt)
    pred_dict = read_json(path_pred, isPytorch=False)
    split_ln_me(gt_dict, pred_dict, path_save)
