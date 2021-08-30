import sys
import cv2
import os
import pandas as pd
from pandas.core.frame import DataFrame
from glob import glob

def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def readCSV(path:str):
    return pd.read_csv(path)

def _load(path):
    if os.path.exists(path):
        dictionary = {}
        df = readCSV(path)
        for _, row in df.iterrows():
            filename = row['filename']
            bb = [row['x1'], row['y1'], row['x2'], row['y2']]
            if not filename in dictionary.keys():
                dictionary[filename] = [bb]
            else:
                dictionary[filename].append(bb)
        return dictionary
    else:
        print("Not find {path}".format(path = path))
        sys.exit()

def draw(image, bb, color:tuple):
    pt1 = (bb[0], bb[1])
    pt2 = (bb[2], bb[3])
    return cv2.rectangle(image, pt1, pt2, color, 1)

def main(extension = ".jpg"):
    path_gt = r"D:\02_BME\003_evaluation_Model\LUNA16\004_LUNA_Gray24bit_CAD\000_GroundTruth\groundTruth.csv"
    path_pred = r"D:\02_BME\003_evaluation_Model\LUNA16\004_LUNA_Gray24bit_CAD\001_Model_20210720\003_Result\0.5\result.csv"
    path_image = r"D:\02_BME\000_LUNA16\003_Gray24bit_CAD"

    save_path = os.path.dirname(path_pred)
    save_folder = os.path.join(save_path, "result_image")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    ground_truth = _load(path_gt)
    prediction = _load(path_pred)

    subsets = ["subset{}".format(index) for index in range(8,10,1)]

    filenames = []
    subset_folder = []
    for subset in subsets:
        for path_img in glob(os.path.join(path_image, subset, "*{}".format(extension))):
            filenames.append(get_filename(path_img))
            subset_folder.append(subset)

    for subset, filename in zip(subset_folder, filenames):
        if filename in ground_truth.keys() and filename in prediction.keys():
            path_img = os.path.join(path_image, subset, filename + extension)
            img = cv2.imread(path_img)
            bb_gt = ground_truth[filename]
            bb_pred = prediction[filename]
            path_output_tp = os.path.join(save_folder, "TP")
            if not os.path.exists(path_output_tp):
                os.makedirs(path_output_tp)
            path_output_tp_subset = os.path.join(path_output_tp, subset)
            if not os.path.exists(path_output_tp_subset):
                os.makedirs(path_output_tp_subset)
            
            for bb in bb_gt:
                img = draw(img, bb, (0, 255, 0))
            for bb in bb_pred:
                img = draw(img, bb, (255, 0, 0))
            save_img = os.path.join(path_output_tp_subset, filename + extension)
            cv2.imwrite(save_img, img)
            print("TP", filename)
        else:
            if filename in prediction.keys():
                path_img = os.path.join(path_image, subset, filename + extension)
                img = cv2.imread(path_img)
                bb_pred = prediction[filename]
                path_output_fp = os.path.join(save_folder, "FP")
                if not os.path.exists(path_output_fp):
                    os.makedirs(path_output_fp)
                    path_output_fp_subset = os.path.join(path_output_fp, subset)
                if not os.path.exists(path_output_fp_subset):
                    os.makedirs(path_output_fp_subset)
                
                for bb in bb_pred:
                    img = draw(img, bb, (255, 0, 0))
                save_img = os.path.join(path_output_fp_subset, filename + extension)
                cv2.imwrite(save_img, img)
                print("FP", filename)
            if filename in ground_truth.keys():
                path_img = os.path.join(path_image, subset, filename + extension)
                img = cv2.imread(path_img)
                bb_gt = ground_truth[filename]
                path_output_fn = os.path.join(save_folder, "FN")
                if not os.path.exists(path_output_fn):
                    os.makedirs(path_output_fn)
                path_output_fn_subset = os.path.join(path_output_fn, subset)
                if not os.path.exists(path_output_fn_subset):
                    os.makedirs(path_output_fn_subset)
                
                for bb in bb_gt:
                    img = draw(img, bb, (0, 255, 0))
                save_img = os.path.join(path_output_fn_subset, filename + extension)
                cv2.imwrite(save_img, img)
                print("FN", filename)
    print("DONE")







if __name__ == "__main__":
    main()
    