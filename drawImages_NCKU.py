import sys
import os
import glob
import PIL
from PIL import Image, ImageDraw, ImageOps
import pandas as pd

def readImage(path:str):
    return Image.open(path)

def writeImage(image:Image, path:str):
    return image.save(path, "JPEG")

def drawBB(image:Image, annotations:list, color = "green"):
    """
    Annotation is a list of annotaion with format x1, y1, x2, y2
    """
    draw_image = ImageDraw.Draw(image)
    for anno in annotations:
        pt1 = (anno[0], anno[1])
        pt2 = (anno[2], anno[3])
        draw_image.rectangle([pt1, pt2], outline=color)

    return image 

def resize_imagePIL(image:Image, size, resample = Image.BICUBIC):
    return image.resize(size, resample = resample)

def crop_imagePIL(image: Image, bb):
    left, top, right, bottom = bb
    left = left -20
    top = top - 20
    right = right + 20
    bottom = bottom +20
    im1 = image.crop((left, top, right, bottom))
    im1 = ImageOps.expand(im1, border=2, fill="#000")
    return im1

def get_concate_h(img1, img2:list, color = (0,0,0)):
    """
    set defaut size:
    image1 : (512x512)
    list image2 : (256x256)
    """
    index_num = len(img2)//2
    position = (img1.size[0] + (index_num +1) * img2[0].size[0], max(img1.size[1], img2[0].size[1]))
    dst = Image.new('RGB', position, color)
    dst.paste(img1, (0,0))
    for index, img in enumerate(img2):
        row, column = index %2, index //2
        dst.paste(img, (img1.size[0] + column * img.size[0], row * img.size[1]))
    return dst

def getbasename(path:str):
    return os.path.basename(path)

def getfolder(path:str, folder_in_basename = False):
    if not folder_in_basename:
        return os.path.split(os.path.dirname(path))[-1]
    else:
        folder = getfilename(path).split('_', 1)[-1]
        return folder

def getdirname(path:str):
    return os.path.dirname(path)

def getfilename(path:str):
    return os.path.splitext(os.path.basename(path))[0]

def readcsv(path:str):
    if os.path.exists(path) and path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        print("Error: {} is wrong".format(path))
        sys.exit()

def select_file(root_path:str, title:str = 'Open a CSV file', filetypes = (('CSV file', '*.csv'),('All files', '*.*'))):
    import tkinter as tk
    from tkinter import filedialog as fd
    root = tk.Tk()
    root.withdraw()
    filetypes = filetypes

    filename = fd.askopenfilename(
        title= title,
        initialdir= root_path,
        filetypes=filetypes)
    root.destroy()
    return filename

def select_directionary(root_path:str, title:str = 'Select folder'):
    import tkinter as tk
    from tkinter import filedialog as fd
    root = tk.Tk()
    root.withdraw()

    filename = fd.askdirectory(initialdir=root_path, title= title)
    root.destroy()
    return filename

def _load(path, path2 = None):
    if os.path.exists(path):
        dictionary = {}
        df = readcsv(path)
        for _, row in df.iterrows():
            filename = getfilename(row['filename'])
            bb = [row['x1'], row['y1'], row['x2'], row['y2']]
            if not filename in dictionary.keys():
                dictionary[filename] = [bb]
            else:
                dictionary[filename].append(bb)
        if path2 != None and os.path.exists(path2):
            df = readcsv(path2)
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

def create_dirs(path:str):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_images(path_groundtruth:str, path_prediction:str, path_folder_image:str,
                path_groundtruth2 =None, extension = ".jpg"):
    path_save = os.path.join(getdirname(path_prediction), 'result_image')
    create_dirs(path_save)

    ground_truth = _load(path_groundtruth, path_groundtruth2)
    prediction = _load(path_prediction)

    folder = getfolder(path_prediction, folder_in_basename=True)
    # path_output_tp = os.path.join(path_save, "TP")
    # create_dirs(path_output_tp)
    # path_output_tp_subset = os.path.join(path_output_tp, subset)
    # if not os.path.exists(path_output_tp_subset):
    #     os.makedirs(path_output_tp_subset)

    for path_image in glob.glob(os.path.join(path_folder_image, folder, "*{}".format(extension))):
        filename = getfilename(path_image)
        if filename in ground_truth.keys() and filename in prediction.keys():
            image = readImage(path_image)
            bb_gt = ground_truth[filename]
            bb_pred = prediction[filename]
            path_output_tp_subset = os.path.join(path_save, "TP", folder)
            create_dirs(path_output_tp_subset)
            image = drawBB(image, bb_gt, "red")
            image = drawBB(image, bb_pred, "blue")
            img2 = []
            for bb in bb_pred:
                img = crop_imagePIL(image, bb)
                img = resize_imagePIL(img, (256,256))
                img2.append(img)
            image = get_concate_h(image, img2)
            writeImage(image, os.path.join(path_output_tp_subset, getbasename(path_image)))
            print("TP", filename)
        else:
            if filename in prediction.keys():
                image = readImage(path_image)
                bb_pred = prediction[filename]
                path_output_fp_subset = os.path.join(path_save, "FP", folder)
                create_dirs(path_output_fp_subset)
                image = drawBB(image, bb_pred, "blue")
                img2 = []
                for bb in bb_pred:
                    img = crop_imagePIL(image, bb)
                    img = resize_imagePIL(img, (256,256))
                    img2.append(img)
                image = get_concate_h(image, img2)
                writeImage(image, os.path.join(path_output_fp_subset, getbasename(path_image)))
                print("FP", filename)

            if filename in ground_truth.keys():
                image = readImage(path_image)
                bb_gt = ground_truth[filename]
                path_output_fn_subset = os.path.join(path_save, "FN", folder)
                create_dirs(path_output_fn_subset)
                image = drawBB(image, bb_gt, "red")
                img2 = []
                for bb in bb_gt:
                    img = crop_imagePIL(image, bb)
                    img = resize_imagePIL(img, (256,256))
                    img2.append(img)
                image = get_concate_h(image, img2)
                writeImage(image, os.path.join(path_output_fn_subset, getbasename(path_image)))
                print("FN", filename)

if __name__ == "__main__":
    path_folder_image = r"D:\02_BME\002_NCKUH\Db_20210816_RGB"
    path = r"D:\02_BME\003_evaluation_Model\NCKUH"
    # path_groundtruth_w = select_file(path, title="choose ground truth file")
    # path_groundtruth_wo = select_file(path, title="choose ground truth 2 file")
    path_dir = select_directionary(path, title= "Choose prediction result folder")
    folders = ["Test_A", "Test_B", "Test_C", "Test_Me", "Test_S"]
    for folder in folders:
        path_prediction = os.path.join(path_dir, "result_{}.csv".format(folder))
        path_groundtruth_w = os.path.join(path, "000_GroundTruth", "ggo_{}_w.csv".format(folder))
        path_groundtruth_wo = os.path.join(path, "000_GroundTruth", "ggo_{}_wo.csv".format(folder))
        if not os.path.exists(path_groundtruth_w):
            path_groundtruth_w = os.path.join(path, "000_GroundTruth", "ggo_{}.csv".format(folder))
        if not os.path.exists(path_groundtruth_wo):
            path_groundtruth_wo = None
        
        draw_images(path_groundtruth_w, 
                    path_prediction, 
                    path_folder_image, 
                    path_groundtruth_wo)
        




    





