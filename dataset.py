import torch
import pandas as pd
import os
import xml.etree.ElementTree as ET
from PIL import Image
import csv
import shutil


# Preprocessing original dataset from PASCAL VOC 2007 trainval, 2012 trainval
def scale_box(img_width, img_height, b):
    x = ((b[0] + b[1]) / 2.0) / img_width
    y = ((b[2] + b[3]) / 2.0) / img_height
    w = (b[1] - b[0]) / img_width
    h = (b[3] - b[2]) / img_height
    return x, y, w, h


# From official website, the 20 classes are
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
if not os.path.exists("VOCdevkit/tar/"):
    os.makedirs("VOCdevkit/tar/", exist_ok=True)
    for year, fold in [('2007', "trainval"), ('2012', 'trainval')]:
        annotation_dir = 'VOCdevkit/VOC%s/Annotations/' % year
        file_list = os.listdir(annotation_dir)
        for filename in file_list:
            tar_file_dir = 'VOCdevkit/tar/%s.txt' % filename.split(".")[0]
            file_path = os.path.join(annotation_dir, filename)
            with open(file_path) as file:
                tree = ET.parse(file)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    if cls not in classes or int(difficult) == 1:
                        continue
                    cls_id = classes.index(cls) + 1     # 1-indexed classes ids
                    box = obj.find('bndbox')
                    bbox = (float(box.find('xmin').text),
                            float(box.find('xmax').text),
                            float(box.find('ymin').text),
                            float(box.find('ymax').text))
                    bbox = scale_box(width, height, bbox)
                    with open(tar_file_dir, 'a') as out_file:
                        out_file.write(str(cls_id) + " " + " ".join([str(_) for _ in bbox]) + "\n")

    # generate .csv file containing [imgname.jpg imgname.txt] mappings for Dataset usage
    for year, fold in [('2007', "trainval"), ('2012', 'trainval')]:
        lines = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, fold)).readlines()
        with open('VOCdevkit/tar/%s.csv' % fold, "w") as trainval:
            for line in lines:
                img_name = line.replace("\n", "")
                txt_name = img_name.replace(".jpg", ".txt")
                writer = csv.writer(trainval)
                writer.writerow([img_name, txt_name])

if not os.path.exists("VOCdevkit/img/"):
    os.makedirs("VOCdevkit/img/")
    for year in ['2007', '2012']:
        img_list = os.listdir("VOCdevkit/VOC%s/JPEGImages/" % year)
        for img in img_list:
            source_img_path = os.path.join("VOCdevkit/VOC%s/JPEGImage/" % year, img)
            target_img_path = "VOCdevkit/img/"
            shutil.copy(source_img_path, target_img_path)


# Construct Pytorch custom dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv, transforms=None, img_dir="VOCdevkit/img/", tar_dir="VOCdevkit/tar/"):
        self.annotations = pd.read_csv(csv)
        self.tar_path = tar_dir
        self.img_path = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        txt_path = os.path.join(self.tar_path, self.annotations.iloc[item, 1])
        bboxes = []
        with open(txt_path) as t:
            for line in t.readlines():
                cls, x, y, w, h = [float(x) if float(x) != int(float(x)) else int(x)
                                   for x in line.replace("\n", "").split()]
                bboxes.append([cls, x, y, w, h])

        img_path = os.path.join(self.img_path, self.annotations.iloc[item, 0])
        img = Image.open(img_path)
        tar = torch.zeros((7, 7, 25))
        if self.transforms:
            img, bboxes = self.transforms(img, bboxes)
        for bbox in bboxes:
            i, j = int(7 * bbox[2]), int(7 * bbox[1])
            x_cell, y_cell = bbox[1] * 7 - j, bbox[2] * 7 - i
            w_cell, h_cell = bbox[3] * 7, bbox[4] * 7
            if tar[i, j, 20] == 0:
                tar[i, j, 20] = 1
                tar[i, j, 21:25] = torch.Tensor([x_cell, y_cell, w_cell, h_cell])
                tar[i, j, bbox[0] - 1] = 1
        # tar contains cell-relative coordiantes
        return img, tar
