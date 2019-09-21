# encoding: utf-8
# Author: zTaylor

import os
import sys
import numpy as np
import math

import time
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config


def img_process(img_name, id, bndbox, data_path, save_path):
    aaa = os.path.join(save_path, id)
    if not(os.path.isdir(aaa)):
        os.makedirs(aaa)        # todo -p ???\

    img_path = os.path.join(data_path, img_name + ".JPEG")
    img_save_path = os.path.join(aaa, img_name + ".JPEG")
    img = cv2.imread(img_path)    # type(img): ndarray
    img_h, img_w, img_c = img.shape
    # c_mean = []     # channel mean
    #
    # for i in range(img_c):
    #     c_mean.append(img[:, :, i].mean())

    img_mean = img.mean()
    # print(img_mean)

    # todo padding by channel (c_mean)
    if(bndbox[0] + bndbox[2] + 1 > img_w):
        img = np.pad(img, ((0, 0), (0, bndbox[0] + bndbox[2] + 1 - img_w), (0, 0),), "constant", constant_values=img_mean)
    if(bndbox[1] + bndbox[3] + 1 > img_h):
        img = np.pad(img, ((0, bndbox[1] + bndbox[3] + 1 - img_h), (0, 0), (0, 0),), "constant", constant_values=img_mean)
    if(bndbox[0] < 0):
        img = np.pad(img, ((0, 0), (-bndbox[0], 0), (0, 0),), "constant", constant_values=img_mean)
        bndbox[0] = 0
    if(bndbox[1] < 0):
        img = np.pad(img, ((0, 0), (-bndbox[1], 0), (0, 0),), "constant", constant_values=img_mean)
        bndbox[1] = 0

    cropped_img = img[bndbox[1]:bndbox[1]+bndbox[3]+1, bndbox[0]:bndbox[0]+bndbox[2]+1, :]      # remember in cv2.img, first dim is "y" of bndbox
    resized_img = cv2.resize(src=cropped_img, dsize=(255, 255), interpolation=cv2.INTER_LINEAR)

    if not (os.path.exists(img_save_path)):
        # print("writing img to {}".format(img_save_path))
        cv2.imwrite(img_save_path, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])       # todo cropped_img 比手算的 w 和 h 多1

    return bndbox   # todo change original bndbox


def vid_process(anno_path, data_path, save_path):
    """
    processing VID dataset
    :param anno_path: annotation files path for one specific video series
    :param data_path: data path for one specific video series
    :param save_path: path that saving processed frames
    :return: None
    """
    bounding_box = {}
    new_bounding_box = {}
    list = os.listdir(anno_path)
    list.sort()
    for i, xml in enumerate(list):
        tree = ET.parse(os.path.join(anno_path, xml))
        root = tree.getroot()
        img_name = root.find("filename").text
        object_set = root.findall("object")
        for object in object_set:
            id = object.find("trackid").text        # type(id): str
            if not id in bounding_box:
                bounding_box[id] = []
                new_bounding_box[id] = []
            bndbox = object.find("bndbox")
            xmax = int(bndbox.find("xmax").text)
            xmin = int(bndbox.find("xmin").text)
            ymax = int(bndbox.find("ymax").text)
            ymin = int(bndbox.find("ymin").text)

            w = xmax - xmin
            h = ymax - ymin
            crop = [xmin, ymin, w, h]
            bounding_box[id].append([i, crop])

            p = (w + h) / 4
            xmin = int(xmin - p)        # todo 取整操作
            w = math.ceil(w + 2 * p)    #
            ymin = int(ymin - p)        #
            h = math.ceil(h + 2 * p)    #
            crop = [xmin, ymin, w, h]
            new_bounding_box[id].append([i, crop])      # "i" is for indicating which frame the bounding_box is dividing to

            # todo !!!!!!!!!!!!!!
            # c = (w + h) / 4 * 2
            # ss = np.sqrt((w+c) * (w+h))
            # context = ss / 2
            # xmin = int(xmin - context)
            # w = math.ceil(w + 2 * context)
            # ymin = int(ymin - context)
            # h = math.ceil(h + 2 * context)

            img_process(img_name, id, crop, data_path, save_path)


if __name__ == "__main__":
    anno = os.path.join(config.VID_path, "Annotations", "VID")
    data = os.path.join(config.VID_path, "Data", "VID")
    processed_data = os.path.join(config.VID_new_path, "Data", "VID")

    # train_list = [os.path.join(data, "train", item) for item in os.listdir(os.path.join(data, "train"))]
    # val_list = [os.path.join(data, "val")]
    class_dir_list = [os.path.join("train", item) for item in os.listdir(os.path.join(data, "train"))]
    class_dir_list.sort()
    class_dir_list.append("val")

    start_time = time.time()

    for item in class_dir_list:
        # if(item != "train/ILSVRC2015_VID_train_0003"):
        #     continue
        leaf_dir_list = os.listdir(os.path.join(anno, item))
        leaf_dir_list.sort()

        print("processing video series:  {}".format(item))

        s = time.time()
        for i in tqdm(range(len(leaf_dir_list))):
            leaf_dir = leaf_dir_list[i]
            dir_relative_path = os.path.join(item, leaf_dir)
            # print(dir_relative_path)
            anno_dir_abs_path = os.path.join(anno, dir_relative_path)
            data_dir_abs_path = os.path.join(data, dir_relative_path)
            processed_data_dir_abs_path = os.path.join(processed_data, dir_relative_path)
            if(os.path.isdir(processed_data_dir_abs_path)):
                continue

            vid_process(anno_dir_abs_path, data_dir_abs_path, processed_data_dir_abs_path)

        e = time.time()
        m, s = divmod(e - s, 60)
        h, m = divmod(m, 60)
        print("Used time:", h, ':', m, ':', int(s), sep='')
        # break

    end_time = time.time()

    m, s = divmod(end_time-start_time, 60)
    h, m = divmod(m, 60)
    print("TOTAL time:", h, ':', m, ':', int(s), sep='')