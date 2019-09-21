# encoding: utf-8
# Author: zTaylor

import os
import numpy as np
import tensorflow as tf
import cv2
import itertools

import config

# tf.enable_eager_execution()

class next_img:
    def __init__(self, predicted_bbox_path):
        self.traking_data_path = config.tracking_data_path

        self.video_path_list = []

        self.predicted_bbox_path = predicted_bbox_path

        self.build()


    def build(self):
        video_dir_list = os.listdir(self.traking_data_path)
        video_dir_list.sort()
        for dir_name in video_dir_list:
            self.video_path_list.append(os.path.join(self.traking_data_path, dir_name))

        self.f_reader = open(self.predicted_bbox_path, 'r')


    def gen(self):
        for video in self.video_path_list:
            img_name_list = os.listdir(os.path.join(video, "frames"))
            img_name_list.sort()
            img_num = len(img_name_list)

            bbox_path = os.path.join(video, "gt_bounding_box.txt")
            with open(bbox_path, "r") as f:
                init_gt_bbox = f.readlines()[0].strip().split(',')
            gt_bbox = [int(item) for item in init_gt_bbox]      # presume that coordinate format of bbox_file is "xmin, ymin, width, hight"

            predicted_bbox = os.path.join(video, "prediction.txt")
            with open(predicted_bbox, "w+") as f:
                i = 0
                exemplar = [self.img_processign(img_name_list[i], gt_bbox, os.path.join(video, "frames"))[64:64 + 127, 64:64 + 127, :].astype(np.float32)/255]
                f.write()

                i += 1
                while i <= img_num:
                    init_gt_bbox = f.readlines()[0].strip().split(',')      # todo: f.write() and f.readlines() is compatible?
                    gt_bbox = [int(item) for item in init_gt_bbox]
                    original_search = [self.img_processign(img_name_list[i], gt_bbox, os.path.join(video, "frames")).astype(np.float32)/255]
                    # search = [(cv2.imread(os.path.join(video, item))).astype(np.float32)/255 for item in img_name_list[i : i+self.batch_size]
                    f.write()
                    yield (exemplar, search)


    def img_processign(self, img_name, bndbox, data_path):
        """

        :param img_name:
        :param bndbox:      x_min, y_min, w, h
        :param data_path:
        :param size:
        :return:
        """

        img_path = os.path.join(data_path, img_name + ".JPEG")
        img = cv2.imread(img_path)  # type(img): ndarray
        img_h, img_w, img_c = img.shape
        # c_mean = []     # channel mean
        #
        # for i in range(img_c):
        #     c_mean.append(img[:, :, i].mean())

        img_mean = img.mean()

        # todo padding by channel (c_mean)
        if (bndbox[0] + bndbox[2] + 1 > img_w):
            img = np.pad(img, ((0, 0), (0, bndbox[0] + bndbox[2] + 1 - img_w), (0, 0),), "constant",
                         constant_values=img_mean)
        if (bndbox[1] + bndbox[3] + 1 > img_h):
            img = np.pad(img, ((0, bndbox[1] + bndbox[3] + 1 - img_h), (0, 0), (0, 0),), "constant",
                         constant_values=img_mean)
        if (bndbox[0] < 0):
            img = np.pad(img, ((0, 0), (-bndbox[0], 0), (0, 0),), "constant", constant_values=img_mean)
            bndbox[0] = 0
        if (bndbox[1] < 0):
            img = np.pad(img, ((0, 0), (-bndbox[1], 0), (0, 0),), "constant", constant_values=img_mean)
            bndbox[1] = 0

        cropped_img = img[bndbox[1]:bndbox[1] + bndbox[3] + 1, bndbox[0]:bndbox[0] + bndbox[2] + 1,
                      :]  # remember in cv2.img, first dim is "y" of bndbox
        resized_img = cv2.resize(src=cropped_img, dsize=(255, 255), interpolation=cv2.INTER_LINEAR)

        return resized_img


if __name__ == "__main__":
    batch = batch_loader()

