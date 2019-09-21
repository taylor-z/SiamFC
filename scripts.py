# encoding: utf-8
# Author: zTaylor

"""some common customized functions"""

import os

import config

processed_data = os.path.join(config.VID_new_path, "Data", "VID")
training_data = [os.path.join("train", item) for item in os.listdir(os.path.join(processed_data, "train"))]
validation_data = [os.path.join(processed_data, "val")]


# todo!!! 因为要记录已经处理的图片，所以不应该用函数，而应该构建一个类
# def get_batch(size, is_training=True):
#     # todo
#
#     if is_training:
#         c = training_data
#     else:
#         c = validation_data
#
#     for item in c:
#         if is_training:
#             path = os.path.join(processed_data, item)
#         else:
#             path = os.path.join(processed_data, item)
#
#         list = os.listdir(path)
#         list.sort()
#         for i, video_name in enumerate(list):
#             video_path = os.path.join(path, video_name)
#             for ins in os.listdir(video_path):
#                 ins_path = os.path.join(video_path, ins)
#                 print(ins_path)
#         break
#
#     return None

def get_y_true(size):
    # todo
    return None

# if __name__ == "__main__":
#     get_batch(1)