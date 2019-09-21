# encoding: utf-8
# Author: zTaylor

"""configuration parameters"""
import os
import numpy as np


# seed
seed = 955


# hyper-parameter
lr = 1e-2       # todo: epochs and decreasing lr
training_steps = int(5e4)
batch_size = 8
is_batch_norm = True


# input image parameters
z_size = 127
x_size = 255
channel = 3

# z_batch = [batch_size, z_size, z_size, channel]     # to#do change to [1, .., .., ..]
z_batch = [1, z_size, z_size, channel]
x_batch = [batch_size, x_size, x_size, channel]

w_num = 17                                            # output feature size (window num): 17 * 17
# y_true_batch = [batch_size, w_num, w_num, 1]        # to#do change to [8, 17, 17]
y_true_batch = [batch_size, w_num, w_num]

# feature_channel = 1


# saving paras
show_interval         = 100
summary_interval      = 100
model_saving_interval = 2500
max_saving_nums       = 10


# path
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
VID_path = os.path.join(ROOT_PATH, "data", "ILSVRC")
VID_new_path = os.path.join(ROOT_PATH, "data", "ILSVRC_new")        # directory which is used for saving processed VID
tracking_data_path = os.path.join(ROOT_PATH, "data", "tracking_data")

parameters_saving_path = os.path.join(ROOT_PATH, "saved_file", "summary_parameters")
model_saving_path = os.path.join(ROOT_PATH, "saved_file", "trained_model")
restore_path = os.path.dirname(model_saving_path)


# network backbone(conv model)
# conv_arch = [
#     {"kernel_size": 11, "in_channel": 3,   "out_channel": 96,  "stride": 2, "is_pooling": True, "pooling_size": 3, "pooling_stride": 2, "is_batch_normal": True, "is_ReLU": True, "is_padding": "valid"},
#     {"kernel_size": 5,  "in_channel": 96,  "out_channel": 256, "stride": 1, "is_pooling": True, "pooling_size": 3, "pooling_stride": 2, "is_batch_normal": True, "is_ReLU": True, "is_padding": "valid"},
#     {"kernel_size": 3,  "in_channel": 256, "out_channel": 384, "stride": 1, "is_pooling": False, "is_batch_normal": True, "is_ReLU": True, "is_padding": "valid"},
#     {"kernel_size": 3,  "in_channel": 384, "out_channel": 192, "stride": 1, "is_pooling": False, "is_batch_normal": True, "is_ReLU": True, "is_padding": "valid"},
#     {"kernel_size": 3,  "in_channel": 192, "out_channel": 256, "stride": 1, "is_pooling": False, "is_batch_normal": True, "is_ReLU": True, "is_padding": "valid"},
# ]

conv_arch = [
    {"kernel_size": 11, "in_channel": 3,   "out_channel": 96,  "stride": [1, 2, 2, 1], "is_pooling": True,  "is_batch_normal": True,  "is_ReLU": True,  "is_padding": "VALID",
     "pooling_size": [1, 3, 3, 1], "pooling_stride": [1, 2, 2, 1]},

    {"kernel_size": 5,  "in_channel": 96,  "out_channel": 256, "stride": [1, 1, 1, 1], "is_pooling": True,  "is_batch_normal": True,  "is_ReLU": True,  "is_padding": "VALID",
     "pooling_size": [1, 3, 3, 1], "pooling_stride": [1, 2, 2, 1]},

    {"kernel_size": 3,  "in_channel": 256, "out_channel": 384, "stride": [1, 1, 1, 1], "is_pooling": False, "is_batch_normal": True,  "is_ReLU": True,  "is_padding": "VALID"},

    {"kernel_size": 3,  "in_channel": 384, "out_channel": 192, "stride": [1, 1, 1, 1], "is_pooling": False, "is_batch_normal": True,  "is_ReLU": True,  "is_padding": "VALID"},

    {"kernel_size": 3,  "in_channel": 192, "out_channel": 256, "stride": [1, 1, 1, 1], "is_pooling": False, "is_batch_normal": True,  "is_ReLU": False, "is_padding": "VALID"},
]