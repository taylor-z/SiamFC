# encoding: utf-8
# Author: zTaylor

import os
import numpy as np
import tensorflow as tf
import cv2
import itertools

import config

# tf.enable_eager_execution()

class batch_loader:
    def __init__(self, is_training=True):
        self.is_training = True
        self.batch_size = config.batch_size

        self.processed_data = os.path.join(config.VID_new_path, "Data", "VID")

        if self.is_training:
            t = os.listdir(os.path.join(self.processed_data, "train"))
            t.sort()
            self.class_dir_list = [os.path.join("train", item)
                                   for item in t]
            self.class_dir_list.sort()
        else:
            self.class_dir_list = [os.path.join(self.processed_data, "val")]

        self.video_list = []
        self.build()


    def build(self):
        for item in self.class_dir_list:
            video_path = os.path.join(self.processed_data, item)
            lead_dir_list = os.listdir(video_path)
            lead_dir_list.sort()
            for video in lead_dir_list:
                # self.video_list.append(os.path.join(path, video))
                fragment_path = os.path.join(video_path, video)
                fragment_list = os.listdir(fragment_path)
                fragment_list.sort()
                for frag in fragment_list:
                    self.video_list.append(os.path.join(fragment_path, frag))
        # print(len(self.video_list))

        # todo y_true_batch
        self.dataset = tf.data.Dataset.from_generator(generator=self.gen,
                                                      output_types=(tf.float32, tf.float32),
                                                      output_shapes=(tf.TensorShape(config.z_batch),
                                                                     tf.TensorShape(config.x_batch)))
        # output_shapes=(tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None, None])))
        self.iterator = self.dataset.make_initializable_iterator()
        self.exemplar_batch, self.search_batch = self.iterator.get_next()


    # def get_one_batch(self):
    #     # to#do exemplar_z 没有提前设计   # gen()内使exemplar重复了8次(self.batch_size)
    #     for exemplar, search in self.dataset:
    #         # print(type(exemplar.shape), "  ", type(search.shape))
    #         # print(exemplar.shape, "  ", search.shape)
    #         return (exemplar, search)       # todo 每次调用是否会继续

    def get_one_batch(self, sess):
        return sess.run(self.exemplar_batch, self.search_batch)
    # sess.run(self.iterator.initializer)


    def gen(self):
        # for i in itertools.count(1):
        #     yield (None)
        for frag in self.video_list:
            imgs_list = os.listdir(frag)
            imgs_list.sort()
            imgs_num = len(imgs_list)
            i = 0

            # while i + 9 <= imgs_num:
            #     exemplar = [cv2.imread(os.path.join(frag, imgs_list[i]))]
            #     search = [cv2.imread(os.path.join(frag, item)) for item in imgs_list[i+1 : i+9]]
            #     i += 9
            #     yield (exemplar, search)

            # todo selecting initial exemplar every batch(every img can be exemplar)
            # todo 上述情况只需要再套一层循环即可，待查验效果
            # todo 提前cropping好exemplar???
            # exemplar = [cv2.imread(os.path.join(frag, imgs_list[i]))] * self.batch_size                            # to#do 在此处crop,构造127 * 127
            # exemplar = [cv2.imread(os.path.join(frag, imgs_list[i]))[64:64+127, 64:64+127, :]] * self.batch_size   # to#do change to sahpe=[1, .., .., .. ]
            exemplar = [(cv2.imread(os.path.join(frag, imgs_list[i]))[64:64 + 127, 64:64 + 127, :]).astype(np.float32)/255]

            i += 1
            while i + self.batch_size <= imgs_num:
                search = [(cv2.imread(os.path.join(frag, item))).astype(np.float32)/255 for item in imgs_list[i : i+self.batch_size]]    # todo search_img is selected randomly
                i += self.batch_size
                yield (exemplar, search)
            # break




if __name__ == "__main__":
    batch = batch_loader(True)
#     with tf.Session() as sess:
#         while(1):
#             e, s = batch.get_one_batch()
#             print(sess.run(e, s))

