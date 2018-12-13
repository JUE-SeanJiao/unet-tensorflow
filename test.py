# coding:utf-8

import os
import cv2
import glob
import tensorflow as tf
import numpy as np
import argparse
from math import *
import time
import matplotlib.pyplot as plt

h_in = 224 # 704 768
w_in = 224 # 704 1024
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
input_path = './test_data/raw'
output_path = './test_data/pred'
label_path = './test_data/label'
label_view_path = './test_data/label_view'

if not os.path.exists(output_path):
    os.makedirs(output_path)

def load_model():
    file_meta = './model/model_final.ckpt.meta'
    file_ckpt = './model/model_final.ckpt'
    saver = tf.train.import_meta_graph(file_meta)

    sess = tf.InteractiveSession()
    saver.restore(sess, file_ckpt)
    return sess


def images_path(input_path):
    images_path_list=[os.path.join(input_path, i) for i in os.listdir(input_path)]
    return images_path_list


def read_image(image_path, gray=False):

    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def visualize_labels(labels):
    labels = tf.cast(labels[..., 0], tf.int32)
    table = tf.constant([[0, 0, 0], [128, 64, 128], [64, 128, 255], [255, 64, 128], [64, 255, 64]], tf.int32)
    out = tf.nn.embedding_lookup(table, labels)
    out = tf.cast(out, tf.uint8)
    return out

def main():
    sess = load_model()
    X, mode = tf.get_collection('inputs')
    pred = tf.get_collection('outputs')[0]
    # pred = tf.get_collection('pred')[0]

    images_path_list = images_path(input_path)

    for i, path in enumerate(images_path_list):
        # print('==================== Test =====================')
        image = read_image(path)
        # label = read_image(path.replace('raw', 'label'), gray=True)
        # print(image.shape)
        h_, w_ = image.shape[:2]

        image = cv2.resize(image, (w_in, h_in))
        # sess=tf.InteractiveSession()

        print('Test image {}'.format(i+1))
        print(path.split('/')[3])
        start_time = time.time()
        label_pred = sess.run(pred, feed_dict={X: np.expand_dims(image, 0), mode: False})
        print("--- {}s seconds ---".format((time.time() - start_time)))
        merged = np.squeeze(label_pred)
        merged = cv2.resize(merged, (w_, h_), interpolation=cv2.INTER_NEAREST)
        merged = merged.astype(int) * 255/4
        # save_name = os.path.join(flags.disk_save_dir, flags.disk_save_name)
        if os.path.exists(output_path)==False:
            os.mkdir(output_path)
        cv2.imwrite(output_path+'/'+path.split('/')[3], merged)

        # np.set_printoptions(threshold=np.inf)
        # print(merged.shape)
        # print(merged)

        # cv2.imwrite(label_view_path+'/'+path.split('/')[3], label.astype(int) * 255/4)


if __name__ == '__main__':
    main()






