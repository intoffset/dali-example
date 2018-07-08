#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import inputs_dali, inputs_tf
from util import show_standardized_images

import tensorflow as tf
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecord', None, """path to tfrecord""")
tf.app.flags.DEFINE_boolean('dali', None, """Whether to use NVIDIA-dali""")

TFRECORD2IDX_SCRIPT = "tfrecord2idx"
DEVICES = 1
ITERATIONS = 32
BURNIN_STEPS = 16


def main(argv=None):
    batch_size = 16

    if FLAGS.dali:
        images, labels = inputs_dali(batch_size, devices=DEVICES, tfrecord=FLAGS.tfrecord)
    else:
        images, labels = inputs_tf(batch_size, devices=DEVICES, tfrecord=FLAGS.tfrecord)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        all_img_per_sec = []
        total_batch_size = batch_size * DEVICES

        for i in range(ITERATIONS):
            start_time = time.time()

            # The actual run with our dali_tf tensors
            images_val, labels_val = sess.run([images, labels])

            elapsed_time = time.time() - start_time
            img_per_sec = total_batch_size / elapsed_time
            if i > BURNIN_STEPS:
                all_img_per_sec.append(img_per_sec)
                print("\t%7.1f img/s" % img_per_sec)

        print("Total average %7.1f img/s" % (sum(all_img_per_sec) / len(all_img_per_sec)))

    print('len(images_val) = {}'.format(len(images_val)))
    print('len(labels_val) = {}'.format(len(labels_val)))
    print('images_val[0].shape = {}'.format(images_val[0].shape))
    print('labels_val[0].shape = {}'.format(labels_val[0].shape))
    print('images_val[0].dtype = {}'.format(images_val[0].dtype))
    print('labels_val[0].dtype = {}'.format(labels_val[0].dtype))
    print('np.mean(images_val[0]) = {}'.format(np.mean(images_val[0])))
    print('np.std(images_val[0]) = {}'.format(np.std(images_val[0])))

    show_standardized_images(images_val[0].astype(np.uint8), labels_val[0], batch_size)


if __name__ == '__main__':
    tf.app.run()