#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import inputs_dali

import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecord', None, """path to tfrecord""")

TFRECORD2IDX_SCRIPT = "tfrecord2idx"
DEVICES = 1
ITERATIONS = 32
BURNIN_STEPS = 16


def main(argv=None):
    batch_size = 16

    images, labels = inputs_dali(batch_size, devices=DEVICES, tfrecord=FLAGS.tfrecord)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        all_img_per_sec = []
        total_batch_size = batch_size * DEVICES

        for i in range(ITERATIONS):
            start_time = time.time()

            # The actual run with our dali_tf tensors
            images_val, labels_val = sess.run([images, labels])
            print('len(images) = {}'.format(len(images)))
            print('len(labels) = {}'.format(len(labels)))
            print('images[0].shape = {}'.format(images[0].shape))
            print('labels[0].shape = {}'.format(labels[0].shape))

            elapsed_time = time.time() - start_time
            img_per_sec = total_batch_size / elapsed_time
            if i > BURNIN_STEPS:
                all_img_per_sec.append(img_per_sec)
                print("\t%7.1f img/s" % img_per_sec)

        print("Total average %7.1f img/s" % (sum(all_img_per_sec) / len(all_img_per_sec)))

if __name__ == '__main__':
    tf.app.run()