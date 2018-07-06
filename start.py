#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import  os
import  fnmatch

import numpy as np
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline
from timeit import default_timer as timer

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


import  nvidia.dali.ops as ops
import nvidia.dali.types as types

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir', None, """path to input directory""")

image_dir = "../dali/examples/images"
batch_size = 8


class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=FLAGS.image_dir)
        self.decode = ops.HostDecoder(output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)


class ShuffledSimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(ShuffledSimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root=FLAGS.image_dir, random_shuffle = True, initial_fill = 21)
        self.decode = ops.HostDecoder(output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)


class nvJPEGPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(nvJPEGPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root=FLAGS.image_dir, random_shuffle = True, initial_fill = 21)
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        # images are on the GPU
        return (images, labels)


def show_images(image_batch):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))

    plt.show()


def main(argv=None):
    # Simple pipeline example
    pipe = ShuffledSimplePipeline(batch_size, 1, 0)
    pipe.build()
    pipe_out = pipe.run()
    print(pipe_out)
    images, labels = pipe_out
    print("Images type is: " + str(type(images)))
    print("Labels type is: " + str(type(labels)))
    print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
    print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))

    # is_dense_tensor() = False なTensorListはas_tensorが使えない。atメソッドで要素を一つずつ取り出すしかない
    print(type(images.at(0)))
    labels_tensor = labels.as_tensor()

    print(labels_tensor.shape())
    print(np.array(labels_tensor))

    show_images(images)

    # nvJPEG pipeline example
    pipe = nvJPEGPipeline(batch_size, 1, 0)
    pipe.build()
    pipe_out = pipe.run()
    print(pipe_out)
    images, labels = pipe_out
    print("Images type is: " + str(type(images)))
    print("Labels type is: " + str(type(labels)))
    print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
    print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))

    images, labels = pipe_out
    show_images(images.asCPU())

    test_batch_size = 64

    def speedtest(pipeclass, batch, n_threads):
        pipe = pipeclass(batch, n_threads, 0)
        pipe.build()
        # warmup
        for i in range(5):
            pipe.run()
        # test
        n_test = 20
        t_start = timer()
        for i in range(n_test):
            pipe.run()
        t = timer() - t_start
        print("Speed: {} imgs/s".format((n_test * batch) / t))

    speedtest(ShuffledSimplePipeline, test_batch_size, 4)
    speedtest(nvJPEGPipeline, test_batch_size, 4)


if __name__ == '__main__':
    tf.app.run()
