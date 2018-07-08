#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

image_dir = "../dali/examples/images/"
batch_size = 8


class nvJPEGPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(nvJPEGPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=image_dir, random_shuffle=True, initial_fill=21)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        # images are on the GPU
        return images, labels


def show_images(image_batch):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(32, (32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))

    plt.show()


def main(argv=None):
    # nvJPEG pipeline example
    pipe = nvJPEGPipeline(batch_size, 1, 0)
    pipe.build()
    pipe_out = pipe.run()
    images, labels = pipe_out
    print("Images type is: " + str(type(images)))
    print("Labels type is: " + str(type(labels)))
    print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
    print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))

    print("Label is :" + str(np.array(labels.as_tensor())))

    show_images(images.asCPU())

if __name__ == '__main__':
    main()
