# -*- coding: utf-8 -*-

import os
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf

FLAGS = tf.app.flags.FLAGS


class TFRecordPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, tfrecord, tfrecord_idx):
        super(TFRecordPipeline, self).__init__(batch_size,
                                         num_threads,
                                         device_id)
        self.input = ops.TFRecordReader(
            path=tfrecord,
            index_path=tfrecord_idx,
            features={"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                      'image/class/label':       tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                      'image/class/text':        tfrec.FixedLenFeature([ ], tfrec.string, ''),
                      # 'image/object/bbox/xmin':  tfrec.VarLenFeature(tfrec.float32, 0.0),
                      # 'image/object/bbox/ymin':  tfrec.VarLenFeature(tfrec.float32, 0.0),
                      # 'image/object/bbox/xmax':  tfrec.VarLenFeature(tfrec.float32, 0.0),
                      # 'image/object/bbox/ymax':  tfrec.VarLenFeature(tfrec.float32, 0.0)
                      })
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu", resize_a = 256, resize_b = 256)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            crop = (224, 224),
                                            image_type = types.RGB,
                                            mean = [0., 0., 0.],
                                            std = [1., 1., 1.])
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.iter = 0

    def define_graph(self):
        inputs = self.input()
        images = self.decode(inputs["image/encoded"])
        resized_images = self.resize(images)
        output = self.cmnp(resized_images, crop_pos_x = self.uniform(),
                           crop_pos_y = self.uniform())
        return (output, inputs["image/class/label"].gpu())

    def iter_setup(self):
        pass


def inputs_dali(batch_size, devices, tfrecord):
    tfrecord_idx = os.path.splitext(tfrecord)[0] + '.idx'
    tfrecord2idx_script = "tfrecord2idx"

    if not os.path.isfile(tfrecord_idx):
        call([tfrecord2idx_script, tfrecord, tfrecord_idx])

    pipes = [
        TFRecordPipeline(
            batch_size=batch_size, num_threads=2, device_id=device_id,
            tfrecord=FLAGS.tfrecord, tfrecord_idx=tfrecord_idx) for device_id
        in range(devices)]

    serialized_pipes = [pipe.serialize() for pipe in pipes]
    del pipes

    daliop = dali_tf.DALIIterator()

    images = []
    labels = []
    for d in range(devices):
        with tf.device('/gpu:%i' % d):
            image, label = daliop(serialized_pipeline=serialized_pipes[d],
                                  batch_size=batch_size,
                                  height=224,
                                  width=224,
                                  device_id=d)
            images.append(image)
            labels.append(label)

    return images, labels


