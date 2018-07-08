
from __future__ import division
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def show_images(image_batch, labels, batch_size):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        ascii = labels.at(j)
        plt.title("".join([chr(item) for item in ascii]))
        img_chw = image_batch.at(j)
        img_hwc = np.transpose(img_chw, (1,2,0))/255.0
        plt.imshow(img_hwc)

    plt.show()


def show_standardized_images(image_batch, labels, batch_size):
    print(image_batch.shape)
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize = (32, (32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        ascii = labels[j]
        plt.title("".join([chr(item) for item in ascii]))
        img_hwc = image_batch[j]
        print(img_hwc.shape)
        print(image_batch.shape)
        plt.imshow(img_hwc)

    plt.show()
