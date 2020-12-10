import os

import tensorflow as tf

from domain_adaptation.datasets.utils import random_apply


class SwaVDataset:

    def __init__(self, path: str):
        dataset = tf.data.Dataset.list_files(path)
        assert isinstance(dataset, tf.data.Dataset)
        self.dataset = dataset.map(self.pipeline)

    @staticmethod
    @tf.function
    def get_labels(self, file_path):
        label = tf.strings.split(file_path, os.sep)[-2]
        return tf.io.read_file(file_path), label

    @staticmethod
    @tf.function
    @random_apply(p=0.5)
    def gaussian_blur(self, img, kernel_size=23, sigma=None):

        def gauss_kernel(channels, kernel_size, sigma):
            ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
            xx, yy = tf.meshgrid(ax, ax)
            kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
            kernel /= tf.reduce_sum(kernel)
            kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
            return kernel

        gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
        gaussian_kernel = gaussian_kernel[..., tf.new_axis]

        return tf.nn.depthwise_conv2d(img,
                                      gaussian_kernel, [1, 1, 1, 1],
                                      padding='SAME',
                                      data_format='NHWC')

    # @tf.function
    # def random_gaussian_blur(self, img):
    #     do_it = tf.random.uniform((1,)) > 0.5
    #     if not do_it:
    #         return img
    #     sigma = tf.random.uniform((1,)) * 1.9 + 0.1
    #     return self.gaussian_blur(img, kernel_size=23, sigma=sigma)

    @staticmethod
    @tf.function
    @random_apply(p=0.8)
    def color_jitter(img, s=1.0):
        img = tf.image.random_saturation(img,
                                         lower=1 - 0.8 * s,
                                         upper=1 + 0.8 * s)
        img = tf.image.random_brightness(img, max_delta=0.8 * s)
        img = tf.image.random_contrast(img,
                                       lower=1 - 0.8 * s,
                                       higher=1 + 0.8 * s)
        img = tf.image.random_hue(img, max_delta=0.2 * s)
        img = tf.clip_by_value(img, 0, 1)
        return img

    @staticmethod
    @tf.function
    @random_apply(p=0.2)
    def gray_scale(img):
        return tf.image.rgb_to_grayscale(img)

    @staticmethod
    @tf.function
    def random_flip(img):
        return tf.image.random_flip_left_right(img)
