from typing import List, Union

import tensorflow as tf

from domain_adaptation.datasets.utils import random_apply


class SwaVDataset:

    def __init__(self,
                 dataset_or_path: Union[str, tf.data.Dataset, None] = None,
                 size_crops: List[int] = [224],
                 nmb_crops: List[int] = [2],
                 min_scale_crops: List[int] = [0.14],
                 max_scale_crops: List[int] = [1],
                 override: bool = True):
        try:
            assert len(size_crops) == len(nmb_crops)
            assert len(min_scale_crops) == len(nmb_crops)
            assert len(max_scale_crops) == len(min_scale_crops)
        except AssertionError:
            raise Exception("Crops parameters should have equal length")
        if not dataset_or_path:
            raise Exception("either a tf.Dataset or path should be passed")
        else:
            if isinstance(dataset_or_path, str):
                self.dataset = tf.data.Dataset.list_files(dataset_or_path)
            elif isinstance(dataset_or_path, tf.data.Dataset):
                self.dataset = dataset_or_path
            else:
                raise Exception("dataset_or_path argument should either be a"
                                " tf.Dataset or a string")
        self.size_crops, self.nmb_crops = size_crops, nmb_crops
        self.min_scale_crops, self.max_scale_crops = (min_scale_crops,
                                                      max_scale_crops)
        self.init_dataset(override)
        # self.dataset = dataset.map(self.pipeline)

    # @staticmethod
    # @tf.function
    # def get_labels(self, file_path):
    #     label = tf.strings.split(file_path, os.sep)[-2]
    #     return tf.io.read_file(file_path), label
    def init_dataset(self, override):
        dataloaders = tuple()
        for i in range(len(self.size_crops)):

            def main_func(x):
                return self.pipeline(x, self.size_crops[i],
                                     self.min_scale_crops[i],
                                     self.max_scale_crops[i])

            for _ in range(self.nmb_crops[i]):
                loader = self.dataset.map(
                    main_func,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataloaders += (loader,)
        if override:
            self.dataset = tf.data.Dataset.zip(dataloaders)
        else:
            self.dataset_swaved = tf.data.Dataset.zip(dataloaders)

    def pipeline(self, img, crop_size, min_scale, max_scale):
        img = SwaVDataset.normalizing(img)
        img = SwaVDataset.random_crop_resize(img=img,
                                             crop_size=crop_size,
                                             min_scale=min_scale,
                                             max_scale=max_scale)
        img = SwaVDataset.random_flip(img=img)
        img = SwaVDataset.color_jitter(img=img, s=1.0)
        img = SwaVDataset.gray_scale(img=img)
        if len(img.shape) == 3:
            img = tf.expand_dims(img, axis=0)
        img = SwaVDataset.gaussian_blur(img=img, kernel_size=23, sigma=None)
        return tf.squeeze(img, axis=0)

    @staticmethod
    @tf.function
    @random_apply(p=0.5)
    def gaussian_blur(img, kernel_size=23, sigma=None):
        if not sigma:
            sigma = tf.random.uniform((1,)) * 1.9 + 0.1

        def gauss_kernel(channels, kernel_size, sigma):
            ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
            xx, yy = tf.meshgrid(ax, ax)
            kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
            kernel /= tf.reduce_sum(kernel)
            kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
            return kernel

        gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
        gaussian_kernel = gaussian_kernel[..., tf.newaxis]

        return tf.nn.depthwise_conv2d(img,
                                      gaussian_kernel, [1, 1, 1, 1],
                                      padding='SAME',
                                      data_format='NHWC')

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
                                       upper=1 + 0.8 * s)
        img = tf.image.random_hue(img, max_delta=0.2 * s)
        img = tf.clip_by_value(img, 0, 1)
        return img

    @staticmethod
    @tf.function
    @random_apply(p=0.2)
    def gray_scale(img):
        return tf.tile(tf.image.rgb_to_grayscale(img), [1, 1, 3])

    @staticmethod
    @tf.function
    def random_flip(img):
        return tf.image.random_flip_left_right(img)

    @staticmethod
    @tf.function
    def random_crop_resize(img, crop_size, min_scale, max_scale):
        shape = img.shape[1]
        size = tf.random.uniform((1,),
                                 minval=min_scale * shape,
                                 maxval=max_scale * shape,
                                 dtype=tf.float32)[0]
        if len(img.shape) == 4:
            img = tf.squeeze(img, axis=0)
        crop = tf.image.random_crop(img, (size, size, img.shape[2]))
        crop_resize = tf.image.resize(crop, (crop_size, crop_size))
        return crop_resize

    @staticmethod
    @tf.function
    def normalizing(img):
        return tf.cast(img / 255, tf.float32)
