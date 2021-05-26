import unittest
import pathlib

import tensorflow as tf
from tensorflow import test

from domain_adaptation.datasets.SwaVDataset import SwaVDataset
from domain_adaptation.datasets import utils


def process_path(file_path):
    return utils.process_path(file_path,
                              width=180,
                              height=180,
                              channels=3,
                              with_label=False)


class TestSwaVDataset(test.TestCase):
    flowers_root = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/'
        'example_images/flower_photos.tgz',
        untar=True)
    flowers_root = pathlib.Path(flowers_root)
    flowers_ds = tf.data.Dataset.list_files(str(flowers_root / '*/*'))
    flowers_ds = flowers_ds.map(process_path)

    def test_pipeline(self):
        size_crops = [224, 168]
        nmb_crops = [2, 3]
        b_s = 16
        swav_ds = SwaVDataset(self.flowers_ds,
                              size_crops=size_crops,
                              nmb_crops=nmb_crops,
                              min_scale_crops=[0.14, 0.16],
                              max_scale_crops=[1., 1.],
                              b_s=b_s)
        random_images = next(iter(swav_ds.dataset_swaved.take(1)))
        self.assertEqual(len(random_images), sum(nmb_crops))
        crops = [[i] * j for (i, j) in zip(size_crops, nmb_crops)]
        crops = [item for sublist in crops for item in sublist]
        for i, img in enumerate(random_images):
            shape = img.shape
            self.assertAllLessEqual(tf.round(img), 1.)
            self.assertAllGreaterEqual(img, 0.)
            self.assertEqual(len(shape), 4)
            self.assertTrue(shape[1] == shape[2] == crops[i])
            self.assertTrue(shape[0], b_s)


if __name__ == '__main__':
    unittest.main()
