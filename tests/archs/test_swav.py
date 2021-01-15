import unittest

import tensorflow as tf

from domain_adaptation.datasets.SwaVDataset import SwaVDataset
from domain_adaptation.datasets import utils
from domain_adaptation.archs import swav
from domain_adaptation.models import resnet


def process_path(file_path):
    return utils.process_path(file_path,
                              width=180,
                              height=180,
                              channels=3,
                              with_label=False)


class SwaVTest(unittest.TestCase):
    flowers_root = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/'
        'example_images/flower_photos.tgz',
        untar=True)
    flowers_root = pathlib.Path(flowers_root)
    flowers_ds = tf.data.Dataset.list_files(str(flowers_root / '*/*'))
    flowers_ds = flowers_ds.map(process_path)

    def test_fit(self):
        ds = SwaVDataset.SwaVDataset(self.flowers_ds,
                                     override=False,
                                     nmb_crops=[3, 4],
                                     size_crops=[224, 168],
                                     min_scale_crops=[224, 168],
                                     max_scale_crops=[1., 1.]).dataset_swaved
        model = resnet.Resnet50().models
        swav_model = swav.SwAV(model=model)
        print(swav_model.prototype_model.summary())

        decay_steps = 1000
        lr_decayed_fn = tf.keras.experimental.CosineDecay(
            initial_learning_rate=.1, decay_steps=decay_steps)
        opt = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn)
        swav_model.fit(ds, optimizer=opt, epochs=10)


if __name__ == '__main__':
    unittest.main()
