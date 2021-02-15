import unittest
import pathlib

import tensorflow as tf
from tensorflow import test

from domain_adaptation.archs.deepcluster import DeepCluster
from domain_adaptation.datasets.SwaVDataset import SwaVDataset
from domain_adaptation.datasets import utils
from domain_adaptation.models import resnet


def process_path(file_path):
    return utils.process_path(file_path,
                              width=180,
                              height=180,
                              channels=3,
                              with_label=False)


class TestDeepCluster(test.TestCase):
    flowers_root = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/'
        'example_images/flower_photos.tgz',
        untar=True)
    flowers_root = pathlib.Path(flowers_root)
    flowers_ds = tf.data.Dataset.list_files(str(flowers_root / '*/*'))
    flowers_ds = flowers_ds.map(process_path)

    SIZE_CROPS = [224, 168]
    NMB_CROPS = [2, 3]
    MIN_SCALE_CROPS = [.14, .16]
    MAX_SCALE_CROPS = [1., 1.]
    b_s = 16
    ds = SwaVDataset(flowers_ds.take(256),
                     size_crops=SIZE_CROPS,
                     nmb_crops=NMB_CROPS,
                     min_scale_crops=MIN_SCALE_CROPS,
                     max_scale_crops=MAX_SCALE_CROPS)
    res_mod = resnet.Resnet50().model

    PROTOTYPES = [50, 50, 50]
    P_D1, FEAT_DIM = 1024, 128
    swav_mod = DeepCluster(model=res_mod,
                           p_d1=P_D1,
                           feat_dim=FEAT_DIM,
                           nmb_prototypes=PROTOTYPES,
                           crops_for_assign=[0, 1],
                           temperature=.1)

    def test_prototype(self):
        for i, prots in enumerate(self.PROTOTYPES):
            layer = self.swav_mod.prototype_model.get_layer(f'prototype_{i}')
            self.assertEqual(layer.output_shape, (None, prots))
        projection = self.swav_mod.prototype_model.get_layer('projection')
        self.assertEqual(projection.output_shape, (None, self.FEAT_DIM))

    def test_fit(self):
        DECAY_STEPS = 1000
        EPOCHS = 2
        lr_decayed_fn = tf.keras.experimental.CosineDecay(
            initial_learning_rate=0.1, decay_steps=DECAY_STEPS)
        opt = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn)
        self.swav_mod.fit(self.ds, optimizer=opt, epochs=EPOCHS)
        self.assertEqual(len(self.swav_mod.epoch_loss), EPOCHS)


if __name__ == '__main__':
    unittest.main()
