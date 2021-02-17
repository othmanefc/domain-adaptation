import unittest
import pathlib

import tensorflow as tf
from tensorflow import test

from domain_adaptation.archs.swav import SwAV
from domain_adaptation.datasets.SwaVDataset import SwaVDataset
from domain_adaptation.datasets import utils
from domain_adaptation.models import resnet


def process_path(file_path):
    return utils.process_path(file_path,
                              width=180,
                              height=180,
                              channels=3,
                              with_label=False)


class TestSwAV(test.TestCase):
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
    b_s = 32
    ds = SwaVDataset(flowers_ds.take(256),
                     size_crops=SIZE_CROPS,
                     nmb_crops=NMB_CROPS,
                     min_scale_crops=MIN_SCALE_CROPS,
                     max_scale_crops=MAX_SCALE_CROPS)
    res_mod = resnet.Resnet50().model

    P_D1, P_D2 = 1024, 128
    P_DIM = 10
    swav_mod = SwAV(model=res_mod,
                    p_d1=P_D1,
                    p_d2=P_D2,
                    p_dim=P_DIM,
                    normalize=True,
                    sinkhorn_iter=5,
                    epsilon=0.05,
                    temperature=0.1,
                    crops_for_assign=[0, 1])

    def test_prototype(self):
        projection = self.swav_mod.prototype_model.get_layer('projection')
        prototype = self.swav_mod.prototype_model.get_layer('prototype')
        self.assertEqual((projection.output_shape, prototype.output_shape),
                         ((None, self.P_D2), (None, self.P_DIM)))

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
