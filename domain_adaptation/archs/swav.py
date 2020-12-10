from typing import List

from logging import getLogger
from itertools import groupby
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

logger = getLogger()


class SwAV:

    def __init__(self,
                 model: models.Model,
                 p_d1: int = 1024,
                 p_d2: int = 96,
                 p_dim: int = 10,
                 normalize: bool = True,
                 sinkhorn_iter: int = 5,
                 epsilon: float = 0.05,
                 temperature: float = 0.1,
                 crops_for_assign: List[int] = [0, 1]) -> None:
        self.model = model
        self.l2norm = normalize
        self.p_d1, self.p_d2, self.p_dim = p_d1, p_d2, p_dim
        self.prototype_model = self.prototype(self.p_d1, self.p_d2, self.p_dim)
        self.sinkhorn_iter, self.epsilon = sinkhorn_iter, epsilon
        self.temperature = temperature
        self.crops_for_assign = crops_for_assign
        self.epoch_loss: List[float] = []
        self.step_loss: List[float] = []

    def fit(
        self,
        dataloader: tf.data.Dataset,  # Create a SwaVDataset param
        optimizer: optimizers.Optimizer,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        for ep in range(epochs):
            w = self.prototype_model.get_layer("prototype").get_weights()
            w = tf.math.l2_normalize(w, axis=1)
            self.prototype_model.get_layer.set_weights(w)
            pbar = tqdm(enumerate(dataloader))
            for i, inputs in pbar:
                loss = self.epoch(dataloader, optimizer, i)
                self.step_loss.append(loss)
                pbar.set_description(
                    f"Current loss: {np.mean(self.step_loss):.4f}")
            self.epoch_loss.append(np.mean(self.step_loss))
            logger.info(f"Epoch: {ep+1}/{epochs}\t"
                        f"Loss: {np.mean(self.step_loss):.4f}")

    def epoch(self, dataloader, optimizer, i):
        for _, inputs in enumerate(dataloader):
            images = inputs.tolist()
            b_s = images[0].shape[0]
            # getting a list of consecutive idxs with same crop size
            crop_sizes = [img[0].shape[1] for img in images]
            idx_crops = tf.math.cumsum(
                [len(list(g)) for _, g in groupby(crop_sizes)], axis=0)
            start = 0

            with tf.GradientTape() as tape:
                for end in idx_crops:
                    concat_input = tf.stop_gradient(
                        tf.concat(values=inputs[start:end], axis=0))
                    _embedding = self.model(concat_input)
                    if start == 0:
                        embeddings = _embedding
                    else:
                        embeddings = tf.concat(values=(embeddings, _embedding),
                                               axis=0)
                    start = end
                projection, prototype = self.prototype_model(embeddings)
                projection = tf.stop_gradient(projection)

                loss = 0
                for i, crop_id in enumerate(self.crops_for_assign):
                    with tape.stop_recording():
                        out = prototype[b_s * crop_id:b_s * (crop_id + 1)]
                        clus = self.sinkhorn(out)[-b_s:]

                    subloss = 0
                    for v in np.delete(
                            np.arange(np.sum(dataloader.data.nb_crops)),
                            crop_id):
                        prob = tf.nn.softmax(prototype[b_s * v:b_s * (v + 1)] /
                                             self.temperature)
                        subloss -= tf.math.reduce_mean(
                            tf.math.reduce_sum(clus * tf.math.log(prob),
                                               axis=1))
                    loss += subloss / tf.cast(
                        tf.reduce_sum(dataloader.data.nb_crops) - 1,
                        tf.float32)
                loss /= len(self.crops_for_assign)
            vars = (self.model.trainable_variables +
                    self.prototype_model.trainable_variables)
            gradients = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(gradients, vars))

            return loss

    def prototype(self, d1: int, d2: int, dim: int) -> models.Model:
        inputs = layers.Input((2048,))
        projection1 = layers.Dense(d1)(inputs)
        projection1 = layers.BatchNormalization()(projection1)
        projection1 = layers.Activation("relu")(projection1)
        projection2 = layers.Dense(d2, name="projection")(projection1)
        if self.l2norm:
            projection2 = tf.math.l2_normalize(projection2,
                                               axis=1,
                                               name="projection_normalized")
        prototype = layers.Dense(dim, use_bias=False,
                                 name="prototype")(projection2)
        SwAV.prototype_head(prototype, "prototype")
        return models.Model(inputs=inputs, outputs=[projection2, prototype])

    @staticmethod
    def prototype_head(prototype: models.Model,
                       prototype_layer: str = "prototype") -> None:
        weights = tf.transpose(
            prototype.get_layer(prototype_layer).get_weigts())
        weights_norm = tf.math.l2_normalize(weights, axis=1)
        prototype.get_layer(prototype_layer).set_weights(weights_norm)

    def sinkhorn(
        self,
        scores,
    ):
        Q = tf.transpose(tf.exp(scores / self.epsilon))
        Q /= tf.keras.backend.sum(Q)
        K, B = Q.shape
        u, r, c = tf.zeros(K, dtype=tf.float32), tf.ones(
            K, dtype=tf.float32), tf.ones(B, dtype=tf.float32) / B
        for _ in range(self.sinkhorn_iter):
            u = tf.keras.backend.sum(Q, axis=1)
            Q *= tf.expand_dims((r / u), axis=1)
            Q *= tf.expand_dims(c / tf.keras.backend.sum(Q, axis=0), axis=0)
        return tf.transpose(Q / tf.keras.backend.sum(Q, axis=0, keepdims=True))
