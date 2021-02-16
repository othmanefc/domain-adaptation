from typing import List, Tuple
import os
import logging
from itertools import groupby, compress
from tqdm import tqdm

import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

from domain_adaptation.datasets.SwaVDataset import SwaVDataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DeepCluster:

    def __init__(self,
                 model: models.Model,
                 p_d1: int,
                 feat_dim: int,
                 nmb_prototypes: List[int],
                 crops_for_assign: List[int],
                 temperature: float = 0.1):
        self.model = model
        self.crops_for_assign = crops_for_assign
        self.nmb_prototypes = nmb_prototypes
        self.p_d1, self.p_dim = p_d1, feat_dim
        self.feat_dim = feat_dim
        self.l2norm = True
        self.temperature = 0.1,
        self.prototype_model = self.prototype(self.p_d1, self.feat_dim,
                                              self.nmb_prototypes)
        self.epoch_loss: List[float] = []
        self.step_loss: List[float] = []

    def fit(self,
            dataloader: SwaVDataset,
            optimizer: optimizers.Optimizer,
            epochs: int = 50,
            nb_kmeans_iters: int = 10):
        num_batches = dataloader.dataset_swaved.cardinality().numpy()
        self.local_mem_idx, self.local_mem_ebds = self._init_memory(
            dataloader, num_batches)
        for ep in range(epochs):
            if num_batches in [
                    tf.data.INFINITE_CARDINALITY, tf.data.UNKNOWN_CARDINALITY
            ]:
                logger.warn(
                    "couldn't compute number of batches, no progress bar...")
                pbar = tqdm(enumerate(dataloader.dataset_swaved))
            else:
                pbar = tqdm(enumerate(dataloader.dataset_swaved),
                            total=num_batches,
                            desc=f"Epoch {ep+1}/{epochs}")
            logger.info(f"Epoch: {ep+1}...")
            self.assignements = self._cluster_memory(
                size_dataset=num_batches * dataloader.b_s,
                nb_kmeans_iters=nb_kmeans_iters)
            self.step_loss = []
            for i, inputs in pbar:
                loss = self.epoch(inputs, optimizer, i, dataloader.nmb_crops)
                self.step_loss.append(loss)
                pbar.set_description(
                    f"Current loss {np.mean(self.step_loss):.4f}")
            epoch_loss = np.mean(self.step_loss)
            self.epoch_loss.append(epoch_loss)
            logger.info(f"Epoch: {ep+1}/{epochs} Loss: {epoch_loss:.4f}")

    def epoch(self, inputs: Tuple, optimizer: tf.optimizers, idx: int,
              nb_crops: List[int]) -> float:
        images = list(inputs)
        b_s = images[0].shape[0]
        crop_sizes = [img.shape[1] for img in images]
        idx_crops = tf.math.cumsum(
            [len(list(g)) for _, g in groupby(crop_sizes)], axis=0)
        start = 0
        with tf.GradientTape() as tape:
            for end in idx_crops:
                concat_input = tf.concat(values=inputs[start:end], axis=0)
                _embedding = self.model(concat_input)
                if start == 0:
                    embeddings = _embedding
                else:
                    embeddings = tf.concat(values=(embeddings, _embedding),
                                           axis=0)
                start = end
            projection, prototypes = self.prototype_model(embeddings)
            # prototypes = tf.stop_gradient(prototypes)

            loss = .0
            for h in range(len(self.nmb_prototypes)):
                scores = prototypes[h] / self.temperature
                targets = tf.tile(
                    self.assignements[h][b_s * idx:b_s * (idx + 1)],
                    [sum(nb_crops)])
                loss += tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        targets, scores))
            loss /= len(self.nmb_prototypes)
        varrs = (self.model.trainable_variables +
                 self.prototype_model.trainable_variables)
        gradients = tape.gradient(loss, varrs)
        optimizer.apply_gradients(zip(gradients, varrs))

        self._update_memory(b_s, start, idx, projection)
        start += b_s

        return loss

    def prototype(self, d1: int, d2: int, dims: List[int]) -> models.Model:
        inputs = layers.Input((2048,))
        projection1 = layers.Dense(d1)(inputs)
        projection1 = layers.BatchNormalization()(projection1)
        projection1 = layers.Activation("relu")(projection1)
        projection2 = layers.Dense(d2, name="projection")(projection1)
        if self.l2norm:
            projection2 = tf.math.l2_normalize(projection2,
                                               axis=1,
                                               name="projection_normalized")
        prototypes = []
        for i, dim in enumerate(dims):
            prototype = layers.Dense(dim,
                                     use_bias=False,
                                     name=f"prototype_{i}")(projection2)
            prototypes.append(prototype)
        return models.Model(inputs=inputs, outputs=[projection2, prototypes])

    def _init_memory(self, data: SwaVDataset,
                     num_batches: int) -> Tuple[tf.Variable, tf.Variable]:
        size = data.b_s * num_batches
        local_memory_index = tf.Variable(tf.zeros(size, dtype=tf.int32))
        local_memory_embeddings = tf.Variable(
            tf.zeros((len(self.crops_for_assign), size, self.feat_dim),
                     dtype=tf.float32))
        start_idx = 0
        print('Initializing memory banks...')
        logger.info('Initializing memory banks...')
        for idx, inputs in tqdm(enumerate(data.dataset_swaved),
                                total=num_batches):
            inputs = list(inputs)
            nb_unique_idx = inputs[0].shape[0]
            inps = tf.concat([inputs[i] for i in self.crops_for_assign],
                             axis=0)
            embeddings = self.model(inps)
            projection, _ = self.prototype_model(embeddings)
            projection = tf.stop_gradient(projection)

            local_memory_index[start_idx:start_idx + nb_unique_idx].assign(
                [idx] * (nb_unique_idx))
            projection_reshaped = tf.reshape(
                projection,
                (len(self.crops_for_assign), data.b_s, self.feat_dim))
            local_memory_embeddings[:, start_idx:start_idx +
                                    nb_unique_idx].assign(projection_reshaped)
            start_idx += nb_unique_idx
        return local_memory_index, local_memory_embeddings

    def _cluster_memory(self, size_dataset: int,
                        nb_kmeans_iters: int) -> tf.Variable:
        j = 0
        assignements = tf.Variable(-100 * tf.ones(
            (len(self.nmb_prototypes), size_dataset), dtype=tf.int32))

        for i, proto in enumerate(self.nmb_prototypes):
            mem_ebd = self.local_mem_ebds[j]
            centroids = tf.Variable(
                tf.raw_ops.Empty(shape=(proto, self.feat_dim),
                                 dtype=tf.float32))
            rand_idx = tf.random.shuffle(range(len(mem_ebd)))[:proto]
            assert len(rand_idx) >= proto, "reduce the number of centroids"
            centroids = tf.Variable(tf.gather(mem_ebd, rand_idx))

            for n_iter in range(nb_kmeans_iters + 1):
                dot = tf.matmul(mem_ebd, centroids, transpose_b=True)
                local_assignments = tf.argmax(dot, axis=1)

                if n_iter == nb_kmeans_iters:
                    break

                where_helper = DeepCluster._get_indices_sparse(
                    local_assignments.numpy())
                counts = tf.Variable(tf.zeros(proto, dtype=tf.int32))
                emb_sums = tf.Variable(
                    tf.zeros((proto, self.feat_dim), dtype=tf.float32))
                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        summ = tf.reduce_sum(tf.gather(mem_ebd,
                                                       where_helper[k][0]),
                                             axis=0)
                        emb_sums[k].assign(summ)
                        counts[k].assign(len(where_helper[k][0]))
                mask = counts > 0
                counts_mask = tf.expand_dims(counts[mask], axis=1)
                inds = list(compress(range(len(mask)), mask))
                for idx, ii in enumerate(inds):
                    val = emb_sums[ii] / tf.cast(counts_mask[idx], tf.float32)
                    centroids[ii].assign(val)
                centroids = tf.math.l2_normalize(centroids, axis=1)
                centroids = tf.Variable(centroids)

            centroids_t = tf.transpose(centroids)
            self.prototype_model.get_layer(f"prototype_{i}").set_weights(
                [centroids_t])
            assignements[i].assign(tf.cast(local_assignments, tf.int32))

            j = (j + 1) % len(self.crops_for_assign)
        return assignements

    def _update_memory(self, b_s: int, start_idx: int, idx: int,
                       projection: tf.Variable) -> None:
        self.local_mem_idx[idx:idx + b_s].assign([idx] * b_s)
        for i, crop_idx in enumerate(self.crops_for_assign):
            self.local_mem_ebds[i, start_idx:start_idx + b_s].assign(
                projection[crop_idx * b_s:(crop_idx + 1) * b_s])

    def _broadcast(self, mask: tf.Tensor,
                   broadcasted_to: tf.Variable) -> tf.Tensor:
        len_diff = len(broadcasted_to.shape) - len(mask.shape)
        shape = [1] * len_diff + list(mask.shape)
        mask_reshaped = tf.reshape(mask, shape)
        broadcast = list(
            broadcasted_to.shape)[:len_diff] + [1] * len(mask.shape)
        mask_broadcast = tf.tile(mask_reshaped, broadcast)
        return mask_broadcast

    @staticmethod
    def _get_indices_sparse(data):
        cols = np.arange(data.size)
        matrix = csr_matrix((cols, (data.ravel(), cols)),
                            shape=(int(data.max()) + 1, data.size))
        return [np.unravel_index(row.data, data.shape) for row in matrix]

    def save(self, path):
        directory = os.mkdir(path)
        self.model.save(os.path.join(directory, "main_model"))
        self.prototype_model.save(os.path.join(directory, "prototype_model"))

    def load(self, path):
        self.model = models.load_model(os.path.join(path, "main_model"))
        self.prototype_model = models.load_model(
            os.path.join(path, "prototype_model"))
