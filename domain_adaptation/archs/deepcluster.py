from typing import List
import logging
from itertools import groupby
from tqdm import tqdm

import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

from domain_adaption.datasets.SwaVDataset import SwaVDataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DeepCluster:
    def __init__(self, model: models.Model, feat_dim: int,
                 nmb_prototypes: List[int], crops_for_assign: List[int]):
        self.model = model
        self.prototype_model = self.prototype()
        self.crops_for_assign = crops_for_assign
        self.nmb_prototypes = nmb_prototypes
        self.feat_dim = feat_dim
        self.epoch_loss: List[float] = []
        self.step_loss: List[float] = []

    def fit(self,
            dataloader: SwaVDataset,
            optimizer: optimizers.Optimizer,
            epochs: int = 50,
            nb_kmeans_iters: int = 10):
        num_batches = dataloader.dataset_swaved.cardinality().numpy()
        self.local_mem_idx, self.local_mem_ebds = self._init_memory(
            self.dataloader, num_batches)
        for ep in range(epochs):
            if num_batches in [
                    tf.data.INFINITE_CARDINALITY, tf.data.UNKNOWN_CARDINALITY
            ]:
                logger.warn(
                    "couldn't compute number of batches, no progress bar...")
                pbar = tqdm(enumerate(dataloader.dataset_swaved))
            else:
                pbar = tqdm(enumerate(dataloader.dataset_swaved),
                            total=num_batches)
            logger.info(f"Epoch: {ep+1}...")
            self.assignments = self._cluster_memory(
                size_dataset=num_batches * dataloader.b_s,
                nb_kmeans_iters=nb_kmeans_iters)
            for i, inputs in pbar:
                loss = self.epoch(inputs, optimizer, i, dataloader.nmb_crops)
                self.step_loss.append(loss)
                pbar.set_description(
                    f"Current loss: {np.mean(self.step_loss):.4f}")
            self.epoch_loss.append(np.mean(self.step_loss))
            logger.info(f"Epoch: {ep+1}/{epochs}\t"
                        f"Loss: {np.mean(self.step_loss):.4f}")

    def epoch(self, inputs, optimizer, idx, nb_crops):
        images = list(inputs)
        b_s = images[0].shape[0]
        crop_sizes = [img.shape[1] for img in images]
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
            projection, prototypes = self.prototype_model(embeddings)
            projection = tf.stop_gradient(projection)

            loss = .0
            for h in range(len(self.nmb_prototypes)):
                with tape.stop_recording():
                    scores = projection[h] / self.temperature
                    targets = tf.tile(self.assignements[h][i], (sum(nb_crops)))

    def prototype(self, d1: int, d2: int, dims: List[int]) -> models.Model:
        inputs = layers.Input((2048, ))
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
        return models.Model(inputs=inputs, outputs=[projection2, [prototypes]])

    def _init_memory(self, data: SwaVDataset, num_batches):
        if isinstance(num_batches, int):
            size = data.b_s * num_batches
        else:
            size = data.dataset_swaved.map(
                lambda x: 1,
                num_parallel_calls=tf.data.experimental.AUTOTUNE).reduce(
                    tf.constant(0), lambda x, _: x + 1) * data.b_s
        local_memory_index = tf.zeros(size, dtype=tf.int32)
        local_memory_embeddings = tf.zeros(
            (len(self.crops_for_assign), size, self.feat_dim),
            dtype=tf.float32)
        start_idx = 0

        logger.info('Initializing memory banks...')
        for idx, inputs in enumerate(data.dataset_swaved):
            inputs = list(inputs)
            nb_unique_idx = inputs[0].shape[0]
            crop_sizes = [img.shape[1] for img in inputs]
            idx_crops = tf.math.cumsum(
                [len(list(g)) for _, g in groupby(crop_sizes)], axis=0)
            start = 0
            with tf.GradientTape() as _:
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
                projection, _ = self.prototype_model(embeddings)
                projection = tf.stop_gradient(projection)

            local_memory_index[start_idx:start_idx + nb_unique_idx] = idx
            for i in range(len(projection)):
                local_memory_embeddings[i][start_idx:start_idx +
                                           nb_unique_idx] = embeddings
            start_idx += nb_unique_idx
        return local_memory_index, local_memory_embeddings

    def _cluster_memory(self, size_dataset: int, nb_kmeans_iters: int):
        j = 0
        assignements = -100 * tf.ones(len(self.nmb_prototypes), size_dataset)

        for i, proto in enumerate(self.nmb_prototypes):
            mem_ebd = self.local_mem_ebds[j]
            centroids = tf.raw_ops.Empty((proto, self.feat_dim))
            rand_idx = tf.random.shuffle(range(len(mem_ebd)))[:proto]
            assert len(rand_idx) >= proto, "reduce the number of centroids"
            centroids = mem_ebd[j][rand_idx]

            for n_iter in range(nb_kmeans_iters + 1):
                dot = tf.matmul(mem_ebd, centroids, transpose_b=True)
                local_assignments = tf.argmax(dot, axis=1)

                if n_iter == nb_kmeans_iters:
                    break

                where_helper = DeepCluster._get_indices_sparse(
                    local_assignments.numpy())
                counts = tf.zeros(proto, dtype=tf.int32)
                emb_sums = tf.zeros((proto, self.feat_dim))
                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        emb_sums[k] = tf.reduce_sum(
                            mem_ebd[where_helper[k][0]],
                            axis=0,
                        )
                        counts[k] = len(where_helper[k][0])
                mask = counts > 0
                counts_mask = tf.expand_dims(counts[mask], axis=1)
                centroids[mask] = emb_sums[mask] / counts_mask
                centroids = tf.math.l2_normalize(centroids, axis=1)
            self.prototype_model.get_layer(f"prototypes_{i}").set_weights(
                centroids)
            assignements[i][self.local_mem_idx] = local_assignments

            j = (j + 1) % len(self.crops_for_assign)
        return assignements

    @staticmethod
    def _get_indices_sparse(data):
        cols = np.arange(data.size)
        matrix = csr_matrix((cols, (data.ravel(), cols)),
                            shape=(int(data.max()) + 1, data.size))
        return [np.unravel_index(row.data, data.shape) for row in matrix]
