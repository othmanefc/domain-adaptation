import tensorflow as tf
from tensorflow.keras import models, layers


class DeepCluster:

    def __init__(self, model: models.Model):
        pass

    def fit(elf,
            dataloader: SwaVDataset,
            optimizer: optimizers.Optimizer,
            epochs: int = 50):
        num_batches = dataloader.dataset_swaved.cardinality().numpy()
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
            for i, inputs in pbar:
                loss = self.epoch(inputs, optimizer)
                self.step_loss.append(loss)
                pbar.set_description(
                    f"Current loss: {np.mean(self.step_loss):.4f}")
            self.epoch_loss.append(np.mean(self.step_loss))
            logger.info(f"Epoch: {ep+1}/{epochs}\t"
                        f"Loss: {np.mean(self.step_loss):.4f}")

    def epoch(self, inputs, optimizer):
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
        return models.Model(inputs=inputs, outputs=[projection2, prototype])
