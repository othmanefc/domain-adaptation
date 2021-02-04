import tensorflow as tf


class SGD:
    def __init__(self,
                 initial_lr: float = 0.1,
                 decay_steps: int = 1000,
                 momentum: float = 0.8,
                 **kwargs):

        self.lr_decayed_fn = tf.keras.experimental.CosineDecay(
            initial_learning_rate=initial_lr, decay_steps=decay_steps)
        self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_decayed_fn,
                                           momentum=momentum,
                                           **kwargs)
