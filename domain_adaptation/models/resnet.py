from tensorflow.keras import Input, layers, Model  # type: ignore
from tensorflow.keras.activations import relu  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore


class Resnet50:

    def __init__(self):
        self.model = self.init_model()

    def init_model(self):

        input_im = Input(shape=(None, None, 3))
        x = layers.ZeroPadding2D(padding=(3, 3))(input_im)
        # 1st stage
        # here we perform maxpooling, see the figure above

        x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(relu)(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # 2nd stage
        # frm here on only conv block and identity block, no pooling

        x = self._res_conv(x, s=1, filters=(64, 256))
        x = self._res_identity(x, filters=(64, 256))
        x = self._res_identity(x, filters=(64, 256))

        # 3rd stage

        x = self._res_conv(x, s=2, filters=(128, 512))
        x = self._res_identity(x, filters=(128, 512))
        x = self._res_identity(x, filters=(128, 512))
        x = self._res_identity(x, filters=(128, 512))

        # 4th stage

        x = self._res_conv(x, s=2, filters=(256, 1024))
        x = self._res_identity(x, filters=(256, 1024))
        x = self._res_identity(x, filters=(256, 1024))
        x = self._res_identity(x, filters=(256, 1024))
        x = self._res_identity(x, filters=(256, 1024))
        x = self._res_identity(x, filters=(256, 1024))

        # 5th stage

        x = self._res_conv(x, s=2, filters=(512, 2048))
        x = self._res_identity(x, filters=(512, 2048))
        x = self._res_identity(x, filters=(512, 2048))

        # ends with average pooling and dense connection

        x = layers.GlobalAveragePooling2D()(x)

        # define the model

        model = Model(inputs=input_im, outputs=x, name='Resnet50')

        return model

    def _res_identity(self, x, filters):
        # renet block where dimension doesnot change.
        # The skip connection is just simple identity conncection
        # we will have 3 blocks and then input will be added

        x_skip = x  # this will be used for addition with the residual block
        f1, f2 = filters

        # first block
        x = layers.Conv2D(f1,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='valid',
                          kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(relu)(x)

        # second block # bottleneck (but size kept same with padding)
        x = layers.Conv2D(f1,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(relu)(x)

        # third block activation used after adding the input
        x = layers.Conv2D(f2,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='valid',
                          kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)

        # add the input
        x = layers.Add()([x, x_skip])
        x = layers.Activation(relu)(x)

        return x

    def _res_conv(self, x, s, filters):
        x_skip = x
        f1, f2 = filters

        # first block
        x = layers.Conv2D(f1,
                          kernel_size=(1, 1),
                          strides=(s, s),
                          padding='valid',
                          kernel_regularizer=l2(0.001))(x)
        # when s = 2 then it is like downsizing the feature map
        x = layers.BatchNormalization()(x)
        x = layers.Activation(relu)(x)

        # second block
        x = layers.Conv2D(f1,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(relu)(x)

        # third block
        x = layers.Conv2D(f2,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='valid',
                          kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)

        # shortcut
        x_skip = layers.Conv2D(f2,
                               kernel_size=(1, 1),
                               strides=(s, s),
                               padding='valid',
                               kernel_regularizer=l2(0.001))(x_skip)
        x_skip = layers.BatchNormalization()(x_skip)

        # add
        x = layers.Add()([x, x_skip])
        x = layers.Activation(relu)(x)

        return x
