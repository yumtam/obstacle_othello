import keras
from keras import layers
import numpy as np
import game.bitboard as bitop
import game.util as util


weight_path = 'weights/'


class ResidualCNN:
    def __init__(self):
        self.model = ResidualCNN.make_model()

    @staticmethod
    def convolutional_layer(x, filters, kernel_size):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return x

    @staticmethod
    def residual_layer(x, filters, kernel_size):
        orig = x
        x = ResidualCNN.convolutional_layer(x, filters, kernel_size)
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, orig])
        x = layers.LeakyReLU()(x)
        return x

    @staticmethod
    def make_model():
        inputs = keras.Input(shape=(8, 8, 3))
        x = ResidualCNN.convolutional_layer(inputs, filters=36, kernel_size=(3, 3))

        for _ in range(6):
            x = ResidualCNN.residual_layer(x, filters=36, kernel_size=(3, 3))

        p = ResidualCNN.convolutional_layer(x, filters=2, kernel_size=(1, 1))
        p = layers.Flatten()(p)
        policy_head = layers.Dense(8 * 8 + 1)(p)

        v = ResidualCNN.convolutional_layer(x, filters=1, kernel_size=(1, 1))
        v = layers.Flatten()(v)
        v = layers.Dense(36)(v)
        v = layers.ReLU()(v)
        value_head = layers.Dense(1, activation='tanh')(v)

        model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryCrossentropy()],
        )

        model.summary()

        return model


    def load(self, id):
        self.model.load_weights(weight_path + str(id))

    def save(self, id):
        self.model.save_weights(weight_path + str(id))

    def eval(self, my, opp, obs):
        arr = bitop.bits_to_array(my, opp, obs)
        arr = np.moveaxis(arr, 0, -1)
        arr = np.reshape(arr, (1, 8, 8, 3))
        return self.model.predict(arr)


m = ResidualCNN()
m.save(0)