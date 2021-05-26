import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class FullyConvNets(keras.layers.Layer):

    def __init__(self, nb_filters=128, **kwargs):
        super(FullyConvNets, self).__init__(**kwargs)

        self.conv1 = keras.Sequential([
            keras.layers.Conv1D(nb_filters, 8, 1, padding='same', kernel_initializer='he_normal'),
            # keras.layers.BatchNormalization(),
            keras.layers.Activation('relu')
        ])

        self.conv2 = keras.Sequential([
            keras.layers.Conv1D(nb_filters*2, 5, 1, padding='same', kernel_initializer='he_normal'),
            # keras.layers.BatchNormalization(),
            keras.layers.Activation('relu')
        ])

        self.conv3 = keras.Sequential([
            keras.layers.Conv1D(nb_filters, 3, 1, padding='same', kernel_initializer='he_normal'),
            # keras.layers.BatchNormalization(),
            keras.layers.Activation('relu')
        ])

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def FCN(init, input_shape):
    # input_shape: [batch_size, time_steps, nb_time_series]
    x = keras.layers.Input(input_shape)

    y = FullyConvNets(init.CNNFilters)(x)
    y = keras.layers.GlobalAveragePooling1D()(y)
    # y = keras.layers.Dropout(0.1)(y)
    y = keras.layers.Dense(init.FeatDims)(y)

    if init.task == 'classification':
        y = keras.layers.Activation('softmax')(y)

    model = keras.models.Model(inputs=x, outputs=y)

    return model

