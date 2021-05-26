import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class Attention(keras.layers.Layer):
    def __init__(self, TimeSteps, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.dense = keras.layers.Dense(TimeSteps, activation='softmax')

    def call(self, inputs, **kwargs):
        x = tf.transpose(inputs, perm=[0, 2, 1])
        x = self.dense(x)
        x_probs = tf.transpose(x, perm=[0, 2, 1])
        x = tf.multiply(inputs, x_probs)
        return x

class FCN(keras.layers.Layer):

    def __init__(self,
                 nb_filters=128,
                 **kwargs):
        super(FCN, self).__init__(**kwargs)

        self.conv1 = keras.Sequential([
            keras.layers.Conv1D(nb_filters, 8, 1, padding='same', kernel_initializer='he_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu')
        ])

        self.conv2 = keras.Sequential([
            keras.layers.Conv1D(nb_filters * 2, 5, 1, padding='same', kernel_initializer='he_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu')
        ])

        self.conv3 = keras.Sequential([
            keras.layers.Conv1D(nb_filters, 3, 1, padding='same', kernel_initializer='he_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu')
        ])

        self.avgpool = keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        return x


def LSTM_FCN(init, input_shape):
    # input_shape: [batch_size, time_steps, nb_time_series]
    i = keras.layers.Input(input_shape)

    # Attention
    if init.is_attention:
        x = keras.layers.Masking()(i)
        x = Attention(input_shape[0])(x)
        x = keras.layers.LSTM(init.RNNUnits)(x)
        x = keras.layers.Dropout(0.8)(x)
    else: 
        x = keras.layers.Masking()(i)
        x = keras.layers.LSTM(init.RNNUnits)(x)
        x = keras.layers.Dropout(0.8)(x)

    # FCN
    y = FCN(init.CNNFilters)(i)

    x_y = keras.layers.concatenate([x, y])
    o = keras.layers.Dense(init.FeatDims)(x_y)
    
    if init.task == 'classification':
        o = keras.layers.Activation('softmax')(o)
    
    model = keras.models.Model(inputs=i, outputs=o)

    return model


def ALSTM_FCN(init, input_shape):
    # input_shape: [batch_size, time_steps, nb_time_series]
    i = keras.layers.Input(input_shape)

    # Attention-LSTM
    x = keras.layers.Masking()(i)
    x = Attention(input_shape[0])(x)
    x = keras.layers.LSTM(init.RNNUnits)(x)
    x = keras.layers.Dropout(0.8)(x)

    # FCN
    y = FCN(init.CNNFilters)(i)

    x_y = keras.layers.concatenate([x, y])
    o = keras.layers.Dense(init.FeatDims)(x_y)

    if init.task == 'classification':
        o = keras.layers.Activation('softmax')(o)
    
    model = keras.models.Model(inputs=i, outputs=o)

    return model
