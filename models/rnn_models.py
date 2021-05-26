import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import backend as K


def naive_RNN(init, input_shape, mode='singal-layer'):

    # input_shape: [batch, timesteps, features]

    inputs = keras.layers.Input(shape=input_shape)

    if mode =='singal-layer':
        x = keras.layers.SimpleRNN(init.RNNUnits)(inputs)
    else:
        # multi-layer
        x, _ = keras.layers.SimpleRNN(init.RNNUnits, return_sequences=True, return_state=True)(inputs)
        x, _ = keras.layers.SimpleRNN(init.RNNUnits, return_sequences=True, return_state=True)(x)
        _, x = keras.layers.SimpleRNN(init.RNNUnits, return_sequences=True, return_state=True)(x)

    outputs = keras.layers.Dense(init.FeatDims)(x)
    if init.task == 'classification':
        outputs = keras.layers.Activation('softmax')(outputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    return model


def LSTM(init, input_shape, mode='singal-layer'):

    # input_shape: [batch, timesteps, features]

    inputs = keras.layers.Input(shape=input_shape)

    if mode =='singal-layer':
        x = keras.layers.LSTM(init.RNNUnits)(inputs)
    else:
        # multi-layer
        x, _ = keras.layers.LSTM(init.RNNUnits, return_sequences=True, return_state=True)(inputs)
        x, _ = keras.layers.LSTM(init.RNNUnits, return_sequences=True, return_state=True)(x)
        _, x = keras.layers.LSTM(init.RNNUnits, return_sequences=True, return_state=True)(x)

    outputs = keras.layers.Dense(init.FeatDims)(x)
    if init.task == 'classification':
        outputs = keras.layers.Activation('softmax')(outputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    return model


def GRU(init, input_shape, mode='singal-layer'):

    # input_shape: [batch, timesteps, features]

    inputs = keras.layers.Input(shape=input_shape)

    if mode =='singal-layer':
        x = keras.layers.LSTM(init.RNNUnits)(inputs)
    else:
        # multi-layer
        x, _, _ = keras.layers.LSTM(init.RNNUnits, return_sequences=True, return_state=True)(inputs)
        x, _, _ = keras.layers.LSTM(init.RNNUnits, return_sequences=True, return_state=True)(x)
        _, _, x = keras.layers.LSTM(init.RNNUnits, return_sequences=True, return_state=True)(x)

    outputs = keras.layers.Dense(init.FeatDims)(x)
    if init.task == 'classification':
        outputs = keras.layers.Activation('softmax')(outputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    return model 



