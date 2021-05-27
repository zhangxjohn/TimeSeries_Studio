import tensorflow as tf 
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K


def naive_RNN(init, input_shape, mode='singal-layer'):

    # input_shape: [batch, timesteps, features]

    inputs = Input(shape=input_shape)

    if mode =='singal-layer':
        x = SimpleRNN(init.RNNUnits)(inputs)
    else:
        # multi-layer
        x, _ = SimpleRNN(init.RNNUnits, return_sequences=True, return_state=True)(inputs)
        x, _ = SimpleRNN(init.RNNUnits, return_sequences=True, return_state=True)(x)
        _, x = SimpleRNN(init.RNNUnits, return_sequences=True, return_state=True)(x)

    outputs = Dense(init.FeatDims)(x)
    if init.task == 'classification':
        outputs = Activation('softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def LSTMs(init, input_shape, mode='singal-layer'):

    # input_shape: [batch, timesteps, features]

    inputs = Input(shape=input_shape)

    if mode =='singal-layer':
        x = LSTM(init.RNNUnits)(inputs)
    else:
        # multi-layer
        x, _ = LSTM(init.RNNUnits, return_sequences=True, return_state=True)(inputs)
        x, _ = LSTM(init.RNNUnits, return_sequences=True, return_state=True)(x)
        _, x = LSTM(init.RNNUnits, return_sequences=True, return_state=True)(x)

    outputs = Dense(init.FeatDims)(x)
    if init.task == 'classification':
        outputs = Activation('softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def GRUs(init, input_shape, mode='singal-layer'):

    # input_shape: [batch, timesteps, features]

    inputs = Input(shape=input_shape)

    if mode =='singal-layer':
        x = GRU(init.RNNUnits)(inputs)
    else:
        # multi-layer
        x, _, _ = GRU(init.RNNUnits, return_sequences=True, return_state=True)(inputs)
        x, _, _ = GRU(init.RNNUnits, return_sequences=True, return_state=True)(x)
        _, _, x = GRU(init.RNNUnits, return_sequences=True, return_state=True)(x)

    outputs = Dense(init.FeatDims)(x)
    if init.task == 'classification':
        outputs = Activation('softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model 
