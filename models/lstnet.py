import numpy as np
from tensorflow.keras.layers import Layer, Input, Lambda, Dropout, Dense, GRU, Conv1D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K


class SkipGRU(Layer):
    """
    pt: the number of period.
    skip: period length.
    cnn_unit: the number of 1D conv output filters.
    rnn_unit: the number of rnn outputs.

    input: CNN output : (none, time_steps, filters)
    output: (none, skip*rnn_units)
    """
    def __init__(self, pt, skip, cnn_units, rnn_units, *kwargs):
        super(SkipGRU, self).__init__(*kwargs)

        # c: batch_size*steps*filters, steps=P-Ck
        self.pretrans = Sequential([
            Lambda(lambda k: k[:, int(-pt*skip):, :]),
            Lambda(lambda k: K.reshape(k, (-1, pt, skip, cnn_units))),
            Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1, 3))),
            Lambda(lambda k: K.reshape(k, (-1, pt, cnn_units)))
        ])
        self.gru = GRU(rnn_units)
        self.posttrans = Lambda(lambda k: K.reshape(k, (-1, skip*rnn_units)))

    def call(self, inputs, *kwargs):
        s = self.pretrans(inputs)
        s = self.gru(s)
        s = self.posttrans(s)
        return s

class Autoregressive(Layer):
    """
    hw: horizon window, the window size of the highway component.
    m:  the number of time series variables.

    input: original time series: (none, time_steps, nb_variables)
    output: (none, nb_variables)
    """
    def __init__(self, hw, m, *kwargs):
        super(Autoregressive, self).__init__(*kwargs)

        self.pretrans = Sequential([
            Lambda(lambda k: k[:, -hw:, :]),
            Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1))),
            Lambda(lambda k: K.reshape(k, (-1, hw)))
        ])
        self.linear = Dense(1)
        self.posttrans = Lambda(lambda k: K.reshape(k, (-1, m)))

    def call(self, inputs, *kwargs):
        z = self.pretrans(inputs)
        z = self.linear(z)
        z = self.posttrans(z)

        return z


def LSTNet(init, input_shape):
    x = Input(input_shape)

    # CNN
    c = Conv1D(init.CNNFilters, kernel_size=init.CNNKernel, activation='relu')(x)
    c = Dropout(init.dropout)(c)

    # RNN
    r = GRU(init.RNNUnits)(c)
    r = Lambda(lambda k: K.reshape(k, (-1, init.RNNUnits)))(r)
    r = Dropout(init.dropout)(r)

    # Skip-RNN
    if init.skip > 0:
        pt = int((init.time_steps - init.CNNKernel + 1) / init.skip)
        s = Lambda(lambda k: k[:, int(-pt*init.skip):, :])(c)
        s = Lambda(lambda k: K.reshape(k, (-1, pt, init.skip, init.CNNFilters)))(s)
        s = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1, 3)))(s)
        s = Lambda(lambda k: K.reshape(k, (-1, pt, init.CNNFilters)))(s)

        s = GRU(init.SkipRNNUnits)(s)
        s = Lambda(lambda k: K.reshape(k, (-1, init.skip*init.SkipRNNUnits)))(s)
        ###############################################
        # s = SkipGRU(pt, init.skip, init.CNNFilters, init.SkipRNNUnits)(c)
        ###############################################

        s = Dropout(init.dropout)(s)
        r = concatenate([r, s])
    
    # Dense
    y = Dense(input_shape[1])(r)

    # Autoregressive
    if init.highway_window > 0:
        z = Lambda(lambda k: k[:, -init.highway:, :])(x)
        z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
        z = Lambda(lambda k: K.reshape(k, (-1, init.highway)))(z)
        z = Dense(1)(z)
        z = Lambda(lambda k: K.reshape(k, (-1, input_shape[1])))(z)
        ###############################################
        # z = Autoregressive(init.highway_window, input_shape[1])(x)
        ###############################################
        y = add([y, z])

    # y = Dense(init.FeatDims)(y)
    if init.task == 'classification':
        y = Activation('softmax')(y)    

    model = Model(inputs=x, outputs=y)

    return model


def AR(init, input_shape):
    x = Input(input_shape)

    y = Autoregressive(init.highway_window, input_shape[1])(x)

    model = Model(inputs=x, outputs=y)

    return model

