import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class Attention_Encoder(keras.layers.Layer):
    def __init__(self, time_steps, nb_rnn_units, **kwargs):
        super(Attention_Encoder, self).__init__(**kwargs)

        self.en_dense_We = keras.layers.Dense(time_steps, use_bias=False)
        self.en_dense_Ue = keras.layers.Dense(time_steps, use_bias=False)
        self.en_dense_ve = keras.layers.Dense(1, use_bias=False)

        self.en_LSTM_cell = keras.layers.LSTM(nb_rnn_units, return_state=True)

    def call(self, inputs, **kwargs):
        # x_shape : <batch_size, time_steps, input_dims>
        x, s, h = inputs[0], inputs[1], inputs[2]
        time_steps, nb_time_series = K.shape(x)[1], K.shape(x)[2]
        attention_weight_t = None
        for t in range(time_steps):
            context = self.one_encoder_attention_step(h, s, x, time_steps)  #(batch_size, 1, input_dims)
            x = keras.layers.Lambda(lambda x: x[:, t, :])(x)
            x = keras.layers.Reshape((1, nb_time_series))(x)
            h, _, s = self.en_LSTM_cell(x, initial_state=[h, s])
            if t!= 0:
                attention_weight_t = keras.layers.Concatenate(axis=1)([attention_weight_t, context])
            else:
                attention_weight_t = context
        x_ = keras.layers.Multiply()([attention_weight_t, x])

        return x_

    def one_encoder_attention_step(self, h_prev, s_prev, x, time_steps):
        """
        :param h_prev: previous hidden state
        :param s_prev: previous cell state
        :param x: (T, n), n is length of input series at time t, T is length of time series
        :return: x_t's attention weights, total n numbers, sum these are 1
        """
        concat = keras.layers.Concatenate()([h_prev, s_prev])         # <none, 1, 2m>  m is the lstm hidden units
        r1 = self.en_dense_We(concat)                                 # <none, 1, T>
        r1 = keras.layers.RepeatVector(x.shape[2])(r1)                # <none, n, T>
        xt = keras.layers.Permute((2, 1))(x)                          # <none, n, T>
        r2 = self.en_dense_Ue(time_steps)(xt)                         # <none, n, T>
        r3 = keras.layers.Add()([r1, r2])                             # <none, n, T>
        r4 = keras.layers.Activation(activation='tanh')(r3)           # <none, n, T>
        r5 = self.en_dense_ve(r4)                                     # <none, n, 1>
        r5 = keras.layers.Permute((2, 1))(r5)                         # <none, 1, n>
        alpha = keras.layers.Activation(activation='softmax')(r5)     # <none, 1, n>

        return alpha


class Attention_Decoder(keras.layers.Layer):
    def __init__(self,  time_steps, nb_rnn_units, **kwargs):
        super(Attention_Decoder, self).__init__(**kwargs)
        self.time_steps = time_steps

        self.dn_dense_We = keras.layers.Dense(nb_rnn_units, use_bias=False)
        self.dn_dense_Ue = keras.layers.Dense(nb_rnn_units, use_bias=False)
        self.dn_dense_ve = keras.layers.Dense(1, use_bias=False)
        self._pred_dense = keras.layers.Dense(1)

        self.dn_LSTM_cell = keras.layers.LSTM(nb_rnn_units, return_state=True)


    def call(self, inputs, **kwargs):
        # y_shape: <batch_size, time_steps, 1>
        h_en_all, y, s, h = inputs[0], inputs[1], inputs[2], inputs[3]
        for t in range(self.time_steps - 1):
            y_prev = keras.layers.Lambda(lambda y: y[:, t, :])(y)
            y_prev = keras.layers.Reshape((1, 1))(y_prev)                 # (batch_size, 1, 1)
            context = self.one_decoder_attention_step(h, s, h_en_all)     # (batch_size, 1, nb_rnn_units)
            y_prev = keras.layers.Concatenate(axis=2)([y_prev, context])  # (batch_size, 1, nb_rnn_units+1)
            y_prev = self._pred_dense(y_prev)                             # (batch_size, 1, 1)
            h, _, s = self.dn_LSTM_cell(y_prev, initial_state=[h, s])

        context = self.one_decoder_attention_step(h, s, h_en_all)

        return h, context

    def one_decoder_attention_step(self, h_de_prev, s_de_prev, h_en_all):
        """
        :param h_de_prev: previous hidden state
        :param s_de_prev: previous cell state
        :param h_en_all: (None, T, m), m is length of rnn output series at time t, T is length of time series
        :return: x_t's attention weights, total m numbers, sum these are 1
        """
        concat = keras.layers.Concatenate()([h_de_prev, s_de_prev])      # <none, 1, 2p> p is the lstm hidden units  p=m
        r1 = self.dn_dense_We(concat)                                    # <none, 1, m>
        r1 = keras.layers.RepeatVector(self.time_steps)(r1)              # <none, T, m>
        r2 = self.dn_dense_Ue(h_en_all)                                  # <none, T, m>
        r3 = keras.layers.Add()([r1, r2])                                # <none, T, m>
        r4 = keras.layers.Activation(activation='tanh')(r3)              # <none, T, m>
        r5 = self.dn_dense_ve(r4)                                        # <none, T, 1>
        r5 = keras.layers.Permute((2, 1))(r5)                            # <none, 1, T>
        beta = keras.layers.Activation(activation='softmax')(r5)         # <none, 1, T>
        context = keras.layers.Dot(axes=1)([beta, h_en_all])             # <none, 1, m>

        return context




def DARNN(init, X_input_shape, Y_input_shape):
    ip_x = keras.layers.Input(X_input_shape)
    ip_y = keras.layers.Input(Y_input_shape)

    en_s0 = keras.layers.Lambda(lambda x: K.zeros(shape=(K.shape(x)[0], init.RNNUnits)))(ip_x)
    en_h0 = keras.layers.Lambda(lambda x: K.zeros(shape=(K.shape(x)[0], init.RNNUnits)))(ip_x)

    de_s0 = keras.layers.Lambda(lambda x: K.zeros(shape=(K.shape(x)[0], init.RNNUnits)))(ip_x)
    de_h0 = keras.layers.Lambda(lambda x: K.zeros(shape=(K.shape(x)[0], init.RNNUnits)))(ip_x)

    x_ = Attention_Encoder(time_steps=init.TimeSteps, nb_rnn_units=init.RNNUnits)([ip_x, en_s0, en_h0])

    h_en_all = keras.layers.LSTM(init.RNNUnits, return_sequences=True)(x_)
    h_en_all = keras.layers.Reshape((init.TimeSteps, -1))(h_en_all)
    h, context = Attention_Decoder(time_steps=init.TimeSteps, nb_rnn_units=init.RNNUnits)([h_en_all, ip_y, de_s0, de_h0])
    concat = keras.layers.Concatenate(axis=2)([h, context])
    concat = keras.layers.Reshape((-1,))(concat)
    result = keras.layers.Dense(init.RNNUnits)(concat)
    out = keras.layers.Dense(init.FeatDims)(result)

    if init.task == 'classification':
        out = keras.layers.Activation('relu')(out)

    model = keras.models.Model(inputs=[ip_x, ip_y], outputs=out)

    return model

