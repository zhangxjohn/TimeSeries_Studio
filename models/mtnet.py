import numpy as np
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
        

class Attention_Encoder(keras.layers.Layer):
    def __init__(self,
                 nb_filters,
                 kernel_size,
                 en_rnn_hidden_sizes,
                 tiny_time_steps,
                 ltms_n,
                 strides=1,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 **kwargs):
        super(Attention_Encoder, self).__init__(**kwargs)
        self.ltms_n = ltms_n  # The number of the long-term memory series
        self.tiny_time_steps= tiny_time_steps
        self.nb_filters  =nb_filters
        # conv
        self.conv_dropout = keras.Sequential([
            keras.layers.Conv1D(nb_filters, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal'),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(dropout)
        ])
        # output shape [batch_size, new_time_steps, nb_filters]

        # tmporal attention layer and gru layer
        self.attention = Attention(tiny_time_steps)

        self.gru = keras.Sequential()
        for h_size in en_rnn_hidden_sizes:
            self.gru.add(keras.layers.GRU(h_size, activation='relu', dropout=dropout, recurrent_dropout=recurrent_dropout, 
                                return_sequences=True, return_state=True))

    def build(self, input_shape):
        super(Attention_Encoder, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape: [batch_size, n*tiny_time_steps, nb_time_series]
        batch_size, nb_time_series = tf.shape(inputs)[0], tf.shape(inputs)[2]

        x = tf.reshape(inputs, shape=[batch_size, self.ltms_n, self.tiny_time_steps, -1])
        x = tf.reshape(x, shape=[-1, self.tiny_time_steps, nb_time_series])

        conv = self.conv_dropout(x)
        # <batch_size, ltms_n, new_time_series, nb_filters>
        rnn_input = tf.reshape(conv, shape=[-1, self.ltms_n, tf.shape(conv)[1], self.nb_filters])

        # ltms_n * <batch_size, last_rnn_size>
        res_hstates = tf.TensorArray(tf.float32, self.ltms_n)
        for k in range(self.ltms_n):
            # <batch_size, new_time_series, nb_filters>
            attr_input = rnn_input[:, k]
            x_t = self.attention(attr_input)
            _, final_state = self.gru(x_t)
            res_hstates.write(k, final_state)
        # <batch_size, ltms_n, last_rnn_size>
        y = tf.transpose(res_hstates.stack(), perm=[1, 0, 2])
        
        return y


class PreARTrans(keras.layers.Layer):
    def __init__(self, hw, **kwargs):
        # hw: Highway = Number of timeseries values to consider for the linear layer (AR layer)

        self.hw = hw
        super(PreARTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PreARTrans, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        batch_size = tf.shape(x)[0]
        input_shape = K.int_shape(x)
        output = x[:, -self.hw:, :]

        # Permute axis 1 and 2. axis=2 is the the dimension having different time-series
        # This dimension should be equal to 'm' which is the number of time-series.
        output = tf.transpose(output, perm=[0, 2, 1])

        # Merge axis 0 and 1 in order to change the batch size
        output = tf.reshape(output, [batch_size * input_shape[2], self.hw])

        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.hw

        return tf.TensorShape(shape)

    def get_config(self):
        config = {'hw': self.hw}
        base_config = super(PreARTrans, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PostARTrans(keras.layers.Layer):
    def __init__(self, m, **kwargs):
        # m: Number of timeseries

        self.m = m
        super(PostARTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PostARTrans, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, original_model_input = inputs

        batch_size = tf.shape(original_model_input)[0]
        output = tf.reshape(x, [batch_size, self.m])

        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.m

        return tf.TensorShape(shape)

    def get_config(self):
        config = {'m': self.m}
        base_config = super(PostARTrans, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SplitInput(keras.layers.Layer):
    def __init__(self, ltms_n, time_step, **kwargs):
        super(SplitInput, self).__init__(**kwargs)
        self.n = ltms_n
        self.time_step = time_step

    def build(self, input_shape):
        super(SplitInput, self).build(input_shape)
        self.shape = input_shape

    def call(self, inputs, **kwargs):
        input1 = inputs[:, : self.n*self.time_step, :]
        input2 = inputs[:, -self.n*self.time_step:, :]

        return [input1, input2]

    def compute_output_shape(self, input_shape):
        shape1 = list(self.shape)
        shape2 = list(self.shape)
        shape1[1] = self.n*self.time_step
        shape2[1] = self.shape[1] - self.n*self.time_step
        return [shape1, shape2]


def MTNet(init, input_shape):
    # input_shape: [batch_size, time_steps, nb_time_series]
    inp = keras.layers.Input(input_shape)
    # x -> [batch_size, n*tiny_time_steps, nb_time_series]
    # q -> [batch_size, tiny_time_steps, nb_time_series]
    tiny_time_steps = input_shape[0] // (init.ltms_n+1)
    x, q = SplitInput(init.ltms_n, tiny_time_steps)(inp)

    # ------------- no-linear component -------------
    # <batch_size, ltms_n, en_rnn_hidden_sizes>
    m_is = Attention_Encoder(
                nb_filters=init.CNNFilters,
                kernel_size=init.CNNKernel,
                en_rnn_hidden_sizes=init.en_rnn_hidden_sizes,
                tiny_time_steps= tiny_time_steps,
                ltms_n=init.ltms_n, name='m')(x)
    # <batch_size, 1, en_rnn_hidden_sizes>
    c_is = Attention_Encoder(
                nb_filters=init.CNNFilters,
                kernel_size=init.CNNKernel,
                en_rnn_hidden_sizes=init.en_rnn_hidden_sizes,
                tiny_time_steps=tiny_time_steps,
                ltms_n=init.ltms_n, name='c')(x)
    # <batch_size, 1, en_rnn_hidden_sizes>
    u = Attention_Encoder(
                nb_filters=init.CNNFilters,
                kernel_size=init.CNNKernel,
                en_rnn_hidden_sizes=init.en_rnn_hidden_sizes,
                tiny_time_steps=tiny_time_steps,
                ltms_n=init.ltms_n, name='c')(q)

    p_is = tf.matmul(m_is, tf.transpose(u, perm=[0, 2, 1]))

    # using softmax
    p_is = tf.squeeze(p_is, axis=[-1])
    p_is = tf.nn.softmax(p_is)
    # <batch_size, n, 1>
    p_is = tf.expand_dims(p_is, -1)

    o_is = tf.multiply(c_is, p_is)

    pred_x = tf.concat([o_is, u], axis=1)
    pred_x = tf.reshape(pred_x, shape=[-1, init.en_rnn_hidden_sizes[-1]*(init.ltms_n + 1)])

    y = keras.layers.Dense(input_shape[1])(pred_x)

    # Auto Regressive
    if init.highway > 0:
        z = PreARTrans(init.highway)(q)
        z = keras.layers.Flatten()(z)
        z = keras.layers.Dense(1)(z)
        z = PostARTrans(input_shape[1])([z, x])

        y = keras.layers.Add()([y, z])

    out = keras.layers.Dense(init.FeatDims)(y)
    if init.task == 'classification':
        out = keras.layers.Activation('softmax')(out)

    model = keras.models.Model(inputs=inp, outputs=out)

    return model
