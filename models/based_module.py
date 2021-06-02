import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, SimpleRNN, GRU, LSTM, SeparableConv1D, Bidirectional
from tensorflow.keras.layers import Layer, Lambda, Dropout, Activation, LayerNormalization, BatchNormalization


ts_custom_objects = {
    'SimpleRNN': SimpleRNN,
    'GRU': GRU,
    'LSTM': LSTM,
    'BidirectionalLSTM': BidirectionalLSTM,
    'SkipGRU': SkipGRU,
    'Autoregressive': Autoregressive,
    'PositionEncoding': PositionEncoding,
    'TrainablePositionEncoding': TrainablePositionEncoding,
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
    'MultiheadAttention': MultiheadAttention,
    'CausalResidual1D': CausalResidual1D,
    'GatedDilatedConv1D': GatedDilatedConv1D,
    'DenseAttention': DenseAttention,
    'SpatialAttention': SpatialAttention,
    'SqueezeExcitation': SqueezeExcitation,
}


class SkipGRU(Layer):
    """
    pt: the number of period.
    skip: period length.
    cnn_unit: the number of 1D conv output filters.
    rnn_unit: the number of rnn outputs.

    input: CNN output : (none, time_steps, input_dim)
    output: (none, skip*rnn_units)
    """
    def __init__(self, pt, skip, cnn_units, rnn_units, dropout=0.0, *kwargs):
        super(SkipGRU, self).__init__(*kwargs)

        self.pretrans = Sequential([
            Lambda(lambda k: k[:, int(-pt*skip):, :]),
            Lambda(lambda k: K.reshape(k, (-1, pt, skip, cnn_units))),
            Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1, 3))),
            Lambda(lambda k: K.reshape(k, (-1, pt, cnn_units)))
        ])
        self.gru = GRU(rnn_units, activation='relu')
        self.posttrans = Lambda(lambda k: K.reshape(k, (-1, skip*rnn_units)))

    def call(self, inputs, training, *kwargs):
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
    def __init__(self, hw, input_dim, *kwargs):
        super(Autoregressive, self).__init__(*kwargs)

        self.pretrans = Sequential([
            Lambda(lambda k: k[:, -hw:, :]),
            Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1))),
            Lambda(lambda k: K.reshape(k, (-1, hw)))
        ])
        self.linear = Dense(1)
        self.posttrans = Lambda(lambda k: K.reshape(k, (-1, input_dim)))

    def call(self, inputs, *kwargs):
        z = self.pretrans(inputs)
        z = self.linear(z)
        z = self.posttrans(z)

        return z


class PositionEncoding(Layer):
    
    def __init__(self, model_dim, **kwargs):
        self.model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        seq_length = inputs.shape[1]

        position_encodings = np.zeros((seq_length, self.model_dim))
        for pos in range(seq_length):
            for i in range(self.model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i-i%2) / self.model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])
        position_encodings = K.cast(position_encodings, 'float32')
        return  position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape


class TrainablePositionEncoding(Layer):

    def __init__(self, **kwargs):
        super(TrainablePositionEncoding, self).__init__(**kwargs)
    
    def build(self, input_shape):
        sequence_length, d_model = input_shape[-2:]
        self.position_embeddings = self.add_weight(
            shape=(sequence_length, d_model),
            initializer='uniform',
            name='word_position_embeddings',
            trainable=True)
        super(TrainablePositionEncoding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        position_encodings = inputs + self.position_embeddings

        return position_encodings    


class MultiHeadSelfAttention(Layer):
    """
    input: original time series: (none, time_steps, nb_variables)
    output: (none, time_steps, nb_variables)
    """
    def __init__(self, embed_dim, num_heads=8, use_residual=False, use_bn=False, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.use_bn = use_bn

        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads

    def build(self, input_shape):
        self.query_dense = Dense(self.embed_dim, activation='relu')
        self.key_dense = Dense(self.embed_dim, activation='relu')
        self.value_dense = Dense(self.embed_dim, activation='relu')
        self.combine_heads = Dense(self.embed_dim, activation='relu')
        self.residual = Dense(self.embed_dim, activation='relu')
        self.batch_norm = BatchNormalization()
        super(MultiHeadSelfAttention, self).build(input_shape)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs) 
        key = self.key_dense(inputs)      
        value = self.value_dense(inputs) 

        if self.use_residual:
            res = self.residual(inputs)

        query = self.separate_heads(query, batch_size) 
        key = self.separate_heads(key, batch_size)     
        value = self.separate_heads(value, batch_size) 

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) 

        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  
        outputs = self.combine_heads(concat_attention)

        if self.use_residual:
            outputs += res 
        outputs = self.batch_norm(outputs) if self.use_bn else outputs

        return outputs


class CausalResidual1D(Layer):
    """
    input: original time series: (none, time_steps, nb_variables)
    output: (none, time_steps, nb_variables)
    """
    def __init__(self, nb_filters, kernel_size, dropout=0.1, dilation_rate=1, norm_before=False, **kwargs):
        super(CausalResidual1D, self).__init__(**kwargs)
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilation_rate = dilation_rate
        self.norm_before = norm_before

    def build(self, input_shape):
        self.conv_layers1 = Conv1D(self.nb_filters, self.kernel_size, 
                        padding='same', dilation_rate=1, activation='relu')
        self.conv_layers2 = Conv1D(self.nb_filters, self.kernel_size, 
                        padding='same', dilation_rate=1, activation='relu')
        self.dilated_conv_layers1 = Conv1D(self.nb_filters, self.kernel_size, 
                        padding='causal', dilation_rate=self.dilation_rate, activation='relu')
        self.dilated_conv_layers2 = Conv1D(self.nb_filters, self.kernel_size, 
                        padding='causal', dilation_rate=self.dilation_rate, activation='relu')
        self.norm =  LayerNormalization()
        self.dropout = Dropout(self.dropout)
        super(CausalResidual1D, self).build(input_shape)

    def call(self, inputs, masks=None, **kwargs):
        no_zero_mask = tf.cast(masks, tf.float32) if masks is None else tf.cast(tf.ones_like(inputs), tf.float32)

        x2 = self.norm(inputs) if self.norm_before else x
        
        x2 = self.conv_layers1(x2) * no_zero_mask
        x2 = self.dilated_conv_layers1(x2) * no_zero_mask
        x2 = self.conv_layers2(x2) * no_zero_mask
        x2 = self.dilated_conv_layers2(x2) * no_zero_mask  

        x = x + self.dropout(x2)

        outputs = self.norm(x) if not self.norm_before else x

        return outputs   


class GatedDilatedConv1D(Layer):
    """
    input: original time series: (none, time_steps, nb_variables)
    output: attention time series: (none, new_time_steps, new_nb_variables)
    """
    def __init__(self, nb_filters, kernel_size, dilation_rate=1, **kwargs):
        super(GatedDilatedConv1D, self).__init__(**kwargs)
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.conv_x = Conv1D(self.nb_filters, 1, padding='same', activation='relu')
        self.conv_f = Conv1D(self.nb_filters, self.kernel_size, padding='causal', 
                        dilation_rate=self.dilation_rate, activation='tanh')
        self.conv_g = Conv1D(self.nb_filters, self.kernel_size, padding='causal', 
                        dilation_rate=self.dilation_rate, activation='sigmoid')
        self.conv_z = Conv1D(self.nb_filters, 1, padding='same', activation='relu')
        super(GatedDilatedConv1D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        conv_x = self.conv_x(inputs)
        conv_f = self.conv_f(conv_x)
        conv_g = self.conv_g(conv_x)
        conv_z = tf.multiply(conv_f, conv_g)
        conv_z = self.conv_z(conv_z)
        outputs = tf.add([conv_x, conv_z])
        return outputs

   
class DenseAttention(Layer):
    """
    input: original time series: (none, time_steps, nb_variables)
    output: attention time series: (none, time_steps, nb_variables)
    """
    def __init__(self, **kwargs):
        super(DenseAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        time_steps = input_shape[1]
        self.dense = Dense(time_steps, activation='softmax')
        super(DenseAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.transpose(inputs, perm=[0, 2, 1])
        x = self.dense(x)
        x_probs = tf.transpose(x, perm=[0, 2, 1])
        x = tf.multiply(inputs, x_probs)
        return x


class SpatialAttention(Layer):
    """
    input: original time series: (none, time_steps, nb_variables)
    output: attention time series: (none, time_steps, nb_variables)
    """
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)   
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.dense = Dense(input_dim, activation='softmax')
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x_probs = self.dense(inputs)
        x = tf.multiply(inputs, x_probs)
        return x


class SqueezeExcitation(Layer):
    """
    input: original time series: (none, time_steps, nb_variables)
    output: attention time series: (none, time_steps, nb_variables)
    """
    def __init__(self, pooling_op='mean', reduction_ratio=3, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs) 
        self.pooling_op = pooling_op
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        self.seq_length = input_shape[1]
        self.input_dim = input_shape[-1]
        self.reduction_num = max(self.seq_length//self.reduction_ratio, 1)
        self.dense_att1 = Dense(self.reduction_num, activation='relu')
        self.dense_att2 = Dense(self.seq_length, activation='sigmoid')
        super(SqueezeExcitation, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.pooling_op == 'max':
            x = tf.reduce_max(inputs, axis=-1)
        else:
            x = tf.reduce_mean(inputs, axis=-1)
        x = self.dense_att1(x)
        x = self.dense_att2(x)
        outputs = tf.multiply(inputs, tf.expand_dims(x, axis=2))
        return outputs


class BidirectionalLSTM(Layer):
    """
    input: original time series: (none, time_steps, nb_variables)
    output: attention time series: (none, time_steps, new_nb_variables)
    """
    def __init__(self, nb_units, **kwargs):
        self.nb_units = nb_units
        super(BidirectionalLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.blstm = Bidirectional(LSTM(self.nb_units, return_sequences=True))
        super(BidirectionalLSTM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.blstm(inputs)


