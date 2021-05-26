import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Input, Lambda, Dropout, LayerNormalization, GlobalAveragePooling1D, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import TimeDistributed


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
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        result = inputs + self.position_embeddings

        return result    

class MultiHeadSelfAttention(Layer):

    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads

        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

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
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)      # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)      # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)

        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)

        return output


class TransformerDecoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerDecoder, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-8)
        self.layernorm2 = LayerNormalization(epsilon=1e-8)
        self.dropout1 =  Dropout(rate)
        self.dropout2 =  Dropout(rate)

    def call(self, inputs, training, **kwargs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

def Transformer(init, input_shape):
    inp = Input(input_shape)
    embeddings = Dense(init.model_dim)(inp)
    position_encodings = PositionEncoding(init.model_dim)(embeddings)
    encodings = embeddings + position_encodings

    for i in range(init.encoder_stack):
        encodings = TransformerDecoder(embed_dim=init.model_dim, num_heads=init.heads, ff_dim=init.ff_dim)(encodings)

    out = GlobalAveragePooling1D()(encodings)
    out = Dense(init.FeatDims)(out)

    if init.task == 'classification':
        out = Activation('softmax')(out)

    model = Model(inputs=inp, outputs=out)

    return model



