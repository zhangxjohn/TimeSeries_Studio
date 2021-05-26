import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class NBeatsLayer(keras.layers.Layer):
    def __init__(self,units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False, nb_harmonics=None, **kwargs):
        super(NBeatsLayer, self).__init__(**kwargs)
        self.units=units
        self.thetas_dim=thetas_dim
        self.share_thetas=share_thetas
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.fc1 = keras.layers.Dense(units=self.units, activation='relu')
        self.fc2 = keras.layers.Dense(units=self.units, activation='relu')
        self.fc3 = keras.layers.Dense(units=self.units, activation='relu')
        self.fc4 = keras.layers.Dense(units=self.units, activation='relu')
        self.backcast_linspace, self.forecast_linspace = self.linspace(backcast_length, forecast_length)

        if self.share_thetas:
            self.theta_b_fc=self.theta_f_fc=keras.layers.Dense(units=self.thetas_dim, activation='relu', use_bias=False)
        else:
            self.theta_b_fc = keras.layers.Dense(units= self.thetas_dim, activation='relu', use_bias=False)
            self.theta_f_fc = keras.layers.Dense(units= self.thetas_dim, activation='relu', use_bias=False)

    def call(self, inputs, **kwargs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def linspace(self, backcast_length, forecast_length):
        lin_space = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
        b_ls=lin_space[:backcast_length]
        f_ls=lin_space[backcast_length:]
        return b_ls, f_ls


class GenericBlock(NBeatsLayer):
    def __init__(self,units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False, nb_harmonics=None, **kwargs):
        super(GenericBlock, self).__init__(units, thetas_dim, share_thetas, **kwargs)

        self.backcast_fc = keras.layers.Dense(units=backcast_length)
        self.forecast_fc = keras.layers.Dense(units=forecast_length)

    def call(self, x, **kwargs):
        x = super(GenericBlock, self).call(x)
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)
        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)
        return backcast, forecast


class TrendBlock(NBeatsLayer):
    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None, **kwargs):
        super(TrendBlock, self).__init__(units, thetas_dim, backcast_length,
                                            forecast_length, share_thetas=True, **kwargs)

    def call(self, x, **kwargs):
        x=super(TrendBlock, self).call(x)
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)
        backcast=self.trend_model(theta_b, self.backcast_linspace)
        forecast=self.trend_model(theta_f, self.forecast_linspace)
        return backcast, forecast

    def trend_model(self, thetas, t):
        p = thetas.get_shape().as_list()[-1]
        assert  p<=4, 'thetas_dim is too big.'
        T = tf.convert_to_tensor([tf.pow(t,i) for i in range(p)], dtype=tf.float32)
        return tf.matmul(thetas, T, transpose_b=True)


class SeasonalityBlock(NBeatsLayer):
    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None, **kwargs):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, backcast_length,
                                    forecast_length, share_thetas=True, **kwargs)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, backcast_length,
                                    forecast_length, share_thetas=True, **kwargs)

    def call(self, x, **kwargs):
        x = super(SeasonalityBlock, self).call(x)
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)
        backcast = self.seasonality_model(theta_b, self.backcast_linspace)
        forecst = self.seasonality_model(theta_f, self.forecast_linspace)
        return backcast, forecst

    def seasonality_model(self, thetas, t):
        p=thetas.get_shape().as_list()[-1]
        assert  p <= thetas.shape[1], 'thetas_dim is too big.'
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
        s1 = tf.convert_to_tensor([tf.cos(2*math.pi*i*t) for i in range(p1)], dtype=tf.float32)  # H/2-1
        s2 = tf.convert_to_tensor([tf.sin(2*math.pi*i*t) for i in range(p2)], dtype=tf.float32)
        S = tf.concat([s1, s2], axis=0)
        return tf.matmul(thetas, S, transpose_b=True)


class NBeatsNet(keras.layers.Layer):

    def __init__(self,
                 stack_types,
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None,
                 **kwargs):
        super(NBeatsNet, self).__init__(**kwargs)
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units=hidden_layer_units
        self.nb_blocks_per_stack=nb_blocks_per_stack
        self.stack_types=stack_types
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.thetas_dim = thetas_dim
        self.stacks = []

        self.block_type = {
            'trend_block': TrendBlock,
            'seasonality_block': SeasonalityBlock,
            'general': GenericBlock
        }

        self.stacks=[]
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))

    def call(self, backcast, **kwargs):
        forecast = tf.zeros([tf.shape(backcast)[0], self.forecast_length], dtype=tf.float32)
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f

        return backcast, forecast

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = self.block_type[stack_type]
            if self.share_weights_in_stack and block_id != 0:
                block=blocks[-1]  # pick up the last one when we share weights
            else:
                block=block_init(self.hidden_layer_units,
                                 self.theta_dims[stack_id],
                                 self.backcast_length,
                                 self.forecast_length,
                                 self.nb_harmonics)
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks


def NbeatsNet(init, input_shape):

    nbeat = NBeatsNet(stack_types=init.stack_types,
                    nb_blocks_per_stack=init.nb_blocks_per_stack,
                    forecast_length=init.forecast_length,
                    backcast_length=init.backcast_length,
                    thetas_dim=init.thetas_dim,
                    share_weights_in_stack=init.share_weights_in_stack,
                    hidden_layer_units=init.hidden_layer_units)

    x = keras.layers.Input(input_shape)
    _, y = nbeat(x)

    if init.task == 'classification':
        out = keras.layers.Activation('softmax')(y)

    model = keras.models.Model(inputs=x, outputs=y)

    return model