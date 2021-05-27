import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv1D, Dense, BatchNormalization, Activation, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.layers import add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K


class BasicBlock(Layer):
    # Residual basic block
    def __init__(self, nb_filters, kernel_size, stride=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)

        self.conv1 = Sequential([
            Conv1D(nb_filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu')
        ])

        self.conv2 = Sequential([
            Conv1D(nb_filters, kernel_size, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization()
        ])

        if stride != 1:
            self.shortcut = Sequential([
                Conv1D(nb_filters, kernel_size=1, strides=stride, kernel_initializer='he_normal'),
                BatchNormalization()
            ])
        else:
            self.shortcut = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        i = self.shortcut(inputs)
        x = add([x, i])
        x = tf.nn.relu(x)
        return x

class ResidualNet(Layer):

    def __init__(self, layer_dims, kernel_sizes, feat_dim, **kwargs):
        self.inplanes = 4
        super(ResidualNet, self).__init__(**kwargs)
        self.stem = Sequential([
            Conv1D(self.inplanes, kernel_size=3, strides=1, kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPool1D(2, 1, padding='same')
        ])
        self.layer1 = self.build_block(self.inplanes*1, layer_dims[0], kernel_sizes[0], stride=1)
        self.layer2 = self.build_block(self.inplanes*2, layer_dims[1], kernel_sizes[1], stride=2)
        self.layer3 = self.build_block(self.inplanes*4, layer_dims[2], kernel_sizes[2], stride=2)

        self.avgpool = GlobalAveragePooling1D()
        self.fc = Dense(feat_dim)

    def call(self, inputs, **kwargs):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_block(self, nb_filter, blocks, kernel_size, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(nb_filter, kernel_size, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(nb_filter, kernel_size, stride))
        return res_blocks


def ResNet(init, input_shape):
    # input_shape: [batch_size, time_steps, nb_time_series]
    x = Input(input_shape)

    y = ResidualNet(layer_dims=init.ResLayers, kernel_sizes=init.ResKernels, feat_dim=init.FeatDims)(x)

    if init.task == 'classification':
        y = Activation('softmax')(y)

    model = Model(inputs=x, outputs=y)

    return model

