from abstract_model import AbstractModel
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Reshape, Permute, Conv1D, BatchNormalization, Activation, \
    Add, Flatten, Dense
from tensorflow.keras.models import Model


class ResNet(AbstractModel):
    def __init__(self, input_size, output_size, word_size, depth=5, hidden_layers=None, num_filters=32,
                 kernel_size=3, reg_param=10 ** -5, final_activation='sigmoid'):
        super(ResNet, self).__init__(input_size=input_size, output_size=output_size, word_size=word_size)
        if hidden_layers is None:
            hidden_layers = [64, 64]
        self.net_name = 'resnet'
        self.depth = depth
        self.hidden_layers = hidden_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.reg_param = reg_param
        self.final_activation = final_activation

    def make_net(self):
        conv = Conv1D
        inp = Input(shape=(self.input_size,))
        rs = Reshape((self.input_size // self.word_size, self.word_size))(inp)
        perm = Permute((2, 1))(rs)
        conv0 = conv(self.num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(self.reg_param))(perm)
        conv0 = BatchNormalization()(conv0)
        conv0 = Activation('relu')(conv0)
        shortcut = conv0
        for i in range(self.depth):
            conv1 = conv(self.num_filters, kernel_size=self.kernel_size, padding='same',
                         kernel_regularizer=l2(self.reg_param))(shortcut)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)
            conv2 = conv(self.num_filters, kernel_size=self.kernel_size, padding='same',
                         kernel_regularizer=l2(self.reg_param))(conv1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Activation('relu')(conv2)
            shortcut = Add()([shortcut, conv2])
        dense = Flatten()(shortcut)
        for hidden_layer in self.hidden_layers:
            dense = Dense(hidden_layer, kernel_regularizer=l2(self.reg_param))(dense)
            dense = BatchNormalization()(dense)
            dense = Activation('relu')(dense)
        out = Dense(self.output_size, activation=self.final_activation, kernel_regularizer=l2(self.reg_param))(dense)
        model = Model(inputs=inp, outputs=out)
        return model
