from abstract_model import AbstractModel

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.models import Model


class DBitNet(AbstractModel):
    def __init__(self, input_size, output_size, word_size, hidden_layers=None, num_filters=32, kernel_size=3,
                 reg_param=10 ** -5, final_activation='sigmoid', n_add_filters=16):
        super(DBitNet, self).__init__(input_size=input_size, output_size=output_size, word_size=word_size)
        if hidden_layers is None:
            hidden_layers = [256, 256, 64]
        self.hidden_layers = hidden_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.reg_param = reg_param
        self.final_activation = final_activation
        self.n_add_filters = n_add_filters

    @staticmethod
    def get_dilation_rates(input_size):
        drs = []
        while input_size >= 8:
            drs.append(int(input_size / 2 - 1))
            input_size = input_size // 2
        return drs

    def make_net(self):
        dilation_rates = self.get_dilation_rates(self.input_size)
        inp = Input(shape=(self.input_size, 1))
        x = tf.keras.layers.Lambda(lambda x: (x - 0.5) / 0.5)(inp)
        for dilation_rate in dilation_rates:
            x = Conv1D(filters=self.num_filters, kernel_size=2, padding='valid', dilation_rate=dilation_rate,
                       strides=1, activation='relu')(x)
            x = BatchNormalization()(x)
            x_skip = x
            x = Conv1D(filters=self.num_filters, kernel_size=2, padding='causal', dilation_rate=1, activation='relu')(x)
            x = Add()([x, x_skip])
            x = BatchNormalization()(x)
            self.num_filters += self.n_add_filters
        dense = Flatten()(x)
        for hidden_layer in self.hidden_layers:
            dense = Dense(hidden_layer, kernel_regularizer=l2(self.reg_param))(dense)
            dense = BatchNormalization()(dense)
            dense = Activation('relu')(dense)
        out = Dense(self.output_size, activation=self.final_activation, kernel_regularizer=l2(self.reg_param))(dense)
        model = Model(inp, out)
        return model

