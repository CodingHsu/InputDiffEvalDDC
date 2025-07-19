import gc
import json
import os
from pickle import dump

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from datetime import datetime

from DataGenerator import DataGeneratorManager
from dbitnet import DBitNet
from resnet import ResNet


class TrainNeuralNetwork:
    def __init__(self, neural_network_name, data_generator_manager_params, input_size, output_size, word_size,
                 depth=None, hidden_layers=None, num_filters=None, kernel_size=None, reg_param=None,
                 final_activation=None, activation=None, n_add_filters=None, epochs=None,
                 lr_high=0.002, lr_low=0.0001, optimizer='adam', loss=None):
        self.saved_path_conf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_path_conf.json')
        self.trained_net_info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_net_info')
        self.saved_trained_net_dir = self.load_saved_trained_net_dir()
        self.neural_network_name = neural_network_name
        self.neural_network = None
        self.input_size = input_size
        self.output_size = output_size
        self.word_size = word_size
        if depth is None:
            depth = 5
        self.depth = depth
        if hidden_layers is None:
            hidden_layers = [64, 64]
        self.hidden_layers = hidden_layers
        if num_filters is None:
            num_filters = 32
        self.num_filters = num_filters
        if kernel_size is None:
            kernel_size = 3
        self.kernel_size = kernel_size
        if reg_param is None:
            reg_param = 10 ** -5
        self.reg_param = reg_param
        if final_activation is None:
            final_activation = 'sigmoid'
        self.final_activation = final_activation
        if activation is None:
            activation = 'relu'
        self.activation = activation
        if n_add_filters is None:
            n_add_filters = 16
        self.n_add_filters = n_add_filters
        if epochs is None:
            epochs = 20
        self.epochs = epochs
        self.lr_high = lr_high
        self.lr_low = lr_low
        self.optimizer = optimizer
        if loss is None:
            loss = 'mse'
        self.loss = loss
        self.data_generator_manager_params = data_generator_manager_params
        self.data_generator_manager = self.load_data_generator_manager(data_generator_manager_params)
        self.saved_net_path = None

    def load_saved_trained_net_dir(self):
        with open(self.saved_path_conf, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return os.path.abspath(os.path.join(os.path.dirname(__file__), data['default'])) \
                if data['use_default'] else data['path']

    def load_data_generator_manager(self, params):
        return DataGeneratorManager(
            cipher_info_path=params['selected_algorithm'],
            test_feature=params['test_feature'],
            test_rounds=params['test_rounds'],
            num_samples=params['dataset_size'],
            batch_size=params['batch_size']
        )

    def params_to_json(self):
        return {
            "neural_network_name": self.neural_network_name,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "word_size": self.word_size,
            "depth": self.depth,
            "hidden_layers": self.hidden_layers,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "reg_param": self.reg_param,
            "final_activation": self.final_activation,
            "activation": self.activation,
            "n_add_filters": self.n_add_filters,
            "epochs": self.epochs,
            "batch_size": self.data_generator_manager.batch_size,
            "lr_high": self.lr_high,
            "lr_low": self.lr_low,
            "optimizer": self.optimizer,
            "loss": self.loss
        }

    @staticmethod
    def cyclic_lr(num_epochs, high_lr, low_lr):
        return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)

    @staticmethod
    def make_checkpoint(file):
        return ModelCheckpoint(file, monitor='val_loss', save_best_only=True)

    def make_net(self):
        if self.neural_network_name == 'dbitnet':
            model = DBitNet(self.input_size, self.output_size, self.word_size, self.hidden_layers, self.num_filters,
                          self.kernel_size, self.reg_param, self.final_activation, self.n_add_filters)
        else:
            model = ResNet(self.input_size, self.output_size, self.word_size, self.depth, self.hidden_layers,
                           self.num_filters, self.kernel_size, self.reg_param, self.final_activation)
        self.neural_network = model.make_net()

    @staticmethod
    def gen_time_stamp():
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def auto_train(self, gpu_list=None):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        if gpu_list is None:
            gpu_list = []
        if len(gpu_list) <= 1:
            self.make_net()
            self.neural_network.compile(optimizer=self.optimizer, loss=self.loss, metrics=['acc'])
        else:
            strategy = tf.distribute.MirroredStrategy(devices=gpu_list,
                                                      cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            self.data_generator_manager.batch_size *= strategy.num_replicas_in_sync
            with strategy.scope():
                self.make_net()
                self.neural_network.compile(optimizer=self.optimizer, loss=self.loss, metrics=['acc'])
        while True:
            _round = self.data_generator_manager_params['test_rounds'] + 1
            self.train_one_round()
            print("The " + str(_round - 1) + " round test: ")
            test_acc = self.test_neural_network()
            del self.data_generator_manager
            if test_acc < 0.505:
                break
            self.data_generator_manager_params['test_rounds'] = _round
            self.data_generator_manager = self.load_data_generator_manager(self.data_generator_manager_params)
        self.clear_gpu()

    @staticmethod
    def list_gpu():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpus = [x.name for x in gpus if x.device_type == 'GPU']
        list_gpus = []
        for i in range(len(gpus)):
            list_gpus.append('/gpu:' + str(i))
        return list_gpus

    @staticmethod
    def clear_gpu():
        tf.keras.backend.clear_session()  # Clear TensorFlow session
        gc.collect()  # Perform garbage collection to release GPU memory

    def train_net(self, gpu_list=None):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        if gpu_list is None:
            gpu_list = []
        if len(gpu_list) <= 1:
            self.make_net()
            self.neural_network.compile(optimizer=self.optimizer, loss=self.loss, metrics=['acc'])
        else:
            strategy = tf.distribute.MirroredStrategy(devices=gpu_list,
                                                      cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            self.data_generator_manager.batch_size *= strategy.num_replicas_in_sync
            with strategy.scope():
                self.make_net()
                self.neural_network.compile(optimizer=self.optimizer, loss=self.loss, metrics=['acc'])
        try:
            time_stamp = self.gen_time_stamp()
            self.saved_net_path = os.path.join(self.saved_trained_net_dir, time_stamp + '.h5')
            saved_net_history_path = os.path.join(self.saved_trained_net_dir, time_stamp + '_hist.p')

            lr = LearningRateScheduler(self.cyclic_lr(10, self.lr_high, self.lr_low))
            check = self.make_checkpoint(self.saved_net_path)
            callbacks = [lr, check]
            h = self.neural_network.fit(self.data_generator_manager.train_generator, epochs=self.epochs,
                                        validation_data=self.data_generator_manager.val_generator, callbacks=callbacks,
                                        verbose=1)
            dump(h.history, open(saved_net_history_path, 'wb'))
            saved_net_info = self.data_generator_manager.params_to_json()
            saved_net_info.update(self.params_to_json())
            saved_net_info.update({"saved_net_path": self.saved_net_path,
                                   "saved_net_history_path": saved_net_history_path,
                                   "neural_network_name": self.neural_network_name,
                                   "epochs": self.epochs,
                                   "best_val_acc": np.max(h.history["val_acc"])})
            with open(os.path.join(self.trained_net_info_path, time_stamp + '.json'), 'w') as file:
                json.dump(saved_net_info, file)
            return h, self.saved_net_path
        except Exception as e:
            raise ValueError(str(e))

    def train_one_round(self):
        try:
            time_stamp = self.gen_time_stamp()
            self.saved_net_path = os.path.join(self.saved_trained_net_dir, time_stamp + '.h5')
            saved_net_history_path = os.path.join(self.saved_trained_net_dir, time_stamp + '_hist.p')

            lr = LearningRateScheduler(self.cyclic_lr(10, self.lr_high, self.lr_low))
            check = self.make_checkpoint(self.saved_net_path)
            callbacks = [lr, check]
            h = self.neural_network.fit(self.data_generator_manager.train_generator, epochs=self.epochs,
                                        validation_data=self.data_generator_manager.val_generator, callbacks=callbacks,
                                        verbose=1)
            dump(h.history, open(saved_net_history_path, 'wb'))
            saved_net_info = self.data_generator_manager.params_to_json()
            saved_net_info.update(self.params_to_json())
            saved_net_info.update({"saved_net_path": self.saved_net_path,
                                   "saved_net_history_path": saved_net_history_path,
                                   "neural_network_name": self.neural_network_name,
                                   "epochs": self.epochs,
                                   "best_val_acc": np.max(h.history["val_acc"])})
            with open(os.path.join(self.trained_net_info_path, time_stamp + '.json'), 'w') as file:
                json.dump(saved_net_info, file)
            return h
        except Exception as e:
            raise ValueError(str(e))

    def test_neural_network(self):
        try:
            # Ensure the model has been trained
            if self.neural_network is None:
                raise ValueError("The model has not been trained and cannot be tested.")

            # Single GPU or CPU test
            self.neural_network.load_weights(self.saved_net_path)
            test_loss, test_acc = self.neural_network.evaluate(self.data_generator_manager.test_generator, verbose=1)

            print("test_loss: ", test_loss, " test_acc: ", test_acc)
            return test_acc

        except Exception as e:
            raise ValueError(f"An error occurred during testing: {str(e)}")


if __name__ == '__main__':
    # params = {'DataGeneratorManager': {'selected_algorithm': '../ciphers_info/speck_info.json',
    #                                    'test_feature': '00000040', 'test_rounds': 5,
    #                                    'dataset_size': 12500000, 'batch_size': 4096},
    #           'TrainNeuralNetwork': {'neural_network_name': 'resnet', 'input_size': 64, 'output_size': 1,
    #                                  'word_size': 16, 'depth': 5, 'hidden_layers': [64, 64], 'num_filters': 32,
    #                                  'kernel_size': 3, 'reg_param': 1e-05, 'final_activation': None,
    #                                  'activation': None, 'n_add_filters': None, 'epochs': 20, 'loss': 'mse'},
    #           'selected_gpus': ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]}
    # tn = TrainNeuralNetwork(
    #     neural_network_name=params['TrainNeuralNetwork']['neural_network_name'],
    #     data_generator_manager_params=params['DataGeneratorManager'],
    #     input_size=params['TrainNeuralNetwork']['input_size'],
    #     output_size=params['TrainNeuralNetwork']['output_size'],
    #     word_size=params['TrainNeuralNetwork']['word_size'],
    #     depth=params['TrainNeuralNetwork']['depth'],
    #     hidden_layers=params['TrainNeuralNetwork']['hidden_layers'],
    #     num_filters=params['TrainNeuralNetwork']['num_filters'],
    #     kernel_size=params['TrainNeuralNetwork']['kernel_size'],
    #     reg_param=params['TrainNeuralNetwork']['reg_param'],
    #     final_activation=params['TrainNeuralNetwork']['final_activation'],
    #     activation=params['TrainNeuralNetwork']['activation'],
    #     n_add_filters=params['TrainNeuralNetwork']['n_add_filters'],
    #     epochs=params['TrainNeuralNetwork']['epochs'],
    #     loss=params['TrainNeuralNetwork']['loss']
    # )
    # tn.auto_train(params['selected_gpus'])
    from tensorflow.keras.models import load_model
    test_feature = '4008000004000000'
    test_rounds = 6
    params = {'DataGeneratorManager': {'selected_algorithm': '../ciphers_info/des_info.json',
                                       'test_feature': test_feature, 'test_rounds': test_rounds,
                                       'dataset_size': 12500000, 'batch_size': 4096},
              'TrainNeuralNetwork': {'neural_network_name': 'resnet', 'input_size': 128, 'output_size': 1,
                                     'word_size': 32, 'depth': 5, 'hidden_layers': [64, 64], 'num_filters': 32,
                                     'kernel_size': 3, 'reg_param': 1e-05, 'final_activation': None,
                                     'activation': None, 'n_add_filters': None, 'epochs': 40, 'loss': 'mse'},
              'selected_gpus': ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]}
    tn = TrainNeuralNetwork(
        neural_network_name=params['TrainNeuralNetwork']['neural_network_name'],
        data_generator_manager_params=params['DataGeneratorManager'],
        input_size=params['TrainNeuralNetwork']['input_size'],
        output_size=params['TrainNeuralNetwork']['output_size'],
        word_size=params['TrainNeuralNetwork']['word_size'],
        depth=params['TrainNeuralNetwork']['depth'],
        hidden_layers=params['TrainNeuralNetwork']['hidden_layers'],
        num_filters=params['TrainNeuralNetwork']['num_filters'],
        kernel_size=params['TrainNeuralNetwork']['kernel_size'],
        reg_param=params['TrainNeuralNetwork']['reg_param'],
        final_activation=params['TrainNeuralNetwork']['final_activation'],
        activation=params['TrainNeuralNetwork']['activation'],
        n_add_filters=params['TrainNeuralNetwork']['n_add_filters'],
        epochs=params['TrainNeuralNetwork']['epochs'],
        loss=params['TrainNeuralNetwork']['loss']
    )
    _, path = tn.train_net(params['selected_gpus'])


    def sci_notation(x: float, sig=2) -> str:
        s = f"{x:.{sig}e}"  # '5.13e-04'
        mantissa, exp = s.split('e')
        exp = int(exp)  # -4
        return f"{mantissa}×10^{exp}"


    acc_list = []
    for _ in range(5):
        dm = DataGeneratorManager(
            cipher_info_path='../ciphers_info/des_info.json',
            test_feature=test_feature,
            test_rounds=test_rounds,
            num_samples=1000000,
            batch_size=8192,
            train_ratio=0.01,
            val_ratio=0.01,
        )
        net = load_model(path)
        metrics = net.evaluate(dm.test_generator, verbose=0)
        metric_dict = dict(zip(net.metrics_names, metrics))
        acc_list.append(metric_dict.get('acc', np.nan))

    acc_arr = np.array(acc_list, dtype=float)
    acc_mean = float(np.nanmean(acc_arr))
    acc_std = float(np.nanstd(acc_arr, ddof=0))

    formatted_acc = f"({acc_mean:.3f} ± {sci_notation(3 * acc_std)})"
    print(f'{test_feature}, {test_rounds}r, {formatted_acc}')
