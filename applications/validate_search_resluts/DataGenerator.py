import json
from tensorflow.keras.utils import Sequence
import numpy as np
from DatasetGenerate import DatasetGenerate


class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        x = self.data[start_index:end_index]
        y = self.labels[start_index:end_index]

        return x, y

    def on_epoch_end(self):
        pass


class DataGeneratorManager:
    def __init__(self, cipher_info_path, test_feature, test_rounds, num_samples,
                 batch_size, train_ratio=0.8, val_ratio=0.1):
        self.cipher_info_path = cipher_info_path
        self.cipher_info = self.load_cipher_info()
        self.test_feature = test_feature
        self.test_rounds = test_rounds
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.DatasetGenerate = self.load_dataset_generate(self.cipher_info)
        self.data, self.labels = self.generate_dataset()
        self.train_data, self.val_data, self.test_data = self.split_data()

        self._train_generator = None
        self._val_generator = None
        self._test_generator = None

    def load_cipher_info(self):
        if self.cipher_info_path is None:
            return None
        with open(self.cipher_info_path, 'r', encoding='utf-8') as file:
            info = json.load(file)
        return info

    @staticmethod
    def load_dataset_generate(cipher_info):
        if cipher_info is None:
            return None
        else:
            return DatasetGenerate()

    def generate_dataset(self):
        x, y = self.DatasetGenerate.generate_differential_dataset(
            self.cipher_info['saved_path'], self.cipher_info['data_type'],
            self.cipher_info['plaintext_size'], self.cipher_info['ciphertext_size'],
            self.cipher_info['master_key_size'], self.test_feature, self.test_rounds,
            self.num_samples, self.batch_size
        )
        data = convert_to_binary(np.concatenate(x, axis=1).transpose(),
                                 data_type_to_word_size(self.cipher_info['data_type']))
        return data, y

    def split_data(self):
        # Split dataset into training, validation, and test sets
        total_samples = len(self.data)
        train_end = int(total_samples * self.train_ratio)
        val_end = int(total_samples * (self.train_ratio + self.val_ratio))

        train_data = (self.data[:train_end], self.labels[:train_end])
        val_data = (self.data[train_end:val_end], self.labels[train_end:val_end])
        test_data = (self.data[val_end:], self.labels[val_end:])

        self.batch_size = min(self.batch_size, min(len(train_data[0]), len(val_data[0]), len(test_data[0])) // 20)
        return train_data, val_data, test_data

    @property
    def train_generator(self):
        if self._train_generator is None:
            self._train_generator = DataGenerator(self.train_data[0], self.train_data[1], self.batch_size)
        return self._train_generator

    @property
    def val_generator(self):
        if self._val_generator is None:
            self._val_generator = DataGenerator(self.val_data[0], self.val_data[1], self.batch_size)
        return self._val_generator

    @property
    def test_generator(self):
        if self._test_generator is None:
            self._test_generator = DataGenerator(self.test_data[0], self.test_data[1], self.batch_size)
        return self._test_generator

    def params_to_json(self):
        return {
            "cipher_info": self.cipher_info,
            "test_feature": self.test_feature,
            "test_rounds": self.test_rounds,
            "num_samples": self.num_samples,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
        }


def convert_to_binary(arr, word_size):
    x = np.zeros((len(arr) * word_size, len(arr[0])), dtype=np.uint8)
    for i in range(len(arr) * word_size):
        index = i // word_size
        offset = word_size - (i % word_size) - 1
        x[i] = (arr[index] >> offset) & 1
    return x.transpose()


def data_type_to_word_size(data_type):
    if data_type == 'uint8':
        return 8
    elif data_type == 'uint16':
        return 16
    elif data_type == 'uint32':
        return 32
    elif data_type == 'uint64':
        return 64
    else:
        raise ValueError('Unsupported data type')
