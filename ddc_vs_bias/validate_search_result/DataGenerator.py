from tensorflow.keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):
    def __init__(self, cipher, num_samples, batch_size, num_rounds, diff):
        self.cipher = cipher
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.diff = diff
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        X, Y = self.cipher.make_train_data(self.batch_size, self.num_rounds, self.diff)
        return X, Y

    def on_epoch_end(self):
        pass