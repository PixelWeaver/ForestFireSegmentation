import tensorflow as tf
import numpy as np
from parameters import Parameters
from dataset import Dataset
import math

class Generator(tf.keras.utils.Sequence):
    def __init__(self, p : Parameters, rows):
        self.params = p
        self.rows = rows
        self.on_epoch_end()

    def __len__(self):
        return math.floor(len(self.rows) / self.params.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.params.batch_size:(index+1)*self.params.batch_size]
        batch_of_rows = [self.rows[k] for k in indexes]
        X, y = self.__data_generation(batch_of_rows)

        return X, y

    def __data_generation(self, rows):
        X = np.empty((self.params.batch_size, *self.params.input_dim, 3))
        y = np.empty((self.params.batch_size, *self.params.input_dim, 1))

        for i, row in enumerate(rows):
            X[i,], _, y[i,] = Dataset.load_row(row)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.rows))
        np.random.shuffle(self.indexes)

    