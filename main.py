from math import floor
from dataset import Dataset
from parameters import Parameters
from utils import *
from models import UNetModel
import tensorflow as tf

if __name__ == '__main__':
    p = Parameters.from_file("test_run")
    d = Dataset(p)
    m = UNetModel(p)
    m.build(tf.keras.optimizers.Adam)
    m.train(d)