from dataset import Dataset
from parameters import Parameters
from utils import *
from models import UNetModel
import tensorflow as tf

if __name__ == '__main__':
    p = Parameters()
    d = Dataset(p)
    # d.cp_discarded_samples(10)
    m = UNetModel(p)
    m.build(tf.keras.optimizers.Adam)
    m.train(*d.get_reduced_dataset())
