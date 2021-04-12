from dataset import Dataset
from parameters import Parameters
from utils import *
from models import UNetModel
import tensorflow as tf

if __name__ == '__main__':
    p = Parameters.from_file("test_run")
    d = Dataset(p)
    #d.generate_std_dataset()
    #d.generate_split()
    m = UNetModel(p)
    m.build(tf.keras.optimizers.Adam)
    m.train(*d.get_reduced_dataset())
