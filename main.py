from dataset import Dataset
from parameters import Parameters
from utils import *
from models import UNetModel
import tensorflow as tf

if __name__ == '__main__':
    p = Parameters.from_file("flame_unet_1")
    d = Dataset(p)
    m = UNetModel(p)
    m.load_trained()
    m.prediction_test(d)
