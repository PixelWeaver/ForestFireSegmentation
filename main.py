from dataset import Dataset
from parameters import Parameters
from utils import *
from models import *
import tensorflow as tf

if __name__ == '__main__':
    p = Parameters.from_file("deeplab_v3")
    d = Dataset(p)
    m = DeepLabV3Plus(p)
    m.summarize()
    # m.train(d)
    # m.prediction_test(d)