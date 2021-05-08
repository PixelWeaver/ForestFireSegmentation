from dataset import Dataset
from parameters import Parameters
from utils import *
from models import *

if __name__ == '__main__':
    p = Parameters.from_file("deeplab_v3")
    d = Dataset(p)
    m = DeepLabV3Plus(p)
    m.train(d)
    #m.prediction_test(d)