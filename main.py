from dataset import Dataset
from parameters import Parameters
from utils import *
from models import *

if __name__ == '__main__':
    p = Parameters.from_file("squeeze_unet_opt")
    d = Dataset(p)
    m = SqueezeUNet(p)
    m.load_trained()
    m.prediction_test(d)