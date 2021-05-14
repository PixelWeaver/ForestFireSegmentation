from dataset import Dataset
from parameters import Parameters
from utils import *
from models import *

if __name__ == '__main__':
    p = Parameters.from_file("dlv3_efficientnet_2")
    d = Dataset(p)
    m = DLV3P_EfficientNet_2(p)
    m.train(d)
    m.prediction_test(d)