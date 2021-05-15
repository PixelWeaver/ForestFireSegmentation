from dataset import Dataset
from parameters import Parameters
from utils import *
from models import *

if __name__ == '__main__':
    p = Parameters.from_file("dlv3_efficientnet")
    d = Dataset(p)
    m = DLV3P_EfficientNet(p)
    m.train(d)