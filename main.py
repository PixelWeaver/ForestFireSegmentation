from dataset import Dataset
from parameters import Parameters
from utils import *
from models import *

if __name__ == '__main__':
    p = Parameters.from_file("squeeze_unet_opt")
    d = Dataset(p)
    m = SqueezeUNet(p)
    m.train(d,
        save_history=False,
        save_model=True,
        include_val=True)
    m.test(d)