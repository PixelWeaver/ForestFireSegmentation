from dataset import Dataset
from parameters import Parameters
from utils import *
from models import *

if __name__ == '__main__':
    p = Parameters.from_file("att_squeeze_unet_opt")
    d = Dataset(p)
    m = ATTSqueezeUNet(p)
    m.train(d,
        include_val=True,
        save_history=False,
    )
    m.test(d)