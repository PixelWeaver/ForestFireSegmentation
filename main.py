from dataset import Dataset
from parameters import Parameters
from utils import *

if __name__ == '__main__':
    # Dataset.generate_rgb_split()
    p = Parameters()
    d = Dataset(p)
    d.generate_std_dataset()

