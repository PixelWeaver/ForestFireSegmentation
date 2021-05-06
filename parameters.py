import random
import json
from datetime import datetime


class Parameters:
    def __init__(self, name=None):
        self.epochs = random.choice(list(range(1, 76)))
        self.batch_size = random.choice([16, 32, 64, 128])
        self.learning_rate = random.choice([0.001, 0.0015, 0.002])
        self.input_dim = (256 , 256)
        self.crop_step = 100
        self.name = name

    def to_dict(self):
        return self.__dict__

    def save(self, name=None):
        if name is None:
            name = datetime.now().strftime('%d_%m_%Y_%I_%M_%p')
        
        self.name = name

        with open(f'parameters/{name}.json', 'w') as fp:
            json.dump(self.to_dict(), fp, sort_keys=True, indent=4)

    @staticmethod
    def from_file(name):
        with open(f'parameters/{name}.json', 'r') as fp:
            data = json.load(fp)
            output = Parameters(name)
            for key in output.__dict__.keys():
                if key == "name":
                    continue 
                setattr(output, key, data[key]) 
            return output

    def get_keys(self):
        return self.__dict__.keys()
