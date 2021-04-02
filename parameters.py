import random
import json
from datetime import datetime


class Parameters:
    def __init__(self):
        self.epochs = random.choice(list(range(1, 76)))
        self.batch_size = random.choice([16, 32, 64, 128])
        self.hidden_count = random.choice([1, 2, 3])
        self.optimizer = random.choice([0, 1, 2, 3])
        self.learning_rate = random.choice([0.001, 0.0015, 0.002])
        self.units = random.choice(list(range(0, 201)))
        self.dropout = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.recurrent_dropout = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.bias_l1 = random.choice([0, 0.01, 0.02])
        self.bias_l2 = random.choice([0, 0.01, 0.02])
        self.recurrent_l1 = random.choice([0, 0.01, 0.02])
        self.recurrent_l2 = random.choice([0, 0.01, 0.02])
        self.loss = "binary_crossentropy"
        self.input_dim = (200, 200)

    def to_dict(self):
        return self.__dict__

    def save(self, name=None):
        if name is None:
            name = datetime.now().strftime('%d_%m_%Y_%I_%M_%p')

        with open(f'parameters/{name}.json', 'w') as fp:
            json.dump(self.to_dict(), fp, sort_keys=True, indent=4)

    @staticmethod
    def from_file(name):
        with open(f'parameters/{name}.json', 'r') as fp:
            data = json.load(fp)
            output = Parameters()
            for key in output.__dict__.keys():
                setattr(output, key, data[key]) 
            return output

    def get_keys(self):
        return self.__dict__.keys()
