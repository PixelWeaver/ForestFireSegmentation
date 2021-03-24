import sqlite3
import random
import json


class Parameters:
    def __init__(self):
        conn = sqlite3.connect('samples.db')
        c = conn.cursor()
        self.min_seq_size = c.execute('SELECT MIN(count) FROM (SELECT COUNT(*) AS count FROM snapshots GROUP BY sample_id)').fetchone()[0]
        conn.close()

        self.seq_size = random.choice(list(range(5, self.min_seq_size + 1)))
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

    def to_dict(self):
        return {
            "seq_size": self.seq_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "hidden_count": self.hidden_count,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "units": self.units,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "bias_l1": self.bias_l1,
            "bias_l2": self.bias_l2,
            "recurrent_l1": self.recurrent_l1,
            "recurrent_l2": self.recurrent_l2
        }

    def to_file(self):
        with open('parameters/parameters.json', 'w') as fp:
            json.dump(self.to_dict(), fp, sort_keys=True, indent=4)

    @staticmethod
    def from_file():
        with open('parameters/parameters.json', 'r') as fp:
            data = json.load(fp)
            output = Parameters()
            output.seq_size = data["seq_size"]
            output.epochs = data["epochs"]
            output.batch_size = data["batch_size"]
            output.hidden_count = data["hidden_count"]
            output.optimizer = data["optimizer"]
            output.learning_rate = data["learning_rate"]
            output.units = data["units"]
            output.dropout = data["dropout"]
            output.recurrent_dropout = data["recurrent_dropout"]
            output.bias_l1 = data["bias_l1"]
            output.bias_l2 = data["bias_l2"]
            output.recurrent_l1 = data["recurrent_l1"]
            output.recurrent_l2 = data["recurrent_l2"]
            return output

    @staticmethod
    def get_keys():
        return [
            "seq_size",
            "epochs",
            "batch_size",
            "hidden_count",
            "optimizer",
            "learning_rate",
            "units",
            "dropout",
            "recurrent_dropout",
            "bias_l1",
            "bias_l2",
            "recurrent_l1",
            "recurrent_l2"
        ]