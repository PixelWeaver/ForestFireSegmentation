import csv
import os
from parameters import Parameters


class MetricWriter:
    def __init__(self):
        self.file = None
        self.writer = None
        self.headers = None
        self.should_write_header = not os.path.isfile("results.csv")

    def write(self, m, p):
        if self.file is None:
            self.headers = list(m.keys()) + list(p.get_keys())
            self.file = open("results.csv", "w")
            self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
            if self.should_write_header:
                self.writer.writeheader()

        self.writer.writerow({**m, **p.to_dict()})