import json
import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_venn_circle_height(num):
    baseHeight = 20
    unit_a = math.pow(baseHeight / 2, 2) * math.pi / 42
    output_a = unit_a * (42 + math.pow(num - 42, 1.1))
    output_r = math.sqrt(output_a / math.pi)
    return output_r * 2

def paths_from_name(name):
    return (
        f"dataset/img/{name}.png",
        f"dataset/gt/{name}.png",
        f"dataset/nir/{name}.png"
    )

def load_row(row):
        rgb_path, gt_path, nir_path = paths_from_name(row[1])

        nir_im = None
        if nir_path is not None:
            nir_im = cv2.imread(nir_path)

        return cv2.imread(rgb_path), np.expand_dims(cv2.imread(gt_path, flags=cv2.IMREAD_GRAYSCALE), axis=2)/255, nir_im 

def plot_history(name, plot_val=True):
    with open(f'histories/{name}.json', 'r') as fp:
            dir_list = [
                "figures",
                f"figures/{name}"
            ]

            for path in dir_list:
                if not os.path.isdir(path):
                    os.mkdir(path)

            data = json.load(fp)

            df = pd.DataFrame()
            for key in data.keys():
                df[key] = pd.Series(data[key])

            keys = data.keys()
            if plot_val:
                keys = list(keys)[:int(len(keys)/2)]

            for key in keys:
                columns = [key]
                if plot_val:
                    columns.append(f"val_{key}")
                sns.lineplot(data=df[columns], palette="tab10", linewidth=2.5)
                plt.tight_layout()
                plt.savefig(f"figures/{name}/{key}")
                plt.figure()