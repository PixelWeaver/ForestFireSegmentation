import functools
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

        return (cv2.imread(rgb_path), np.expand_dims(cv2.imread(gt_path, flags=cv2.IMREAD_GRAYSCALE), axis=2)/255, nir_im)

def ensure_folders_exist(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)

def plot_history(name, plot_val=True):
    # Ensure output folder existence
    ensure_folders_exist([
        "figures",
        f"figures/{name}"
    ])

    # Parse and plot all metrics
    with open(f'histories/{name}.json', 'r') as fp:
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
                data=df[columns]
                data.index = range(1,len(data)+1) # Start at epoch 1
                plot = sns.lineplot(data=data, palette="tab10", linewidth=1.5)
                plot.set(yscale='log')
                plot.set(ylabel=key)
                plot.set(xlabel='epoch')
                plt.tight_layout()
                plt.savefig(f"figures/{name}/{key}")
                plt.figure()

def cmp_plot_history(names : "list[str]", plot_val=True):
    plots = {} # there will be as many plots as there are different metrics in the history files

    resulting_name = functools.reduce(lambda a, b : a + "_" + b, sorted(names)) # Compute output folder name

    # Ensure folder existence
    ensure_folders_exist([
        "figures",
        f"figures/cmp_{resulting_name}"
    ])

    # Parse histories and add to plot dict(key = metric)
    for name in names:
        with open(f'histories/{name}.json', 'r') as fp:
                data = json.load(fp)

                keys = data.keys()
                if plot_val:
                    keys = list(keys)[:int(len(keys)/2)]

                for key in keys:
                    # Ensure dataframe had been initialized
                    if key not in plots:
                        plots[key] = pd.DataFrame()
                            
                    # Add data to dataframe
                    plots[key][name] = pd.Series(data[key])
                    if plot_val:
                        plots[key][f"val_{name}"] = pd.Series(data[f"val_{key}"])
    
    # Plot all metrics
    for key in plots.keys():
        plots[key].index = range(1,len(plots[key])+1) # Start at epoch 1
        plot = sns.lineplot(data=plots[key], palette="tab10", linewidth=1.5)
        plot.set(yscale='log')
        plot.set(ylabel=key)
        plot.set(xlabel='epoch')
        plt.tight_layout()
        plt.savefig(f"figures/cmp_{resulting_name}/{key}")
        plt.figure()
