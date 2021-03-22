import sqlite3
import random
import numpy as np
import math
import os
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def row_to_resolution(row):
    return int(row["width"]) * int(row["height"])


def row_to_size_string(row):
    return f'{row["width"]}x{row["height"]}'


def strip(st):
    return st.strip().replace('"', "")


class Dataset:
    def parse_cfdb(self):
        print("corsican_fire_db contains")

    @staticmethod
    def print_cdfb_statistics():
        metadata_df = pd.read_csv('corsican_fire_db/bdfire/data.csv')
        print(f'Total number of pictures: {len(metadata_df)}')
        df = pd.DataFrame([], columns=['width', 'height'])
        for i, row in metadata_df.iterrows():
            im = cv2.imread('corsican_fire_db/' + strip(row[' "photo"']))
            df = df.append({'width': im.shape[1], 'height': im.shape[0]}, ignore_index=True)

        df["resolution"] = df.apply(lambda row: int(row["width"]) * int(row["height"]), axis=1)
        df["size_string"] = df.apply(row_to_size_string, axis=1)

        # sort dataframe
        sorted_df = df.sort_values(by=['resolution', 'size_string'], ascending=[True, True])
        counted_df = sorted_df.groupby(['size_string', 'resolution'], sort=False)["size_string"]\
            .agg(['count'])
        print(len(counted_df))

        # Initialize the matplotlib figure
        f, ax = plt.subplots(figsize=(15, 6))

    def __init__(self):
        print(2)
