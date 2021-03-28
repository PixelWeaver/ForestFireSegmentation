import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np


def strip(st):
    return st.strip().replace('"', "")


class Dataset:
    def __init__(self):
        pass

    @staticmethod
    def generate_rgb_split():
        metadata_df = pd.read_csv('corsican_fire_db/bdfire/data.csv')

        # Push every input image name in a list
        X = []
        for i, row in metadata_df.iterrows():
            X.append(strip(row[' "photo"']))

        # Push every output (GT) image name in a list
        y = []
        for i, row in metadata_df.iterrows():
            y.append(strip(row[' "verite_terrain"']))

        # Split & shuffle it
        # random_state for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1319181)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1319181)

        if not os.path.isdir("split"):
            os.mkdir("split")

        np.savez_compressed("split/train.npz", X=X_train, y=y_train)
        np.savez_compressed("split/val.npz", X=X_val, y=y_val)
        np.savez_compressed("split/test.npz", X=X_test, y=y_test)
        print(f'Splits generated')

    @staticmethod
    def print_cdfb_statistics():
        metadata_df = pd.read_csv('corsican_fire_db/bdfire/data.csv')
        print(f'Total number of pictures: {len(metadata_df)}')

        # Read every image's resolution
        df = pd.DataFrame([], columns=['width', 'height'])
        for i, row in metadata_df.iterrows():
            im = cv2.imread('corsican_fire_db/' + strip(row[' "photo"']))
            df = df.append({'width': im.shape[1], 'height': im.shape[0]}, ignore_index=True)

        df["resolution"] = df.apply(lambda row: int(row["width"]) * int(row["height"]), axis=1)
        df["size_string"] = df.apply(lambda row: f'{row["width"]}x{row["height"]}', axis=1)

        # sort/count dataframe
        sorted_df = df.sort_values(by=['resolution', 'size_string'], ascending=[True, True])
        counted_df = sorted_df.groupby(['size_string', 'resolution'], sort=False, as_index=False)["size_string"]\
            .agg(['count'])
        print(f'Number of different sizes: {len(counted_df)}')

        # Extract top 10 counts
        top_10 = counted_df.sort_values(by=['count'], ascending=[False]).head(10)

        # Plot it
        bars = sns.barplot(x="size_string", y="count", data=top_10.reset_index())
        bars.set_xticklabels(top_10.reset_index()['size_string'], rotation=90)
        print(top_10)
        plt.tight_layout()
        plt.savefig("figures/dataset_img_size_top10")
        plt.show()

