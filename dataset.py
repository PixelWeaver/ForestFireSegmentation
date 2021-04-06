import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from parameters import Parameters
import math
import sqlite3
from tqdm import tqdm


def strip(st : str):
    return st.strip().replace('"', "")


class Dataset:
    def __init__(self, parameters: Parameters):
        self.params = parameters
        self.con = None
        self.cur = None

        Dataset.ensure_dirs_exist()
        self.init_db_if_required()

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

        # Sort/count dataframe
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

    @staticmethod
    def ensure_dirs_exist():
        dir_list = [
            "dataset",
            "dataset/img",
            "dataset/gt",
            "dataset/nir"
        ]

        for path in dir_list:
            if not os.path.isdir(path):
                os.mkdir(path)

    def init_db_if_required(self):
        init_required = False
        if not os.path.isfile("dataset/index.db"):
            init_required = True

        self.con = sqlite3.connect("dataset/index.db")
        self.cur = self.con.cursor()

        if init_required:
            self.cur.execute('''CREATE TABLE data_entries
                                (rowid INTEGER PRIMARY KEY,
                                name TEXT,
                                nir INTEGER,
                                seq INTEGER,
                                fire INTEGER)''')
            print("SQLite database initialized")

    def generate_std_dataset(self):
        metadata_df = pd.read_csv('corsican_fire_db/bdfire/data.csv')
        print("Now deriving samples from images...")
        for i, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
            if i < 636:
                continue
            rgb_path = 'corsican_fire_db/' + strip(row[' "photo"'])
            gt_path = 'corsican_fire_db/' + strip(row[' "verite_terrain"'])
            nir_filename = strip(row[' "photo_IR"'])
            nir_path = None if len(nir_filename) == 0 else 'corsican_fire_db/' + strip(row[' "photo_IR"'])

            self.derive_samples_from_picture(rgb_path, gt_path, nir_path, row['Id'], row[' "sequence"'])
        
        print("Done!")
            
    def add_to_dataset(self, img, gt, nir, name : str, seq : int):
        cv2.imwrite(f"dataset/img/{name}.png", img)
        cv2.imwrite(f"dataset/gt/{name}.png", gt)
        if nir is not None:
            cv2.imwrite(f"dataset/nir/{name}.png", nir)

        self.cur.execute("INSERT INTO data_entries(name, nir, seq, fire) VALUES (?, ?, ?, ?)",
                        (name, 0 if nir is None else 1, seq, 1 if cv2.countNonZero(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)) else 0))
        self.con.commit()

    def derive_samples_from_picture(self, path : str, gt_path : str, nir_path : str, pic_id : str, seq: int):
        # Read all images
        im = cv2.imread(path)
        gt_im = cv2.imread(gt_path)
        if nir_path is not None:
            nir_im = cv2.imread(nir_path)

        # Consistency checks
        if path == gt_path or gt_path == nir_path or nir_path == path:
            raise Exception("Duplicate image path entry")

        dimension_match = im.shape[0:1] == gt_im.shape[0:1]
        if nir_path is not None:
            dimension_match = dimension_match & (gt_im.shape[0:1] == nir_im.shape[0:1])
        if not dimension_match:
            raise Exception("Image dimension mismatch")

        # Crop
        width, height = im.shape[1], im.shape[0]
        target_w, target_h = self.params.input_dim

        w_steps = math.floor((width - target_w)/self.params.crop_step)
        h_steps = math.floor((height - target_h)/self.params.crop_step)
        for i in range(0, h_steps):
            for j in range(0, w_steps):
                h_start, w_start = (i * self.params.crop_step, j * self.params.crop_step) 

                cropped = im[h_start:h_start+target_h, w_start:w_start+target_w]
                cropped_gt = gt_im[h_start:h_start+target_h, w_start:w_start+target_w]
                cropped_nir = None if nir_path is None else nir_im[h_start:h_start+target_h, w_start:w_start+target_w]

                self.add_to_dataset(cropped, cropped_gt, cropped_nir, f"{pic_id}_{i}_{j}", seq)
                