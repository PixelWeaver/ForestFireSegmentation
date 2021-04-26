import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from parameters import Parameters
import math
import sqlite3
from tqdm import tqdm
import random
import shutil
import numpy as np
from generator import Generator
from utils import paths_from_name, load_row

def strip(st : str):
    return st.strip().replace('"', "")


class Dataset:
    # Class variables
    r_seed = 1319181 # Fixed seed to ensure reproducibility

    def __init__(self, parameters: Parameters):
        self.params = parameters
        self.con = None
        self.cur = None

        Dataset._ensure_dirs_exist()
        self._init_db_if_required()

        self.train_rows = list(self.con.execute("SELECT * FROM data_entries WHERE split = 0")) # training
        self.val_rows = list(self.con.execute("SELECT * FROM data_entries WHERE split = 1")) # validation
        self.test_rows = list(self.con.execute("SELECT * FROM data_entries WHERE split = 2")) # test

        rdEngine = random.Random(Dataset.r_seed)
        rdEngine.shuffle(self.train_rows)
        rdEngine.shuffle(self.val_rows)
        rdEngine.shuffle(self.test_rows)

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
    def _ensure_dirs_exist():
        dir_list = [
            "dataset",
            "dataset/img",
            "dataset/gt",
            "dataset/nir",
            "discarded",
            "models",
            "tests",
            "histories",
            "predictions"
        ]

        for path in dir_list:
            if not os.path.isdir(path):
                os.mkdir(path)

    def cp_discarded_samples(self, lower_bound):
        samples = list(self.cur.execute(f"SELECT rowid, name FROM data_entries WHERE fire_pixels > 0 AND fire_pixels < {lower_bound}"))
        print(f"{len(samples)} samples would be discarded")
        for i, sample in enumerate(samples):
            path, _, _ = paths_from_name(sample[1])
            shutil.copy(path, f"discarded/{i}.png")

    def get_train_gen(self):
        return Generator(self.params, self.train_rows)

    def get_train_val_gen(self):
        return Generator(self.params, self.train_rows + self.val_rows)

    def get_val_gen(self):
        return Generator(self.params, self.val_rows)

    def get_test_gen(self):
        return Generator(self.params, self.test_rows)

    def load_specific_ids(self, ids):
        rows = []
        for id in ids:
            rows.extend(list(self.cur.execute(f"SELECT rowid, name FROM data_entries WHERE rowid = {id}")))

        samples = np.zeros((len(ids), self.params.input_dim[0], self.params.input_dim[1], 3))
        for i, row in enumerate(rows):
            x, _, _ = load_row(row)
            samples[i] = x

        return samples


    def _count_fire_pixels(self):
        print("Counting fire pixels")
        samples = list(self.cur.execute("SELECT rowid, name FROM data_entries"))
        
        for sample in tqdm(samples):
            _, gt_path, _ = paths_from_name(sample[1])
            im = cv2.imread(gt_path)
            fpixels = cv2.countNonZero(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            
            self.cur.execute(f"UPDATE data_entries SET fire_pixels = {fpixels} WHERE rowid = {sample[0]}")
        self.con.commit()

    def print_firepixel_distribution(self):
        df = pd.read_sql_query("SELECT fire_pixels FROM data_entries", self.con)
        sns.displot(x="fire_pixels", data=df, kind="ecdf")
        plt.tight_layout()
        plt.savefig("figures/dataset_firepixels_repartition")
        plt.show()


    def _init_db_if_required(self):
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
                                fire_pixels INTEGER,
                                fire INTEGER,
                                split INTEGER)''')
            print("SQLite database initialized")

            self._generate_std_dataset()
            self._count_fire_pixels()
            self._generate_split()

    def _generate_std_dataset(self):
        metadata_df = pd.read_csv('corsican_fire_db/bdfire/data.csv')
        print("Now deriving samples from images...")
        for i, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
            rgb_path = 'corsican_fire_db/' + strip(row[' "photo"'])
            gt_path = 'corsican_fire_db/' + strip(row[' "verite_terrain"'])
            nir_filename = strip(row[' "photo_IR"'])
            nir_path = None if len(nir_filename) == 0 else 'corsican_fire_db/' + strip(row[' "photo_IR"'])

            self._derive_samples_from_picture(rgb_path, gt_path, nir_path, row['Id'], row[' "sequence"'])
        
    def _generate_split(self):
        # negative_samples = list(map(lambda r : r[0], self.cur.execute("SELECT rowid FROM data_entries WHERE fire = 0")))
        # Only include positive samples for segmentation
        positive_samples = list(map(lambda r : r[0], self.cur.execute("SELECT rowid FROM data_entries WHERE fire_pixels > 19")))
        
        rdEngine = random.Random(Dataset.r_seed)
        rdEngine.shuffle(positive_samples)

        train = positive_samples[:math.ceil(len(positive_samples) * 0.7)]
        val = positive_samples[math.ceil(len(positive_samples) * 0.7):math.ceil(len(positive_samples) * 0.85)]
        test = positive_samples[math.ceil(len(positive_samples) * 0.85):]

        self.cur.executemany("UPDATE data_entries SET split = 0 WHERE rowid = ?", zip(iter(train)))
        self.cur.executemany("UPDATE data_entries SET split = 1 WHERE rowid = ?", zip(iter(val)))
        self.cur.executemany("UPDATE data_entries SET split = 2 WHERE rowid = ?", zip(iter(test)))
        self.con.commit()

    def get_reduced_dataset(self):
        x_shape = (200, self.params.input_dim[0], self.params.input_dim[1], 3)
        y_shape = (200, self.params.input_dim[0], self.params.input_dim[1], 1)
        X_train, y_train, X_val, y_val = np.zeros(x_shape), np.zeros(y_shape), np.zeros(x_shape), np.zeros(y_shape)

        train_rows = list(self.con.execute("SELECT * FROM data_entries WHERE split = 0"))[:200] # training
        val_rows = list(self.con.execute("SELECT * FROM data_entries WHERE split = 1"))[:200] # val

        for i in range(200):
            rgb, gt, _ = load_row(train_rows[i])
            X_train[i] = rgb
            y_train[i] = gt

            rgb, gt, _ = load_row(val_rows[i])
            X_val[i] = rgb
            y_val[i] = gt

        return X_train, y_train, X_val, y_val 
            
    def _add_to_dataset(self, img, gt, nir, name : str, seq : int):
        cv2.imwrite(f"dataset/img/{name}.png", img)
        cv2.imwrite(f"dataset/gt/{name}.png", gt)
        if nir is not None:
            cv2.imwrite(f"dataset/nir/{name}.png", nir)

        self.cur.execute("INSERT INTO data_entries(name, nir, seq, fire) VALUES (?, ?, ?, ?)",
                        (name, 0 if nir is None else 1, seq, 1 if cv2.countNonZero(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)) else 0))
        self.con.commit()

    def _derive_samples_from_picture(self, path : str, gt_path : str, nir_path : str, pic_id : str, seq: int):
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

                self._add_to_dataset(cropped, cropped_gt, cropped_nir, f"{pic_id}_{i}_{j}", seq)
                