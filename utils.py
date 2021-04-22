import math
import cv2
import numpy as np

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

