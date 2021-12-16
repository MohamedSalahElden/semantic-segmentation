import pandas as pd
import numpy as np
import os


classes = pd.read_csv("data/classes.csv", index_col='name')


cls2rgb = {cl: list(classes.loc[cl, :]) for cl in classes.index}
idx2rgb = {idx: np.array(rgb)
           for idx, (cl, rgb) in enumerate(cls2rgb.items())}


def map_class_to_rgb(p):
    return idx2rgb[p[0]]


def adjust_mask(mask, flat=False):

    semantic_map = []
    for colour in list(cls2rgb.values()):
        equality = np.equal(mask, colour)  # 256x256x3 with True or False
        # 256x256 If all True, then True, else False
        class_map = np.all(equality, axis=-1)
        # List of 256x256 arrays, map of True for a given found color at the pixel, and False otherwise.
        semantic_map.append(class_map)
    # 256x256x32 True only at the found color, and all False otherwise.
    semantic_map = np.stack(semantic_map, axis=-1)
    if flat:
        semantic_map = np.reshape(semantic_map, (-1, 128*128))

    return np.float16(semantic_map)  # convert to numbers
