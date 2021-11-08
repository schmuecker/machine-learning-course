import sys
import pathlib
import shutil
from os import listdir
from datetime import datetime
from os.path import isfile, join, dirname, realpath, basename, exists
from skimage.io import imread, imsave
from skimage.util import crop
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def flatten(t):
    return [item for sublist in t for item in sublist]


def pca(df: pd.DataFrame):

    # 1. Zentrierung
    for col in df.columns:
        df[col] = df[col] - df[col].mean()

    # 2. Normierung, z-Transformation damit Varianz immer 1 ist
    df = df.apply(stats.zscore)

    # 3. DF zu n * d Matrix X umwandeln
    X = df.to_numpy()

    # 4.-7. Eigenwertproblem
    U, D, V = np.linalg.svd(X, full_matrices=False)

    return U, D, V


# Get paths
script_path = dirname(realpath(sys.argv[0]))
people_path = join(script_path, 'data-processed/test/')


person_folders = listdir(people_path)
images = []
for person in person_folders:
    path = join(people_path, person)
    files = listdir(path)
    for file in files:
        path = join(path, file)
        image = imread(path)
        images.append(image)

images_flattened = np.ndarray.flatten(images)
images_flattened_2 = []
for image in images_flattened:
    print(np.ndarray.flatten(image))
    images_flattened_2.append(np.ndarray.flatten(image))

# print(images_flattened[0])

# Design matrix
# design_matrix = pd.DataFrame(images_flattened)

# pca_result = pca(design_matrix)
# print(pca_result)
