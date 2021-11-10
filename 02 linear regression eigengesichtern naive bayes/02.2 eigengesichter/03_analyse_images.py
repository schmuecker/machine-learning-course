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

# Flatten images
image_pixels = images
for idx, image in enumerate(images):
    image_pixels[idx] = []
    for row in image:
        for pixel in row:
            image_pixels[idx].append(pixel)

# Design matrix
design_matrix = pd.DataFrame(image_pixels)

# PCA
U, D, V = pca(design_matrix)
n = U.shape[0]

d = pd.DataFrame(D, columns=["SingularValue"])
d["EigenValue"] = d["SingularValue"] ** 2
d["Varianz"] = d["SingularValue"] / (n-1)
d["AnteilVarianz%"] = d["Varianz"] / (d["Varianz"].sum()) * 100
d["Kumuliert%"] = d["AnteilVarianz%"].cumsum()
d["Fehler"] = 100 - d["Kumuliert%"]
print(d)
