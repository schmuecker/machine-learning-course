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
from sklearn import decomposition
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


def load_images(path: str):
    script_path = dirname(realpath(sys.argv[0]))
    abs_path = join(script_path, path)

    # Load images
    person_folders = listdir(abs_path)
    images = []
    for person in person_folders:
        path = join(abs_path, person)
        files = listdir(path)
        for file in files:
            filePath = join(path, file)
            image = imread(filePath)
            images.append(image)

    # Flatten images to pixels
    images_pixels = images
    for idx, image in enumerate(images):
        images_pixels[idx] = []
        for row in image:
            for pixel in row:
                images_pixels[idx].append(pixel)

    return images_pixels


# Load pixels from images
images_pixels_t = load_images('data-processed/training/')
images_pixels_te = load_images('data-processed/test/')

# Design matrix
design_matrix = pd.DataFrame(images_pixels_t)

# PCA
U, D, V = pca(design_matrix)
n = U.shape[0]

pca_result = pd.DataFrame(D, columns=["SingularValue"])
pca_result["EigenValue"] = pca_result["SingularValue"] ** 2
pca_result["Varianz"] = pca_result["SingularValue"] / (n-1)
pca_result["AnteilVarianz%"] = pca_result["Varianz"] / \
    (pca_result["Varianz"].sum()) * 100
pca_result["Kumuliert%"] = pca_result["AnteilVarianz%"].cumsum()
pca_result["Fehler"] = 100 - pca_result["Kumuliert%"]
pca_result["Index"] = pca_result.index
print(pca_result)

# Plot Eigenvalues
# pca_result.head(150).plot.scatter(x=6, y=1)
# plt.show()

sk_pca = decomposition.PCA(n_components=150, whiten=True)
sk_pca.fit(images_pixels_t)
# plt.imshow(sk_pca.mean_.reshape(32, 32),
#            cmap=plt.cm.bone)
# plt.show()

fig = plt.figure(figsize=(16, 6))
for i in range(12):
    ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(sk_pca.components_[i].reshape(32, 32),
              cmap=plt.cm.bone)
plt.show()

# Interpretation:
# - first 4 pc's are lighting conditions (dark, left, bright, right)
# - other images show specific features of the faces

# 2d)

# Centralize test data
# for col in df.columns:
#     df[col] = df[col] - df[col].mean()
