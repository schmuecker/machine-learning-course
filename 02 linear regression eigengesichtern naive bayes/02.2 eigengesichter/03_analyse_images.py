import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from skimage.io import imread
from sklearn import decomposition
from os.path import join, dirname, realpath


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
    images_pixels = [0 for i in range(len(images))]
    for idx, image in enumerate(images):
        images_pixels[idx] = []
        for row in image:
            for pixel in row:
                images_pixels[idx].append(pixel)

    return images, images_pixels


# c)

# Load pixels from images
images_t, images_pixels_t = load_images('data-processed/training/')
images_te, images_pixels_te = load_images('data-processed/test/')

# Design matrix
design_matrix_t = pd.DataFrame(images_pixels_t)

# Plot Eigenvalues
# pca_result.head(150).plot.scatter(x=6, y=1)
# plt.show()

pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(images_pixels_t)

# fig = plt.figure(figsize=(20, 6))
# for i in range(12):
#     ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
#     ax.imshow(pca.components_[i].reshape(32, 32),
#               cmap=plt.cm.bone)
# plt.show()

# Interpretation:
# - first 4 pc's are lighting conditions (dark, left, bright, right)
# - other images show specific features of the faces

# d)

# Centralize test data
design_matrix_te = pd.DataFrame(images_pixels_te)
for col in design_matrix_te.columns:
    design_matrix_te[col] = design_matrix_te[col] - design_matrix_t[col].mean()

# Project images to eigenfaces
pca = decomposition.PCA(n_components=7, whiten=True)
pca.fit(images_pixels_t)

X_train_pca = pca.transform(images_pixels_t)
X_test_pca = pca.transform(images_pixels_te)

distances = []
min_distances = []
for test_image in X_test_pca:
    distance = []
    for train_image in X_train_pca:
        d = math.dist(test_image, train_image)
        distance.append(d)
    distances.append(distance)
    min_distances.append(np.argmin(distance))


# Display classified images

fig = plt.figure(figsize=(20, 6))
for idx, test_image in enumerate(images_te):
    ax = fig.add_subplot(2, 7, idx + 1, xticks=[], yticks=[])
    ax.imshow(test_image, cmap=plt.cm.bone)
    ax2 = fig.add_subplot(2, 7, idx + 8, xticks=[], yticks=[])
    ax2.imshow(images_t[min_distances[idx]], cmap=plt.cm.bone)
plt.show()
