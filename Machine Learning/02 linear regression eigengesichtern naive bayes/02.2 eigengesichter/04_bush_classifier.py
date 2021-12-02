import sys
import math
import pandas as pd
from os import listdir
from skimage.io import imread
from sklearn import decomposition
from os.path import join, dirname, realpath
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


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
            pixels = []
            for row in image:
                for pixel in row:
                    pixels.append(pixel)
            datapoint = {"name": person}
            for idx, pixel in enumerate(pixels):
                datapoint[idx] = pixel
            images.append(datapoint)

    return images


# Load images
images = load_images('data-processed/all/')

# Design matrix
design_matrix = pd.DataFrame(images)

y = design_matrix['name']
X = design_matrix.loc[:, 0:]

# Split into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

# PCA
pca = decomposition.PCA(n_components=7, whiten=True)
pca.fit(X_train)

# Project
X_train_proj = pca.transform(X_train)
X_test_proj = pca.transform(X_test)


# Naive Bayes


def is_bush(names):
    result = []
    for name in names:
        if(name == 'George_W_Bush'):
            result.append(1)
        else:
            result.append(-1)
    return result


def check_bush(pred, true):
    result_test_pred = is_bush(pred)
    result_test = is_bush(true)

    test_positives = 0
    test_negatives = 0
    test_true_positives = 0
    test_true_negatives = 0
    test_false_negatives = 0
    test_false_positives = 0

    for idx, name_pred in enumerate(result_test_pred):
        name = result_test[idx]
        if name == 1:
            test_positives += 1
        if name == -1:
            test_negatives += 1
        if name_pred == 1 and name == 1:
            test_true_positives += 1
        if name_pred == 1 and name == -1:
            test_false_positives += 1
        if name_pred == -1 and name == -1:
            test_true_negatives += 1
        if name_pred == -1 and name == 1:
            test_false_negatives += 1

    print('True positives:', test_true_positives, 'of', test_positives,
          'positives. Rate:', round(100*(test_true_positives / test_positives), 2), '%')
    print('False negatives:', test_false_negatives, 'of', test_positives,
          'positives. Rate:', round(100*(test_false_negatives / test_positives), 2), '%')
    print('True negatives:', test_true_negatives, 'of', test_negatives,
          'negatives. Rate:', round(100*(test_true_negatives / test_negatives), 2), '%')
    print('False positives:', test_false_positives, 'of', test_negatives,
          'negatives. Rate:', round(100*(test_false_positives / test_negatives), 2), '%')


gnb = GaussianNB()

# Test data

y_test_pred = gnb.fit(X_train_proj, y_train).predict(X_test_proj)
print('\nTest data')
check_bush(y_test_pred, y_test)

# Training data

print('\nTraining data')
y_train_pred = gnb.fit(X_train_proj, y_train).predict(X_train_proj)
check_bush(y_train_pred, y_train)
