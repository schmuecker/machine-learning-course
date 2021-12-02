import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import pandas as pd
from os import listdir
from skimage.io import imread
from sklearn import decomposition
from os.path import join, dirname, realpath
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
sns.set(style="whitegrid")


def load_images(path: str):
    script_path = dirname(realpath(sys.argv[0]))
    abs_path = join(script_path, path)

    # Load images
    person_folders = listdir(abs_path)
    images = []
    names = []
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
            names.append(person)

    return images, names


# Load images
images, names = load_images('data-processed/all/')

# Design matrix
design_matrix = pd.DataFrame(images)

# Set George_W_Bush to 1 all others to -1
y_list = [-1] * len(images)
bush_counter = 0
for i in range(len(images)):
    if images[i]['name'] == 'George_W_Bush':
        y_list[i] = 1
        bush_counter += 1
    else:
        y_list[i] = -1


y = design_matrix['name']
X = design_matrix.loc[:, 0:]

# Split into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_list, test_size=0.4, random_state=0)

# PCA
pca = decomposition.PCA(n_components=7, whiten=True)
pca.fit(X_train)

# Project
X_train_proj = pca.transform(X_train)
X_test_proj = pca.transform(X_test)


# Naive Bayes


def accuracy_score(y_true, y_pred):
    """	score = (y_true - y_pred) / len(y_true) """

    return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100, 2)


def pre_processing(df):
    """ partioning data into features and target """

    X = df.drop([df.columns[-1]], axis=1)
    y = df[df.columns[-1]]

    return X, y


class NaiveBayesClassifier():

    def calc_prior(self, features, target):
        self.prior = (features.groupby(target).apply(
            lambda x: len(x)) / self.rows).to_numpy()

        return self.prior

    def calc_statistics(self, features, target):
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()

        return self.mean, self.var

    def gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp((-1/2)*((x-mean)**2) / (2 * var))

        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob

    def calc_posterior(self, x):
        posteriors = []

        for i in range(self.count):
            prior = np.log(self.prior[i])
            conditional = np.sum(np.log(self.gaussian_density(i, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def fit(self, features, target):
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.feature_nums = features.shape[1]
        self.rows = features.shape[0]

        self.calc_statistics(features, target)
        self.calc_prior(features, target)

    def predict(self, features):
        preds = [self.calc_posterior(f) for f in features.to_numpy()]
        return preds


def check_bush(pred, true_items):
    result_test_pred = pred
    result_test = true_items

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


X_train_df = pd.DataFrame(X_train_proj)
y_train_df = pd.DataFrame(y_train)

x = NaiveBayesClassifier()

x.fit(X_train_df, y_train)
predictions = x.predict(pd.DataFrame(X_test_proj))

check_bush(predictions, y_test)

# Test data

# y_test_pred = gnb.fit(X_train_proj, y_train).predict(X_test_proj)
# print('\nTest data')
# check_bush(y_test_pred, y_test)

# # Training data

# print('\nTraining data')
# y_train_pred = gnb.fit(X_train_proj, y_train).predict(X_train_proj)
# check_bush(y_train_pred, y_train)
