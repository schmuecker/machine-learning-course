# Exercise 2.2 - Eigengesichter

This exercise is implemented to run locally on your machine.

## Preparation

Make sure you have `pipenv` installed.
Run `pipenv install` in this directory to download and install dependencies.

## Scripts

Make sure to run the python scripts in the following order.

**`01_download_data.py`**

Downloads a dataset of images (223MB) and extracts it into the `data` folder.

**`02_process_images.py`**

Selects images, changes them into grayscale, crops and scales them. The result is saved in the `data-processed` folder, separated into a training and test set as described in the exercise 2.2b.

**`03_analyse_images.py`**

Reads images from the `data-processed` folder and analyses them.
