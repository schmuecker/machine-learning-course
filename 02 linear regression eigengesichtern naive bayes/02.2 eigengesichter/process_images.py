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


def log(*args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time, *args)


def process_image(image):
    # Crop
    crop_top = 90
    crop_bottom = 65
    crop_left = 75
    crop_right = 75
    image_cropped = crop(image, ((crop_top, crop_bottom),
                                 (crop_left, crop_right)), copy=False)
    # Scale
    width = 32
    height = 32
    image_scaled = resize(image_cropped, (width, height), anti_aliasing=True)
    return image_scaled


def export_images(output_path: str, images: list, paths: list, filenames: list):
    # Create output folder if necessary
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    # Export images
    for idx, image in enumerate(images):
        # Get file path
        person_name = basename(paths[idx])
        path = join(output_path, person_name)
        filename = filenames[idx]
        # Create person folder
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save image
        filepath = join(path, filename)
        imsave(filepath, image, quality=100)

    log('Exported saved images to', output_path)


# Get paths
script_path = dirname(realpath(sys.argv[0]))
data_path = join(script_path, 'data/lfw_funneled/')

# Get folders of people with ≥70 images
people_folders = listdir(data_path)
people_over_70_files = []
for file in people_folders:
    path = join(data_path, file)
    if (not isfile(path)):
        files = listdir(path)
        if (len(files) >= 70):
            people_over_70_files.append(path)

log('Found', len(people_over_70_files),
    'people with ≥70 pictures.')

# ---
# Images from people with over 70 images (except for the last image)
# ---

# Get images from files
images = []
images_paths = []
images_filenames = []
for person_path in people_over_70_files:
    files = sorted(listdir(person_path))
    for file in files[:-1]:
        path = join(person_path, file)
        image = imread(path, as_gray=True)
        images.append(image)
        images_filenames.append(file)
        images_paths.append(person_path)

log('Leaving out the last image of each person, we have a total of',
    len(images), 'images.')

# Process images
images_processed = []
for image in images:
    i = process_image(image)
    images_processed.append(i)

# Export processed images to disk
output_folder = join(script_path, 'data-processed/training/')

# Delete output folder
if exists(output_folder):
    shutil.rmtree(output_folder)

# Export images
export_images(output_folder, images_processed, images_paths, images_filenames)

# ---
# Last image from people with over 70 images
# ---

# Get images from files
images = []
images_paths = []
images_filenames = []
for person_path in people_over_70_files:
    files = sorted(listdir(person_path))
    file = files[-1]
    path = join(person_path, file)
    image = imread(path, as_gray=True)
    images.append(image)
    images_filenames.append(file)
    images_paths.append(person_path)

log('Only the last image of each person, we have a total of',
    len(images), 'images.')

# Process images
images_processed = []
for image in images:
    i = process_image(image)
    images_processed.append(i)

# Export processed images to disk
output_folder = join(script_path, 'data-processed/test/')

# Delete output folder
if exists(output_folder):
    shutil.rmtree(output_folder)

# Export images
export_images(output_folder, images_processed, images_paths, images_filenames)

log(images_processed[0])

log('Done.')
