import sys
from os import listdir
from datetime import datetime
from os.path import isfile, join, dirname, realpath
from skimage import io
from skimage.util import crop
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

# Log helper with timestamp


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


# Get images from files
images = []
images_paths = []
for person_path in people_over_70_files:
    files = listdir(person_path)
    for file in files[:-1]:
        path = join(person_path, file)
        image = io.imread(path, as_gray=True)
        images.append(image)
        images_paths.append(person_path)

log('Leaving out the last image of each person, we have a total of',
    len(images), 'images.')

# Process images
images_processed = []
for image in images:
    i = process_image(image)
    images_processed.append(i)

# Put images_paths in a vertical vector
# Is this necessary or can we just use images_paths?
# a = np.asarray(images_paths)
# images_paths_v = a.reshape(len(a), 1)
log("Created list of peoples' names for", len(images_paths), "images.")

# Evaluate: Plot first 100 pictures
plt.figure(figsize=(10, 10))
for idx, image in enumerate(images_processed[:100]):
    plt.subplot(10, 10, 1 + idx), plt.imshow(image), plt.axis('off')
plt.show()

log('Done.')
