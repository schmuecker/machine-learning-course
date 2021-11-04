import sys
from os import listdir
from os.path import isfile, join, dirname, realpath
from skimage import io

# Get paths
script_path = dirname(realpath(sys.argv[0]))
data_path = join(script_path, 'data/lfw_funneled/')

# Get folders of people with equal or more than 70 images
people_folders = listdir(data_path)
people_over_70_files = []
for file in people_folders:
    path = join(data_path, file)
    if (not isfile(path)):
        files = listdir(path)
        if (len(files) >= 70):
            people_over_70_files.append(path)

print(people_over_70_files)

# Get images from files
people_over_70_images = []
for person_path in people_over_70_files:
    for file in listdir(person_path):
        path = join(person_path, file)
        image = io.imread(path)
        people_over_70_images.append(image)

print(people_over_70_images)
