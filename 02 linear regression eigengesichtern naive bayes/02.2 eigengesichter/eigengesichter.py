import sys
from os import listdir
from os.path import isfile, join, dirname, realpath
from skimage import io

# Get paths
script_path = dirname(realpath(sys.argv[0]))
data_path = join(script_path, 'data/lfw_funneled/')

# Get folders of people with equal or more than 70 images
people_folders = listdir(data_path)
people_over_70_images = []
for f in people_folders:
    path = join(data_path, f)
    if (not isfile(path)):
        files = listdir(path)
        if (len(files) >= 70):
            people_over_70_images.append(path)

for person in people_over_70_images:
    for file in listdir(person):
        print(imread(file))
