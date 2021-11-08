import pathlib
import tarfile
from os.path import join
from datetime import datetime
from urllib.request import urlretrieve


def log(*args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time, *args)


# Paths
data_folder = 'data/'
filename = 'lfw-funneled.tgz'
filepath = join(data_folder, filename)

# Create data folder
pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)

# Download
log('Downloading dataset...')
urlretrieve(
    "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz", filepath)
log('Download finished.')

# Extract
log('Extracting file...')
tar = tarfile.open(filepath, 'r:gz')
tar.extractall(data_folder)
tar.close()
log('Extraction finished.')
