import os
from fnmatch import fnmatch
from glob import glob

folder = '/share/wandell/data/reith/2_class_MTF_freq_experiment/'

files = glob(f"{folder}**/*svm*", recursive=True)
for f in files:
    os.remove(f)
