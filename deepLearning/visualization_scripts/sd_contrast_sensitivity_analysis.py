from glob import glob
import pandas as pd
import os


# get csv file paths
data_folder = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sd_experiment'
cs_files = glob(os.path.join(data_folder, rf'**\*sd_seed*contrast_sensitivity.csv'))

# read csv files
data = []
for f in cs_files:
    data.append(pd.read_csv(f))

# remove csv files that aren't complete
cleaned_data = []
max_len = -1
for d in data:
    if len(d) > max_len:
        max_len = len(d)
for d in data:
    if len(d) == max_len:
        cleaned_data.append(d)
data = cleaned_data
print('done')