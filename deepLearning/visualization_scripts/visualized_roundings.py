import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    if exclude_from is not None:
        sort_val = sort_val[sort_idxs]
        col = col[sort_val >= exclude_from]
    return col


include_svm = False
include_oo = True
include_nn = False
fpath = '/share/wandell/data/reith/coneMosaik/various_rounding_rounds/'
folder_paths = [f.path for f in os.scandir(fpath) if f.is_dir()]
folder_paths.sort()
folder_paths.append(folder_paths.pop(0))

fig = plt.figure()
# plt.grid(which='both')
plt.xscale('log')
plt.xlabel('contrast')
plt.ylabel('dprime')
plt.title(f"Sensor data - Comparison between various template data roundings")
for i, p in enumerate(folder_paths, start=1):
    if i <= 8:
        appendix = f' rounded to {i} decimal'
    elif i == 9:
        appendix = ' not rounded'
    csv1 = os.path.join(p, 'results.csv')
    csv_svm = os.path.join(p, 'svm_results.csv')
    fname = 'sensor_data_varous_rounding_points'
    # fname = 'sensor_data_real_mean_to_1dec_rounded_mean'

    oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by='contrast')
    nn = get_csv_column(csv1, 'nn_dprime', sort_by='contrast')
    contrasts = get_csv_column(csv1, 'contrast', sort_by='contrast')
    if include_oo:
        plt.plot(contrasts, oo, label='Ideal Observer'+appendix)
    if include_nn:
        plt.plot(contrasts, nn, label='ResNet18'+appendix)
    epsilon = 0.001
    if include_svm:
        svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by='contrast')
        svm[svm >= (svm.max()-epsilon)] = oo.max()
        plt.plot(contrasts, svm, label='Support Vector Machine'+appendix)

plt.legend(frameon=True, loc='upper left', fontsize='xx-small')
out_path = fpath
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
