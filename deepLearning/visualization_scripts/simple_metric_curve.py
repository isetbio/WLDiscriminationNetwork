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

include_oo = True
include_nn = True
include_svm = True

shift = True
angle = False

if shift:
    metric = 'shift'
elif angle:
    metric = 'angle'
else:
    metric = 'contrast'


folder = '/share/wandell/data/reith/redo_experiments/sensor_harmonic_phase_shift/'
fname = f'harmonic_curve_detection_{metric}'

csv1 = os.path.join(folder, 'results.csv')
csv_svm = os.path.join(folder, 'svm_results.csv')
oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
contrasts = get_csv_column(csv1, metric, sort_by=metric)

fig = plt.figure()
# plt.grid(which='both')
if include_oo:
    plt.plot(contrasts, oo, label='Ideal Observer')
if include_nn:
    plt.plot(contrasts, nn, label='ResNet18')
epsilon = 0.001
if include_svm:
    svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by='dprime_accuracy')
    svm[svm >= (svm.max()-epsilon)] = oo.max()
    plt.plot(contrasts, svm, label='Support Vector Machine')
plt.xscale('log')
plt.xlabel(metric)
plt.ylabel('dprime')
if metric != 'shift':
    plt.title(f'Frequency 1 harmonic - dprime for various {metric} values')
else:
    plt.title(f'Frequency 1 harmonic - dprime for various {metric} values')
plt.legend(frameon=True)

out_path = os.path.dirname(csv1)
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
