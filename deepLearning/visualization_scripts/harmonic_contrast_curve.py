import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


def get_csv_column(csv_path, col_name, sort_by=None):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    return col



csv1 = '/share/wandell/data/reith/2_class_MTF_freq_experiment/frequency_6/results.csv'
fname = 'harmonic_contrast_curve'

oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by='contrast')
nn = get_csv_column(csv1, 'nn_dprime', sort_by='contrast')
contrasts = get_csv_column(csv1, 'contrast', sort_by='contrast')

fig = plt.figure()
# plt.grid(which='both')
plt.xscale('log')
plt.xlabel('contrast')
plt.ylabel('dprime')
plt.title('Frequency 6 harmonic - dprime for various contrast values')

plt.plot(contrasts, oo, label='Ideal Observer')
plt.plot(contrasts, nn, label='ResNet18')
plt.legend(frameon=True)

out_path = os.path.dirname(csv1)
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
