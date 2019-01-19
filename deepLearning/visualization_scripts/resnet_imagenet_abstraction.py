import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


def get_csv_column(csv_path, col_name):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    return np.array(col)


csv0 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_0/results.csv/results.csv'
csv1 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_1/results.csv/results.csv'
csv2 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_2/results.csv/results.csv'
csv3 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_3/results.csv/results.csv'
csv4 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_4/results.csv/results.csv'
fname = 'network_imagenet_abstractions_dprime_curve'

oo = get_csv_column(csv0, 'optimal_observer_d_index')
nn_0 = get_csv_column(csv0, 'nn_dprime')
nn_1 = get_csv_column(csv1, 'nn_dprime')
nn_2 = get_csv_column(csv2, 'nn_dprime')
nn_3 = get_csv_column(csv3, 'nn_dprime')
nn_4 = get_csv_column(csv4, 'nn_dprime')
contrasts = get_csv_column(csv4, 'contrast')

sort_idxs = np.argsort(contrasts)
contrasts = contrasts[sort_idxs]
oo = oo[sort_idxs]
nn_0 = nn_0[sort_idxs]
nn_1 = nn_1[sort_idxs]
nn_2 = nn_2[sort_idxs]
nn_3 = nn_3[sort_idxs]
nn_4 = nn_4[sort_idxs]


fig = plt.figure()
plt.xscale('log')
plt.xlabel('contrast')
plt.ylabel('dprime')
plt.title('Frequency 1 harmonic - dprime for various contrasts')

plt.plot(contrasts, oo, label='Ideal Observer')
plt.plot(contrasts, nn_0, label='ResNet18 - all randomly initialized')
plt.plot(contrasts, nn_1, label='ResNet18 - first quarter ImageNet pretrained and frozen')
plt.plot(contrasts, nn_2, label='ResNet18 - first half ImageNet pretrained and frozen')
plt.plot(contrasts, nn_3, label='ResNet18 - first three quarters Imagenet pretrained and frozen')
plt.plot(contrasts, nn_4, label='ResNet18 - all layers except final fully connected layer ImageNet pretrained and frozen')

plt.legend(frameon=True)

out_path = os.path.dirname(csv1)
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
